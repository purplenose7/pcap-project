"""
p1/kernels.py — motion-score (SAD) estimator kernels, V0–V5, compiled at runtime with CuPy.

score(N) = SAD(frame_N, frame_{N-1}) / (num_elements * 255), integer-accumulated.

Variants (each one demonstrates one CUDA idea):
  v0_atomic       every thread atomicAdds into ONE global counter  -> global-memory contention
  v1_interleaved  shared-memory tree, `tid % (2*stride)`           -> divergent warps, bank conflicts
  v2_sequential   shared-memory tree, `tid < stride`               -> divergence removed, conflict-free
  v3_shuffle      2 loads/thread + warp shuffle for the last 32    -> fewer idle threads, no sync in last warp
  v4_vec4         one 32-bit load = 4 bytes per thread             -> wider memory transactions
  v4_vsadu4       like v4 but uses the __vsadu4 SIMD intrinsic     -> optional; only if it compiles and is faster
  v5_strided      one thread per pixel on a stride-s grid          -> less work per frame (approximate score)

All variants except v0 write one partial sum per block, then `reduce_partials` sums them.
Integers throughout, so the result must equal NumPy EXACTLY (see test_kernels.py).
"""
import numpy as np

try:
    import cupy as cp
except ImportError:          # lets the NumPy reference functions be imported on CPU-only machines
    cp = None

CUDA_SRC = r'''
extern "C" {

__device__ __forceinline__ unsigned int absdiff(unsigned int x, unsigned int y) {
    return x > y ? x - y : y - x;
}

// Block-wide sum; result is valid in thread 0 only. Requires blockDim.x to be a power of two >= 64.
__device__ __forceinline__ unsigned int block_sum(unsigned int v, unsigned int* s) {
    unsigned int tid = threadIdx.x;
    s[tid] = v;
    __syncthreads();
    for (unsigned int st = blockDim.x / 2; st > 32; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid < 32) {
        v = s[tid] + s[tid + 32];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// V0: naive — global atomics only (out must be zeroed before the launch)
__global__ void sad_atomic(const unsigned char* a, const unsigned char* b,
                           unsigned long long n, unsigned long long* out) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, (unsigned long long)absdiff(a[i], b[i]));
}

// V1: shared memory, interleaved addressing (divergent warps)
__global__ void sad_interleaved(const unsigned char* a, const unsigned char* b,
                                unsigned long long n, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    unsigned int tid = threadIdx.x;
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + tid;
    s[tid] = (i < n) ? absdiff(a[i], b[i]) : 0u;
    __syncthreads();
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// V2: shared memory, sequential addressing (no divergence inside active warps, conflict-free)
__global__ void sad_sequential(const unsigned char* a, const unsigned char* b,
                               unsigned long long n, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    unsigned int tid = threadIdx.x;
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + tid;
    s[tid] = (i < n) ? absdiff(a[i], b[i]) : 0u;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// V3: two elements per thread on load + warp shuffle for the last warp
__global__ void sad_shuffle(const unsigned char* a, const unsigned char* b,
                            unsigned long long n, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x * 2 + threadIdx.x;
    unsigned int v = 0;
    if (i < n) v += absdiff(a[i], b[i]);
    if (i + blockDim.x < n) v += absdiff(a[i + blockDim.x], b[i + blockDim.x]);
    v = block_sum(v, s);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// V4: one 32-bit word (4 bytes) per thread; n4 = number of 32-bit words
__global__ void sad_vec4(const unsigned int* a, const unsigned int* b,
                         unsigned long long n4, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned int v = 0;
    if (i < n4) {
        unsigned int x = a[i], y = b[i];
        v = absdiff(x & 0xFFu, y & 0xFFu)
          + absdiff((x >> 8) & 0xFFu, (y >> 8) & 0xFFu)
          + absdiff((x >> 16) & 0xFFu, (y >> 16) & 0xFFu)
          + absdiff(x >> 24, y >> 24);
    }
    v = block_sum(v, s);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// V5: subsampled — one thread per pixel on a stride-s grid, 3 channels.
// planar=0: HWC (interleaved RGB, e.g. NumPy/OpenCV).  planar=1: CHW (e.g. torchvision decode_jpeg).
__global__ void sad_strided(const unsigned char* a, const unsigned char* b,
                            int H, int W, int st, int planar, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    int Ws = (W + st - 1) / st, Hs = (H + st - 1) / st;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    unsigned int v = 0;
    if (idx < (long long)Ws * Hs) {
        int y = (int)(idx / Ws) * st, x = (int)(idx % Ws) * st;
        if (planar) {
            long long plane = (long long)H * W, p = (long long)y * W + x;
            v = absdiff(a[p], b[p]) + absdiff(a[p + plane], b[p + plane])
              + absdiff(a[p + 2 * plane], b[p + 2 * plane]);
        } else {
            long long p = ((long long)y * W + x) * 3;
            v = absdiff(a[p], b[p]) + absdiff(a[p + 1], b[p + 1]) + absdiff(a[p + 2], b[p + 2]);
        }
    }
    v = block_sum(v, s);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// Final pass: ONE block of 1024 threads sums all block partials (overwrites *out).
__global__ void reduce_partials(const unsigned int* partial, unsigned int m,
                                unsigned long long* out) {
    __shared__ unsigned long long s[1024];
    unsigned int tid = threadIdx.x;
    unsigned long long acc = 0;
    for (unsigned int j = tid; j < m; j += blockDim.x) acc += partial[j];
    s[tid] = acc;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *out = s[0];
}

}  // extern "C"
'''

# Optional SIMD-intrinsic variant, compiled separately so a failure can't break the main module.
CUDA_SRC_VSADU4 = r'''
extern "C" {
__device__ __forceinline__ unsigned int block_sum(unsigned int v, unsigned int* s) {
    unsigned int tid = threadIdx.x;
    s[tid] = v;
    __syncthreads();
    for (unsigned int st = blockDim.x / 2; st > 32; st >>= 1) {
        if (tid < st) s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid < 32) {
        v = s[tid] + s[tid + 32];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}
__global__ void sad_vsadu4(const unsigned int* a, const unsigned int* b,
                           unsigned long long n4, unsigned int* partial) {
    extern __shared__ unsigned int s[];
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned int v = (i < n4) ? __vsadu4(a[i], b[i]) : 0u;
    v = block_sum(v, s);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}
}
'''

KERNEL_OF = {
    "v0_atomic": "sad_atomic",
    "v1_interleaved": "sad_interleaved",
    "v2_sequential": "sad_sequential",
    "v3_shuffle": "sad_shuffle",
    "v4_vec4": "sad_vec4",
    "v4_vsadu4": "sad_vsadu4",
    "v5_strided": "sad_strided",
}
FULL_VARIANTS = ["v0_atomic", "v1_interleaved", "v2_sequential", "v3_shuffle", "v4_vec4", "v4_vsadu4"]


# ----------------------------------------------------------------------------- NumPy references
def sad_ref(a, b):
    """Exact integer SAD over all elements (any shape/layout)."""
    return int(np.abs(np.asarray(a).astype(np.int16) - np.asarray(b).astype(np.int16)).sum())


def sad_strided_ref(a, b, s, planar):
    """Exact integer SAD on the stride-s pixel grid, all 3 channels. Returns (sad, n_elements)."""
    a, b = np.asarray(a), np.asarray(b)
    if planar:                       # CHW
        sa, sb = a[:, ::s, ::s], b[:, ::s, ::s]
    else:                            # HWC
        sa, sb = a[::s, ::s, :], b[::s, ::s, :]
    return sad_ref(sa, sb), int(sa.size)


def score_from_sad(sad, n_elements):
    return float(sad) / (n_elements * 255.0)


# ----------------------------------------------------------------------------- GPU wrapper
def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


class SADKernels:
    """Compiles all kernels once; launches them on a given CuPy stream."""

    def __init__(self):
        if cp is None:
            raise ImportError("cupy is not installed (pip install cupy-cuda12x)")
        self.mod = cp.RawModule(code=CUDA_SRC)
        self.k = {name: self.mod.get_function(name) for name in
                  ["sad_atomic", "sad_interleaved", "sad_sequential", "sad_shuffle",
                   "sad_vec4", "sad_strided", "reduce_partials"]}
        self.vsadu4_error = None
        try:
            self.k["sad_vsadu4"] = cp.RawModule(code=CUDA_SRC_VSADU4).get_function("sad_vsadu4")
            self.k["sad_vsadu4"].attributes          # forces compilation now
        except Exception as e:                        # noqa: BLE001
            self.vsadu4_error = f"{type(e).__name__}: {e}"
            self.k.pop("sad_vsadu4", None)

    def available(self):
        return [v for v in FULL_VARIANTS if KERNEL_OF[v] in self.k] + ["v5_strided"]

    # -- workspace: preallocated buffers so timing excludes cudaMalloc
    def workspace(self, n_elements, block, variant, H=None, W=None, s=None):
        grid = self.grid(n_elements, block, variant, H, W, s)
        return {"out": cp.zeros(1, dtype=cp.uint64),
                "partial": cp.empty(max(grid, 1), dtype=cp.uint32),
                "grid": grid}

    @staticmethod
    def grid(n_elements, block, variant, H=None, W=None, s=None):
        if variant == "v3_shuffle":
            return (n_elements + 2 * block - 1) // (2 * block)
        if variant in ("v4_vec4", "v4_vsadu4"):
            return (n_elements // 4 + block - 1) // block
        if variant == "v5_strided":
            Ws, Hs = (W + s - 1) // s, (H + s - 1) // s
            return (Ws * Hs + block - 1) // block
        return (n_elements + block - 1) // block

    def _check(self, a, b, variant, block):
        assert a.dtype == cp.uint8 and b.dtype == cp.uint8, "frames must be uint8"
        assert a.size == b.size, "frames must have the same number of elements"
        assert a.flags.c_contiguous and b.flags.c_contiguous, "frames must be C-contiguous"
        assert _is_pow2(block) and block <= 1024, "block must be a power of two <= 1024"
        if variant in ("v3_shuffle", "v4_vec4", "v4_vsadu4", "v5_strided"):
            assert block >= 64, f"{variant} needs block >= 64 (warp-shuffle tail)"

    def launch_main(self, a, b, variant, block, ws, stream=None, H=None, W=None, s=None, planar=0):
        """Launches only the main kernel (no final reduction). Used for kernel-only timing."""
        self._check(a, b, variant, block)
        n = a.size
        k = self.k[KERNEL_OF[variant]]
        if variant == "v0_atomic":
            k((ws["grid"],), (block,), (a, b, cp.uint64(n), ws["out"]), stream=stream)
        elif variant in ("v4_vec4", "v4_vsadu4"):
            assert n % 4 == 0, "v4 needs element count divisible by 4"
            assert a.data.ptr % 4 == 0 and b.data.ptr % 4 == 0, "v4 needs 4-byte aligned frames"
            k((ws["grid"],), (block,),
              (a.reshape(-1).view(cp.uint32), b.reshape(-1).view(cp.uint32), cp.uint64(n // 4), ws["partial"]),
              shared_mem=block * 4, stream=stream)
        elif variant == "v5_strided":
            k((ws["grid"],), (block,),
              (a, b, np.int32(H), np.int32(W), np.int32(s), np.int32(planar), ws["partial"]),
              shared_mem=block * 4, stream=stream)
        else:
            k((ws["grid"],), (block,), (a, b, cp.uint64(n), ws["partial"]),
              shared_mem=block * 4, stream=stream)

    def launch_reduce(self, variant, ws, stream=None):
        if variant != "v0_atomic":
            self.k["reduce_partials"]((1,), (1024,), (ws["partial"], np.uint32(ws["grid"]), ws["out"]),
                                      stream=stream)

    def sad(self, a, b, variant="v3_shuffle", block=256, stream=None, ws=None):
        """Full-frame SAD. Returns a 1-element cupy uint64 array (call int(x.get()[0]) to read it)."""
        if ws is None:
            ws = self.workspace(a.size, block, variant)
        if variant == "v0_atomic":
            ws["out"].fill(0)
        self.launch_main(a, b, variant, block, ws, stream)
        self.launch_reduce(variant, ws, stream)
        return ws["out"]

    def sad_strided(self, a, b, H, W, s, planar, block=256, stream=None, ws=None):
        """Subsampled SAD. Returns (cupy uint64 array, number of sampled elements)."""
        if ws is None:
            grid = self.grid(None, block, "v5_strided", H, W, s)
            ws = {"out": cp.zeros(1, dtype=cp.uint64), "partial": cp.empty(max(grid, 1), dtype=cp.uint32),
                  "grid": grid}
        assert a.size == H * W * 3, "v5 expects 3-channel frames of size H*W*3"
        self.launch_main(a, b, "v5_strided", block, ws, stream, H=H, W=W, s=s, planar=planar)
        self.launch_reduce("v5_strided", ws, stream)
        n_sampled = ((H + s - 1) // s) * ((W + s - 1) // s) * 3
        return ws["out"], n_sampled

    def attributes(self, variant):
        """Registers, static shared memory and local (spill) memory per thread, from the compiled kernel."""
        k = self.k[KERNEL_OF[variant]]
        out = {}
        for key in ("num_regs", "shared_size_bytes", "local_size_bytes", "max_threads_per_block"):
            try:
                out[key] = int(getattr(k, key))
            except Exception:                         # noqa: BLE001
                out[key] = -1
        return out

    def function_ptr(self, variant):
        """Raw CUfunction pointer (for the driver occupancy API), or None if unavailable."""
        k = self.k[KERNEL_OF[variant]]
        for attr in ("kernel",):
            try:
                return int(getattr(k, attr).ptr)
            except Exception:                         # noqa: BLE001
                pass
        return None
