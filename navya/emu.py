"""
p1/emu.py — NumPy emulation of each kernel's INDEX MATH (which elements every block/thread reads).

Two uses:
  1. --dry-run mode for every P1 script, so the harness can be tested without a GPU.
  2. Coverage tests: for every variant, each element must be read exactly once
     (or, for v5, exactly the stride-s grid). This is where reduction kernels usually break.

It mirrors kernels.py line for line; if you change a kernel's indexing, change it here too.
"""
import numpy as np

from navya.kernels import FULL_VARIANTS, SADKernels


def _absdiff(a, b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32))


def full_indices(variant, n, block):
    """Global element indices read by a full-frame variant, one entry per read."""
    grid = SADKernels.grid(n, block, variant)
    tid = np.arange(block)
    blk = np.arange(grid)[:, None]
    if variant in ("v0_atomic", "v1_interleaved", "v2_sequential"):
        i = blk * block + tid
        return i[i < n]
    if variant == "v3_shuffle":
        i0 = blk * block * 2 + tid
        i1 = i0 + block
        return np.concatenate([i0[i0 < n], i1[i1 < n]])
    if variant in ("v4_vec4", "v4_vsadu4"):
        n4 = n // 4
        w = blk * block + tid
        w = w[w < n4]
        return (w[:, None] * 4 + np.arange(4)).ravel()
    raise ValueError(variant)


def strided_indices(H, W, s, planar, block):
    """Element indices read by v5_strided (3 per sampled pixel)."""
    grid = SADKernels.grid(None, block, "v5_strided", H, W, s)
    Ws, Hs = (W + s - 1) // s, (H + s - 1) // s
    idx = (np.arange(grid)[:, None] * block + np.arange(block)).ravel()
    idx = idx[idx < Ws * Hs]
    y, x = (idx // Ws) * s, (idx % Ws) * s
    if planar:
        p, plane = y * W + x, H * W
        return np.concatenate([p, p + plane, p + 2 * plane])
    p = (y * W + x) * 3
    return np.concatenate([p, p + 1, p + 2])


class EmuKernels:
    """Same interface as SADKernels, implemented with NumPy (no GPU, no timing meaning)."""
    vsadu4_error = None

    def available(self):
        return list(FULL_VARIANTS) + ["v5_strided"]

    def sad(self, a, b, variant="v3_shuffle", block=256, stream=None, ws=None):
        a, b = np.ascontiguousarray(a).ravel(), np.ascontiguousarray(b).ravel()
        if variant in ("v4_vec4", "v4_vsadu4"):
            assert a.size % 4 == 0, "v4 needs element count divisible by 4"
        idx = full_indices(variant, a.size, block)
        return np.array([int(_absdiff(a[idx], b[idx]).sum())], dtype=np.uint64)

    def sad_strided(self, a, b, H, W, s, planar, block=256, stream=None, ws=None):
        a, b = np.ascontiguousarray(a).ravel(), np.ascontiguousarray(b).ravel()
        assert a.size == H * W * 3
        idx = strided_indices(H, W, s, planar, block)
        n_sampled = ((H + s - 1) // s) * ((W + s - 1) // s) * 3
        return np.array([int(_absdiff(a[idx], b[idx]).sum())], dtype=np.uint64), n_sampled

    def attributes(self, variant):
        return {"num_regs": -1, "shared_size_bytes": -1, "local_size_bytes": -1, "max_threads_per_block": -1}

    def function_ptr(self, variant):
        return None
