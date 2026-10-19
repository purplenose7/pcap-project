"""
gate_check.py -- run this FIRST in a Kaggle (or Colab) GPU notebook with Internet ON.
Prints PASS / FAIL / SKIP for everything the project depends on, with the fix for each failure.
Usage in a notebook cell:   !python gate_check.py
"""
import os, shutil, subprocess, sys, tempfile, traceback

RESULTS = []


def check(name):
    def deco(fn):
        try:
            msg = fn()
            RESULTS.append((name, "PASS", msg or ""))
        except Skip as e:
            RESULTS.append((name, "SKIP", str(e)))
        except Exception as e:                       # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        return fn
    return deco


class Skip(Exception):
    pass


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)


@check("GPU visible (nvidia-smi)")
def _():
    r = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if r.returncode != 0 or not r.stdout.strip():
        raise Skip("no GPU attached -> set Accelerator to GPU and restart the session")
    return r.stdout.strip().replace("\n", " | ")


@check("PyTorch can run a kernel on this GPU")
def _():
    import torch
    if not torch.cuda.is_available():
        raise Skip("torch.cuda not available")
    cc = torch.cuda.get_device_capability(0)
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception as e:                           # P100 + cu128 wheels -> "no kernel image"
        v = torch.__version__.split("+")[0]
        raise RuntimeError(
            f"{e} | capability {cc}. If this is a P100, reinstall the SAME version from the cu126 index:\n"
            f"   pip install torch=={v} torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall\n"
            f"   then RESTART the session.")
    return f"torch {torch.__version__}, capability {cc}, matmul ok"


@check("CuPy RawKernel (NVRTC) compiles and matches NumPy")
def _():
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("cupy missing -> pip install cupy-cuda12x")
    src = r'''
    extern "C" __global__ void sad(const unsigned char* a, const unsigned char* b,
                                   unsigned long long n, unsigned long long* out) {
        unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
        if (i < n) { int d = (int)a[i] - (int)b[i]; atomicAdd(out, (unsigned long long)(d < 0 ? -d : d)); }
    }'''
    k = cp.RawKernel(src, "sad")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, 960 * 540 * 3, dtype=np.uint8); b = rng.integers(0, 256, a.size, dtype=np.uint8)
    out = cp.zeros(1, dtype=cp.uint64)
    n = a.size
    k(((n + 255) // 256,), (256,), (cp.asarray(a), cp.asarray(b), cp.uint64(n), out))
    ref = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum())
    got = int(out.get()[0])
    if got != ref:
        raise RuntimeError(f"GPU {got} != NumPy {ref}")
    return f"exact match ({ref})"


@check("GPU JPEG decode (torchvision nvjpeg)")
def _():
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.io import decode_jpeg, read_file
    if not torch.cuda.is_available():
        raise Skip("no GPU")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jpg")
        Image.fromarray((np.random.rand(540, 960, 3) * 255).astype("uint8")).save(p, quality=90)
        img = decode_jpeg(read_file(p), device="cuda")
    return f"decoded to {tuple(img.shape)} on {img.device}"


@check("NVML power readout (for energy numbers)")
def _():
    try:
        import pynvml
    except ImportError:
        raise RuntimeError("pip install nvidia-ml-py")
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mw = pynvml.nvmlDeviceGetPowerUsage(h)
    return f"{mw / 1000:.1f} W"


@check("torch.profiler can record CUDA activity (stream-overlap timeline)")
def _():
    import torch
    from torch.profiler import ProfilerActivity, profile
    if not torch.cuda.is_available():
        raise Skip("no GPU")
    x = torch.randn(2048, 2048, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        (x @ x).sum().item()
        torch.cuda.synchronize()
    n = sum(1 for e in prof.events() if getattr(e, "device_type", None) is not None
            and "CUDA" in str(e.device_type))
    if n == 0:
        raise RuntimeError("profiler recorded no CUDA events")
    return f"{n} CUDA events recorded"


@check("Nsight tools (ncu / nsys) present")
def _():
    found = {t: shutil.which(t) for t in ("ncu", "nsys")}
    if not any(found.values()):
        raise Skip("not installed -> plan assumes NO ncu/nsys; use CUDA events + torch.profiler instead")
    return str(found) + " (presence only; hardware counters may still be blocked -- test with a tiny kernel)"


@check("Ultralytics imports and YOLOv8s loads")
def _():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("pip install ultralytics  (and pin the version in requirements.txt)")
    m = YOLO("yolov8s.pt")                           # downloads weights if Internet is on
    return f"ok, {sum(p.numel() for p in m.model.parameters()) / 1e6:.1f}M params"


@check("Host RAM and working-disk headroom")
def _():
    import shutil as sh_
    ram = sh("free -g | awk '/Mem:/ {print $2\" GB total, \"$7\" GB available\"}'").stdout.strip()
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    free = sh_.disk_usage(work).free / 1e9
    return f"RAM {ram}; free disk at {work}: {free:.0f} GB"


print("\n==== GATE CHECK ====")
for name, status, msg in RESULTS:
    print(f"[{status}] {name}\n        {msg}")
fails = [r for r in RESULTS if r[1] == "FAIL"]
print(f"\n{len(fails)} FAIL, {sum(r[1]=='SKIP' for r in RESULTS)} SKIP, {sum(r[1]=='PASS' for r in RESULTS)} PASS")
sys.exit(1 if fails else 0)
