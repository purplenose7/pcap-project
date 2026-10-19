"""
p1/common.py — helpers shared by all P1 scripts: backend selection, timing, GPU info, CSV output.
"""
import csv, datetime, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Where this session writes. In notebooks set P1_RESULTS=/kaggle/working/out so each session gets its own
# folder (pushed to results/p1_runs/<name>/ in GitHub); merge_runs.py combines them later.
RESULTS = Path(os.environ.get("P1_RESULTS", REPO / "results"))

# Spec peak DRAM bandwidth (GB/s), used only if it can't be computed from device attributes.
PEAK_GBPS_FALLBACK = {"P100": 732.0, "T4": 320.0, "V100": 900.0, "A100": 1555.0, "L4": 300.0}

# Per-compute-capability SM limits, used only if device attributes are missing.
SM_LIMITS_FALLBACK = {
    (6, 0): dict(max_threads_sm=2048, max_blocks_sm=32, regs_sm=65536, smem_sm=65536, max_regs_block=65536),
    (7, 5): dict(max_threads_sm=1024, max_blocks_sm=16, regs_sm=65536, smem_sm=65536, max_regs_block=65536),
    (8, 0): dict(max_threads_sm=2048, max_blocks_sm=32, regs_sm=65536, smem_sm=167936, max_regs_block=65536),
    (8, 6): dict(max_threads_sm=1536, max_blocks_sm=16, regs_sm=65536, smem_sm=102400, max_regs_block=65536),
    (8, 9): dict(max_threads_sm=1536, max_blocks_sm=24, regs_sm=65536, smem_sm=102400, max_regs_block=65536),
}


def run_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def percentiles(samples):
    a = np.asarray(samples, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def append_csv(path, rows):
    """Append dict rows to a CSV, writing the header if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def import_controller():
    """controller.py lives in the repo's code/ folder (shared with P2/P3)."""
    for p in (REPO / "code", HERE, HERE.parent):
        if (p / "controller.py").exists():
            sys.path.insert(0, str(p))
            import controller          # noqa: E402
            return controller
    raise ImportError("controller.py not found: put p1/ next to the repo's code/ folder")


def nvidia_smi_clocks():
    q = "name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(["gpu_name", "sm_mhz", "mem_mhz", "temp_c", "power_w", "util_pct"], vals))
    except Exception:                                   # noqa: BLE001
        return {}


def log_clocks(tag, rid):
    row = {"run_id": rid, "tag": tag, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    row.update(nvidia_smi_clocks())
    append_csv(RESULTS / "p1_clocks.csv", [row])
    return row


# ----------------------------------------------------------------------------- GPU description
def _attr(attrs, *names):
    for n in names:
        if n in attrs:
            return int(attrs[n])
    return None


def gpu_info():
    """Name, compute capability, SM limits and peak bandwidth (with where each number came from)."""
    import cupy as cp
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    cc = (int(props["major"]), int(props["minor"]))
    attrs = dev.attributes
    fb = SM_LIMITS_FALLBACK.get(cc, SM_LIMITS_FALLBACK[(7, 5)])
    lim, source = {}, {}
    for key, names in {
        "max_threads_sm": ("MaxThreadsPerMultiProcessor", "MaxThreadsPerMultiprocessor"),
        "max_blocks_sm": ("MaxBlocksPerMultiprocessor", "MaxBlocksPerMultiProcessor"),
        "regs_sm": ("MaxRegistersPerMultiprocessor", "MaxRegistersPerMultiProcessor"),
        "smem_sm": ("MaxSharedMemoryPerMultiprocessor", "MaxSharedMemoryPerMultiProcessor"),
        "max_regs_block": ("MaxRegistersPerBlock",),
    }.items():
        v = _attr(attrs, *names)
        lim[key], source[key] = (v, "device") if v else (fb[key], "fallback-table")
    sms = _attr(attrs, "MultiProcessorCount") or int(props.get("multiProcessorCount", 0))
    clk_khz = _attr(attrs, "MemoryClockRate") or int(props.get("memoryClockRate", 0) or 0)
    bus_bits = _attr(attrs, "GlobalMemoryBusWidth") or int(props.get("memoryBusWidth", 0) or 0)
    if clk_khz and bus_bits:
        peak, peak_src = 2.0 * clk_khz * 1e3 * (bus_bits / 8) / 1e9, "computed: 2 x mem clock x bus width"
    else:
        peak = next((v for k, v in PEAK_GBPS_FALLBACK.items() if k in name), float("nan"))
        peak_src = "spec-table fallback"
    return {"name": name, "cc": cc, "sms": sms, "limits": lim, "limit_source": source,
            "peak_gbps": round(peak, 1), "peak_source": peak_src}


def write_env(path, extra=None):
    info = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("cupy", "torch", "torchvision"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                               # noqa: BLE001
            info[mod] = None
    try:
        import cupy as cp
        info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        info["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        g = gpu_info()
        info.update({"gpu": g["name"], "cc": g["cc"], "sms": g["sms"], "peak_gbps": g["peak_gbps"],
                     "peak_source": g["peak_source"], "sm_limits": g["limits"], "sm_limit_source": g["limit_source"]})
    except Exception as e:                              # noqa: BLE001
        info["gpu"] = f"unavailable ({type(e).__name__})"
    info.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2, default=str))
    return info


# ----------------------------------------------------------------------------- backends
class Backend:
    """GPU (CuPy kernels + CUDA-event timing) or dry-run (NumPy emulation + wall-clock timing)."""

    def __init__(self, dry_run=False):
        self.dry = dry_run
        if dry_run:
            from navya.emu import EmuKernels
            self.xp, self.k, self.label = np, EmuKernels(), "cpu-emulation"
        else:
            import cupy as cp
            from navya.kernels import SADKernels
            self.xp, self.k = cp, SADKernels()
            self.label = gpu_info()["name"]

    def to_int(self, x):
        return int(x.get()[0]) if hasattr(x, "get") else int(x[0])

    def sync(self):
        if not self.dry:
            self.xp.cuda.Device().synchronize()

    def time_us(self, fn, warmup, iters):
        """Per-call time in microseconds. GPU: CUDA events around each call (GPU time, not host time)."""
        for _ in range(warmup):
            fn()
        self.sync()
        out = []
        if self.dry:
            for _ in range(iters):
                t0 = time.perf_counter(); fn(); out.append((time.perf_counter() - t0) * 1e6)
            return out
        cp = self.xp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        for _ in range(iters):
            start.record(); fn(); end.record(); end.synchronize()
            out.append(cp.cuda.get_elapsed_time(start, end) * 1e3)
        return out
