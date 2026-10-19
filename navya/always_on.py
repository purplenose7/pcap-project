"""
p1/always_on.py — Day 4. What one frame costs when it is NOT sent to the detector (the "skip floor").

Per frame:  read JPEG bytes (CPU) -> GPU decode (nvjpeg, includes uploading the compressed bytes)
            -> estimator kernel -> score read back to the host -> controller.step()

Two measurements:
  isolated  — each stage timed on its own, synchronising before and after (what each stage costs)
  pipelined — the whole loop with no extra synchronisation (what a skipped frame really costs: t_on)

  python p1/always_on.py --frames-dir /kaggle/input/<mini>/MVI_20011 --tag p100-run1
  python p1/always_on.py --frames-dir ... --estimator v5s4 --tag p100-v5s4
  python p1/always_on.py --frames-dir ... --dry-run          # no GPU: PIL decode + NumPy, tests the script

Writes results/always_on.csv (one row per mode x stage).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from navya.common import RESULTS, append_csv, import_controller, log_clocks, percentiles, run_id, write_env  # noqa: E402
from navya.kernels import sad_ref, sad_strided_ref, score_from_sad                                     # noqa: E402

STAGES = ["read", "decode", "estimate", "readback", "controller"]


def list_frames(d):
    files = sorted(list(Path(d).glob("*.jpg")) + list(Path(d).glob("*.jpeg")))
    if len(files) < 2:
        raise SystemExit(f"need >= 2 JPEGs in {d}")
    return files


class GPUPath:
    def __init__(self, estimator, block):
        import cupy as cp
        import torch
        from torchvision.io import decode_jpeg
        from navya.kernels import SADKernels
        self.cp, self.torch, self.decode_jpeg = cp, torch, decode_jpeg
        self.K, self.est, self.block = SADKernels(), estimator, block
        self.name = torch.cuda.get_device_name(0)

    def sync(self):
        self.torch.cuda.synchronize()

    def read(self, f):
        return self.torch.frombuffer(bytearray(f.read_bytes()), dtype=self.torch.uint8)

    def decode(self, data):
        img = self.decode_jpeg(data, device="cuda")         # CHW uint8 on the GPU
        return img if img.is_contiguous() else img.contiguous()

    def estimate(self, prev, cur):
        a, b = self.cp.asarray(prev), self.cp.asarray(cur)   # zero-copy views of the torch tensors
        _, H, W = cur.shape
        if self.est.startswith("v5s"):
            out, n = self.K.sad_strided(a, b, H, W, int(self.est[3:]), 1, self.block)
            return out, n
        return self.K.sad(a, b, self.est, self.block), a.size

    def readback(self, x):
        out, n = x
        return score_from_sad(int(out.get()[0]), n)

    def gpu_event_us(self, fn):
        """GPU-side time of fn via CUDA events (complements host wall time)."""
        s, e = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(); e.record(); e.synchronize()
        return r, s.elapsed_time(e) * 1e3


class CPUPath:
    """Dry-run: same structure, PIL + NumPy, so the script can be tested anywhere."""
    name = "cpu-dry-run"

    def __init__(self, estimator, block):
        from PIL import Image
        import io
        self.Image, self.io, self.est = Image, io, estimator

    def sync(self):
        pass

    def read(self, f):
        return f.read_bytes()

    def decode(self, data):
        return np.asarray(self.Image.open(self.io.BytesIO(data)).convert("RGB"))   # HWC

    def estimate(self, prev, cur):
        if self.est.startswith("v5s"):
            return sad_strided_ref(prev, cur, int(self.est[3:]), planar=0)
        return sad_ref(prev, cur), cur.size

    def readback(self, x):
        return score_from_sad(*x)

    def gpu_event_us(self, fn):
        return fn(), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="folder with ONE sequence of consecutive JPEG frames")
    ap.add_argument("--n-frames", type=int, default=1000, help="frames to time (cycles the folder if shorter)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--estimator", default="v3_shuffle",
                    help="v0_atomic|v1_interleaved|v2_sequential|v3_shuffle|v4_vec4|v5s2|v5s4")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "always_on.csv"))
    a = ap.parse_args()

    files = list_frames(a.frames_dir)
    P = CPUPath(a.estimator, a.block) if a.dry_run else GPUPath(a.estimator, a.block)
    ctl = import_controller()
    rid = run_id()
    if not a.dry_run:
        write_env(RESULTS / f"p1_env_always_on_{a.tag}.json", {"tag": a.tag, "run_id": rid})
        log_clocks(f"always_on-start:{a.tag}", rid)

    first = P.decode(P.read(files[0]))
    H, W = (first.shape[1], first.shape[2]) if not a.dry_run else first.shape[:2]
    controller = ctl.DecoupledController(t_low=0.01, t_high=0.03, k_px=16, refresh=50, long_side=max(H, W))
    order = [files[(i + 1) % len(files)] for i in range(a.warmup + a.n_frames)]
    print(f"{P.name}: {H}x{W}, {len(files)} frames in folder, timing {a.n_frames} (+{a.warmup} warm-up), "
          f"estimator={a.estimator}")

    # ---------------- isolated: sync before/after each stage
    iso = {s: [] for s in STAGES}
    gpu_t = {"decode": [], "estimate": []}
    prev = first
    for i, f in enumerate(order):
        rec = i >= a.warmup
        P.sync(); t = time.perf_counter(); data = P.read(f); P.sync(); t_read = time.perf_counter() - t
        t = time.perf_counter(); cur, g_dec = P.gpu_event_us(lambda: P.decode(data)); P.sync()
        t_dec = time.perf_counter() - t
        t = time.perf_counter(); x, g_est = P.gpu_event_us(lambda: P.estimate(prev, cur)); P.sync()
        t_est = time.perf_counter() - t
        t = time.perf_counter(); score = P.readback(x); t_rb = time.perf_counter() - t
        t = time.perf_counter(); controller.step(score, None); t_ctl = time.perf_counter() - t
        if rec:
            for s, v in zip(STAGES, (t_read, t_dec, t_est, t_rb, t_ctl)):
                iso[s].append(v * 1e6)
            gpu_t["decode"].append(g_dec); gpu_t["estimate"].append(g_est)
        prev = cur

    # ---------------- pipelined: no extra syncs (readback is the only, inherent, sync)
    pipe = []
    prev = first
    for i, f in enumerate(order):
        t = time.perf_counter()
        cur = P.decode(P.read(f))
        score = P.readback(P.estimate(prev, cur))
        controller.step(score, None)
        if i >= a.warmup:
            pipe.append((time.perf_counter() - t) * 1e6)
        prev = cur

    rows = []
    base = {"run_id": rid, "tag": a.tag, "gpu": P.name, "estimator": a.estimator, "block": a.block,
            "frames_dir": Path(a.frames_dir).name, "H": H, "W": W, "n": a.n_frames}
    for s in STAGES:
        m, p25, p75 = percentiles(iso[s])
        g = np.nanmedian(gpu_t[s]) if s in gpu_t and len(gpu_t[s]) and not a.dry_run else ""
        rows.append({**base, "mode": "isolated", "stage": s, "median_us": round(m, 2), "p25_us": round(p25, 2),
                     "p75_us": round(p75, 2), "gpu_event_median_us": round(float(g), 2) if g != "" else ""})
    m, p25, p75 = percentiles(pipe)
    rows.append({**base, "mode": "pipelined", "stage": "total_per_frame", "median_us": round(m, 2),
                 "p25_us": round(p25, 2), "p75_us": round(p75, 2), "gpu_event_median_us": ""})
    append_csv(a.out, rows)
    if not a.dry_run:
        log_clocks(f"always_on-end:{a.tag}", rid)

    print("\nisolated stage cost (median us):")
    for r in rows[:-1]:
        extra = f"   [GPU events: {r['gpu_event_median_us']} us]" if r["gpu_event_median_us"] != "" else ""
        print(f"  {r['stage']:11s} {r['median_us']:10.2f}{extra}")
    print(f"  {'sum':11s} {sum(r['median_us'] for r in rows[:-1]):10.2f}")
    print(f"pipelined per skipped frame (t_on): {rows[-1]['median_us']:.2f} us  "
          f"-> skip-floor ceiling ~ {1e6 / rows[-1]['median_us']:.0f} frames/s if no inference ran")
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
