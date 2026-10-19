"""
p1/bench_kernels.py — Day 3. Times every variant x frame size x block size and writes results/kernel_bench.csv.

  python p1/bench_kernels.py --tag p100-run1            # full grid (~10-20 min on a P100/T4)
  python p1/bench_kernels.py --quick --tag smoke        # 1 size, 1 block — check it works first
  python p1/bench_kernels.py --dry-run --quick          # no GPU: tests the script itself

Per configuration: 20 warm-up launches, then 200 individually timed launches (CUDA events).
  total_*  = everything the pipeline pays: (memset for v0) + main kernel + final reduction
  main_*   = main kernel only -> used for achieved bandwidth (the final reduction is tiny)
Run it at least 3 times per GPU at different times of day (different --tag) — free-tier GPUs are shared.
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from navya.common import RESULTS, Backend, append_csv, gpu_info, log_clocks, percentiles, run_id, write_env  # noqa: E402
from navya.occupancy import occupancy                                                                 # noqa: E402

SIZES = [(360, 640), (540, 960), (720, 1280), (1080, 1920)]
BLOCKS = [128, 256, 512]
STRIDES = [2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--variants", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--out", default=str(RESULTS / "kernel_bench.csv"))
    a = ap.parse_args()

    be = Backend(dry_run=a.dry_run)
    xp, K = be.xp, be.k
    rid = run_id()
    if a.dry_run:
        info = {"name": "cpu-emulation", "cc": (0, 0), "peak_gbps": float("nan"), "limits": None}
        iters, warmup = min(a.iters, 5), min(a.warmup, 1)
    else:
        info, iters, warmup = gpu_info(), a.iters, a.warmup
        write_env(RESULTS / f"p1_env_{a.tag}.json", {"tag": a.tag, "run_id": rid})
        log_clocks(f"bench-start:{a.tag}", rid)
    print(f"GPU: {info['name']}  peak DRAM bandwidth: {info['peak_gbps']} GB/s "
          f"({info.get('peak_source', '-')})")

    variants = [v for v in K.available() if v != "v5_strided"] if a.variants == "all" else a.variants.split(",")
    sizes = SIZES[1:2] if a.quick else SIZES
    blocks = [256] if a.quick else BLOCKS
    rng = np.random.default_rng(0)
    rows = []

    def record(variant, H, W, blk, total, main_, useful_bytes, dyn_smem, s=None):
        tm, t25, t75 = percentiles(total)
        mm, m25, m75 = percentiles(main_)
        gbps = useful_bytes / (mm * 1e-6) / 1e9 if mm > 0 else float("nan")
        attrs = K.attributes(variant)
        occ = (occupancy(K, variant, blk, dyn_smem, info) if not a.dry_run
               else {"theo_occ": float("nan"), "limiter": "-", "method": "-"})
        rows.append({
            "run_id": rid, "tag": a.tag, "gpu": info["name"],
            "variant": variant if s is None else f"{variant}_s{s}",
            "H": H, "W": W, "block": blk, "elements": H * W * 3,
            "total_median_us": round(tm, 3), "total_p25_us": round(t25, 3), "total_p75_us": round(t75, 3),
            "main_median_us": round(mm, 3), "main_p25_us": round(m25, 3), "main_p75_us": round(m75, 3),
            "useful_bytes": useful_bytes, "gbps": round(gbps, 2),
            "pct_peak": round(100 * gbps / info["peak_gbps"], 1) if info["peak_gbps"] == info["peak_gbps"] else "",
            "regs": attrs["num_regs"], "smem_static": attrs["shared_size_bytes"], "smem_dynamic": dyn_smem,
            "local_bytes": attrs["local_size_bytes"],
            "theo_occ": occ["theo_occ"], "occ_limiter": occ["limiter"], "occ_method": occ["method"],
            "iters": iters,
        })
        print(f"  {rows[-1]['variant']:17s} {H:4d}x{W:<4d} block={blk:4d}  total {tm:8.2f} us  "
              f"main {mm:8.2f} us  {gbps:7.1f} GB/s  occ={occ['theo_occ']}")

    for H, W in sizes:
        A = xp.asarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
        B = xp.asarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
        n = H * W * 3
        print(f"\n== {H}x{W} ({n / 1e6:.2f} M elements, {2 * n / 1e6:.1f} MB read per score) ==")
        for v in variants:
            for blk in blocks:
                if a.dry_run:
                    f = lambda: K.sad(A, B, v, blk)                                   # noqa: E731
                    t = be.time_us(f, warmup, iters)
                    record(v, H, W, blk, t, t, 2 * n, blk * 4)
                    continue
                ws = K.workspace(n, blk, v)

                def total_call():
                    K.sad(A, B, v, blk, ws=ws)

                def main_call():
                    K.launch_main(A, B, v, blk, ws)
                record(v, H, W, blk, be.time_us(total_call, warmup, iters),
                       be.time_us(main_call, warmup, iters), 2 * n, 0 if v == "v0_atomic" else blk * 4)
        for s in STRIDES:
            for blk in blocks:
                n_s = ((H + s - 1) // s) * ((W + s - 1) // s) * 3
                if a.dry_run:
                    f = lambda: K.sad_strided(A, B, H, W, s, 0, blk)                  # noqa: E731
                    t = be.time_us(f, warmup, iters)
                    record("v5_strided", H, W, blk, t, t, 2 * n_s, blk * 4, s=s)
                    continue
                ws = K.workspace(n, blk, "v5_strided", H, W, s)

                def total_s():
                    K.sad_strided(A, B, H, W, s, 0, blk, ws=ws)

                def main_s():
                    K.launch_main(A, B, "v5_strided", blk, ws, H=H, W=W, s=s, planar=0)
                record("v5_strided", H, W, blk, be.time_us(total_s, warmup, iters),
                       be.time_us(main_s, warmup, iters), 2 * n_s, blk * 4, s=s)
        be.sync()

    if not a.dry_run:
        log_clocks(f"bench-end:{a.tag}", rid)
    append_csv(a.out, rows)
    print(f"\nwrote {len(rows)} rows -> {a.out}")

    # Quick reading aid — the paper's claims come from the CSV, not from this printout.
    import pandas as pd
    df = pd.DataFrame(rows)
    piv = df[df.block == (256 if 256 in blocks else blocks[0])].pivot_table(
        index=["H", "W"], columns="variant", values="main_median_us")
    print("\nmain-kernel median (us), block 256:\n", piv.round(2).to_string())
    if {"v1_interleaved", "v2_sequential"} <= set(piv.columns):
        print("\nDivergence evidence: v1 (divergent) / v2 (non-divergent) time, same memory traffic:")
        print((piv["v1_interleaved"] / piv["v2_sequential"]).round(2).to_string())


if __name__ == "__main__":
    main()
