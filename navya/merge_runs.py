"""
p1/merge_runs.py — combine the per-session result folders into single CSVs for analysis and figures.

Each GPU session pushes its outputs to results/p1_runs/<gpu>_<tag>/ (see README). This script stacks
every kernel_bench.csv, always_on.csv and p1_fidelity.csv found there into results/p1_merged/.

  python p1/merge_runs.py                      # reads results/p1_runs/, writes results/p1_merged/
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILES = ["kernel_bench.csv", "always_on.csv", "p1_fidelity.csv", "p1_clocks.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "p1_runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "p1_merged"))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        parts = []
        for f in sorted(runs.rglob(name)):
            df = pd.read_csv(f)
            df.insert(0, "session", f.parent.name)
            parts.append(df)
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged.to_csv(out / name, index=False)
            print(f"{name}: {len(parts)} session(s), {len(merged)} rows -> {out / name}")
        else:
            print(f"{name}: none found")


if __name__ == "__main__":
    main()
