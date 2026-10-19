"""
p1/fidelity.py — Day 4. How well does the cheap subsampled score (v5, stride 2 or 4) track the full score?

For every consecutive frame pair it computes score_full, score_sub2, score_sub4 and writes:
  results/p1_scores/<sequence>.csv   frame, file, score_full, score_sub2, score_sub4   (hand to P3)
  results/p1_fidelity.csv            per sequence: Spearman and Pearson correlation of sub-scores vs full

  python p1/fidelity.py --seq-root /kaggle/input/<mini-dataset>      # every sub-folder = one sequence
  python p1/fidelity.py --seq-root .../MVI_20011 --max-frames 600    # a single sequence folder
  python p1/fidelity.py --seq-root ... --dry-run                     # no GPU (PIL decode + NumPy)

GPU mode decodes with torchvision on the GPU, as the contract requires, so P3 can use these scores directly.
Frame 0 of each sequence has no previous frame, so its scores are left empty.
"""
import argparse, io, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from navya.common import RESULTS, append_csv, run_id                                  # noqa: E402
from navya.kernels import sad_ref, sad_strided_ref, score_from_sad                     # noqa: E402


def sequences(root, min_frames=2):
    root = Path(root)
    dirs = sorted({p.parent for p in root.rglob("*.jpg")} | {p.parent for p in root.rglob("*.jpeg")})
    out = []
    for d in dirs:
        files = sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))
        if len(files) >= min_frames:
            out.append((d.name, files))
    return out


class GPUScorer:
    decoder = "gpu-nvjpeg"

    def __init__(self, block):
        import cupy as cp
        import torch
        from torchvision.io import decode_jpeg
        from navya.kernels import SADKernels
        self.cp, self.torch, self.dec, self.K, self.block = cp, torch, decode_jpeg, SADKernels(), block

    def load(self, f):
        img = self.dec(self.torch.frombuffer(bytearray(f.read_bytes()), dtype=self.torch.uint8), device="cuda")
        return img.contiguous()

    def scores(self, prev, cur):
        a, b = self.cp.asarray(prev), self.cp.asarray(cur)
        _, H, W = cur.shape
        full = score_from_sad(int(self.K.sad(a, b, "v3_shuffle", self.block).get()[0]), a.size)
        out = [full]
        for s in (2, 4):
            x, n = self.K.sad_strided(a, b, H, W, s, 1, self.block)
            out.append(score_from_sad(int(x.get()[0]), n))
        return out


class CPUScorer:
    decoder = "cpu-pil"

    def __init__(self, block):
        from PIL import Image
        self.Image = Image

    def load(self, f):
        return np.asarray(self.Image.open(io.BytesIO(f.read_bytes())).convert("RGB"))

    def scores(self, prev, cur):
        out = [score_from_sad(sad_ref(prev, cur), cur.size)]
        for s in (2, 4):
            out.append(score_from_sad(*sad_strided_ref(prev, cur, s, planar=0)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--max-seqs", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=str(RESULTS / "p1_scores"))
    ap.add_argument("--summary", default=str(RESULTS / "p1_fidelity.csv"))
    a = ap.parse_args()

    S = CPUScorer(a.block) if a.dry_run else GPUScorer(a.block)
    seqs = sequences(a.seq_root)[: a.max_seqs]
    if not seqs:
        raise SystemExit(f"no sequence folders with JPEGs under {a.seq_root}")
    rid = run_id()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary, all_rows = [], []

    for name, files in seqs:
        if a.max_frames:
            files = files[: a.max_frames]
        rows = [{"frame": 0, "file": files[0].name, "score_full": None, "score_sub2": None, "score_sub4": None}]
        prev = S.load(files[0])
        for i, f in enumerate(files[1:], start=1):
            cur = S.load(f)
            full, s2, s4 = S.scores(prev, cur)
            rows.append({"frame": i, "file": f.name, "score_full": full, "score_sub2": s2, "score_sub4": s4})
            prev = cur
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        d = df.dropna()
        all_rows.append(d)
        row = {"run_id": rid, "sequence": name, "decoder": S.decoder, "n_pairs": len(d)}
        for s in ("sub2", "sub4"):
            row[f"spearman_{s}"] = round(d["score_full"].corr(d[f"score_{s}"], method="spearman"), 4)
            row[f"pearson_{s}"] = round(d["score_full"].corr(d[f"score_{s}"], method="pearson"), 4)
            row[f"mean_abs_rel_err_{s}"] = round(float(np.mean(np.abs(d[f"score_{s}"] - d["score_full"])
                                                               / d["score_full"].clip(lower=1e-12))), 4)
        summary.append(row)
        print(f"{name}: {len(d)} pairs  spearman s2={row['spearman_sub2']}  s4={row['spearman_sub4']}")

    allp = pd.concat(all_rows)
    pooled = {"run_id": rid, "sequence": "ALL (pooled)", "decoder": S.decoder, "n_pairs": len(allp)}
    for s in ("sub2", "sub4"):
        pooled[f"spearman_{s}"] = round(allp["score_full"].corr(allp[f"score_{s}"], method="spearman"), 4)
        pooled[f"pearson_{s}"] = round(allp["score_full"].corr(allp[f"score_{s}"], method="pearson"), 4)
        pooled[f"mean_abs_rel_err_{s}"] = round(float(np.mean(np.abs(allp[f"score_{s}"] - allp["score_full"])
                                                                  / allp["score_full"].clip(lower=1e-12))), 4)
    summary.append(pooled)
    append_csv(a.summary, summary)
    print(f"\npooled over {len(allp)} pairs: spearman s2={pooled['spearman_sub2']}  s4={pooled['spearman_sub4']}")
    print(f"per-frame scores -> {out_dir}/   summary -> {a.summary}")


if __name__ == "__main__":
    main()
