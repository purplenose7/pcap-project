"""
p1/make_test_frames.py — synthetic JPEG sequences (quiet -> busy -> quiet) for testing before real data exists.

  python p1/make_test_frames.py --out /kaggle/working/test_frames --n-seqs 2 --frames 300

Each sequence: static background with sensor noise (quiet), then several moving rectangles (busy),
then quiet again. Not a substitute for UA-DETRAC — only for checking that scripts run.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_sequence(out, n, H, W, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.stack([(xx / W * 180 + 40), (yy / H * 160 + 50), np.full((H, W), 90.0)], -1)
    objs = [dict(x=rng.uniform(0, W), y=rng.uniform(0.2 * H, 0.8 * H), w=rng.uniform(30, 120),
                 h=rng.uniform(20, 70), vx=rng.uniform(2, 8), c=rng.uniform(0, 255, 3)) for _ in range(6)]
    out.mkdir(parents=True, exist_ok=True)
    for f in range(n):
        img = bg + rng.normal(0, 2.0, bg.shape)                      # sensor noise
        busy = n // 3 <= f < 2 * n // 3
        if busy:
            for o in objs:
                o["x"] = (o["x"] + o["vx"]) % W
                x0, y0 = int(o["x"]), int(o["y"])
                img[y0:y0 + int(o["h"]), x0:x0 + int(o["w"])] = o["c"]
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out / f"img{f + 1:05d}.jpg", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    for s in range(a.n_seqs):
        d = Path(a.out) / f"SYNTH_{s:02d}"
        make_sequence(d, a.frames, a.height, a.width, seed=s)
        print(f"wrote {a.frames} frames -> {d}")


if __name__ == "__main__":
    main()
