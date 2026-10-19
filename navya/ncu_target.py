"""
p1/ncu_target.py — OPTIONAL (only if `ncu` works on your GPU session). One profiled launch per variant.

Warm-up launches happen outside the profiled region; only the launches between profiler start/stop
are captured. Run:

  ncu --profile-from-start off --set full -o results/p1_ncu python p1/ncu_target.py
  ncu --import results/p1_ncu.ncu-rep --page details --csv > results/p1_ncu_details.csv

If ncu prints ERR_NVGPUCTRPERM (no permission for GPU performance counters), hardware counters are blocked
on this platform: skip this file and say so in the paper's methodology.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cupy as cp                                   # noqa: E402
from navya.kernels import SADKernels                      # noqa: E402

H, W, BLOCK = 540, 960, 256
K = SADKernels()
rng = np.random.default_rng(0)
A = cp.asarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
B = cp.asarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
variants = [v for v in K.available() if v != "v5_strided"]

for _ in range(3):                                  # warm-up (not profiled)
    for v in variants:
        K.sad(A, B, v, BLOCK)
    K.sad_strided(A, B, H, W, 4, 0, BLOCK)
cp.cuda.Device().synchronize()

cp.cuda.profiler.start()                            # ---- profiled region
for v in variants:
    K.sad(A, B, v, BLOCK)
for s in (2, 4):
    K.sad_strided(A, B, H, W, s, 0, BLOCK)
cp.cuda.Device().synchronize()
cp.cuda.profiler.stop()                             # ----
print("profiled one launch per variant:", variants + ["v5_strided_s2", "v5_strided_s4"])
