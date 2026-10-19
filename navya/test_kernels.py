"""
p1/test_kernels.py — Day 2. Every variant must equal the NumPy integer SAD EXACTLY. Run before any timing.

  python p1/test_kernels.py                      # GPU (CuPy)
  python p1/test_kernels.py --frames-dir DIR     # also test real JPEG frames (consecutive pairs)
  python p1/test_kernels.py --dry-run            # no GPU: tests the harness + emulated index math

Exit code 0 = all passed, 1 = at least one failure (details printed).
"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from navya.common import Backend                                  # noqa: E402
from navya.kernels import sad_ref, sad_strided_ref                # noqa: E402

SIZES = [(360, 640), (540, 960), (720, 1280), (1080, 1920), (7, 11), (1, 1)]
BLOCKS = [64, 128, 256, 512, 1024]


def pairs(rng, H, W):
    a = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    yield "random", a, rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    yield "identical", a, a.copy()
    yield "max-diff", np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 255, np.uint8)


def load_real_pairs(frames_dir, n_pairs):
    from PIL import Image
    files = sorted(list(Path(frames_dir).glob("*.jpg")) + list(Path(frames_dir).glob("*.jpeg")))[: n_pairs + 1]
    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    for i in range(len(imgs) - 1):
        yield f"real:{files[i + 1].name}", imgs[i], imgs[i + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--real-pairs", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="fewer sizes/blocks")
    a = ap.parse_args()

    be = Backend(dry_run=a.dry_run)
    xp, K = be.xp, be.k
    print(f"backend: {be.label}")
    if getattr(K, "vsadu4_error", None):
        print(f"note: v4_vsadu4 unavailable ({K.vsadu4_error}) — skipped, not a failure")
    variants = [v for v in K.available() if v != "v5_strided"]
    sizes = SIZES[1:2] + SIZES[4:] if a.quick else SIZES
    blocks = [256] if a.quick else BLOCKS

    rng = np.random.default_rng(0)
    cases = [(f"{H}x{W}:{name}", A, B) for H, W in sizes for name, A, B in pairs(rng, H, W)]
    if a.frames_dir:
        cases += list(load_real_pairs(a.frames_dir, a.real_pairs))

    n_ok = n_bad = n_skip = 0
    for label, A, B in cases:
        H, W = A.shape[:2]
        ref = sad_ref(A, B)
        dA, dB = xp.asarray(np.ascontiguousarray(A)), xp.asarray(np.ascontiguousarray(B))
        for v, blk in itertools.product(variants, blocks):
            if v not in ("v0_atomic", "v1_interleaved", "v2_sequential") and blk < 64:
                continue
            if v.startswith("v4") and A.size % 4:
                n_skip += 1
                continue
            got = be.to_int(K.sad(dA, dB, v, blk))
            if got == ref:
                n_ok += 1
            else:
                n_bad += 1
                print(f"FAIL {v:15s} block={blk:4d} {label}: got {got}, expected {ref}")
        # v5: both layouts, s = 1 must equal the full SAD
        cA, cB = np.ascontiguousarray(A.transpose(2, 0, 1)), np.ascontiguousarray(B.transpose(2, 0, 1))
        dcA, dcB = xp.asarray(cA), xp.asarray(cB)
        for s, planar, blk in itertools.product((1, 2, 4), (0, 1), [b for b in blocks if b >= 64]):
            src_a, src_b = (cA, cB) if planar else (A, B)
            exp, n_exp = sad_strided_ref(src_a, src_b, s, planar)
            got_arr, n_got = K.sad_strided(dcA if planar else dA, dcB if planar else dB, H, W, s, planar, blk)
            got = be.to_int(got_arr)
            if got == exp and n_got == n_exp and (s != 1 or got == ref):
                n_ok += 1
            else:
                n_bad += 1
                print(f"FAIL v5_strided s={s} planar={planar} block={blk} {label}: got {got}/{n_got}, "
                      f"expected {exp}/{n_exp}")
        be.sync()

    print(f"\n{n_ok} passed, {n_bad} failed, {n_skip} skipped (v4 needs size divisible by 4)")
    print("ALL KERNELS CORRECT" if n_bad == 0 else "FIX FAILURES BEFORE ANY TIMING")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
