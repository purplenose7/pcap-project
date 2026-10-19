"""
p1/zip_to_frames.py — pull a FEW sequences of consecutive JPEG frames out of a big dataset zip,
without extracting the whole thing (the UA-DETRAC zips total ~10 GB).

  python p1/zip_to_frames.py --zip /tmp/ua/ua_detrac_training_set.zip --list              # look first
  python p1/zip_to_frames.py --zip /tmp/ua/ua_detrac_training_set.zip --out /tmp/frames --n-seq 3 --max-frames 600

Output layout (what run_all.py --seq-root and fidelity.py expect): one folder per sequence, full of JPEGs.
  /tmp/frames/<sequence>/<frame>.jpg

Notes
  * Frames are put in NATURAL order (img2 before img10) — consecutive-frame differences need true order.
  * Only .jpg/.jpeg are extracted. If the zip holds PNGs, --list says so: the GPU decode path here is JPEG-only.
  * Sequences are picked evenly across the zip (deterministic), not just the first few.
"""
import argparse, re, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from pathlib import Path

JPEG = {".jpg", ".jpeg"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def scan(zf):
    """Returns (sequences: {folder: [member names, natural-sorted]}, extension counts)."""
    seqs, exts = defaultdict(list), Counter()
    for n in zf.namelist():
        if n.endswith("/"):
            continue
        ext = PurePosixPath(n).suffix.lower()
        exts[ext or "(none)"] += 1
        if ext in JPEG:
            seqs[str(PurePosixPath(n).parent)].append(n)
    for k in seqs:
        seqs[k].sort(key=lambda m: natural_key(PurePosixPath(m).name))
    return seqs, exts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="/tmp/frames")
    ap.add_argument("--n-seq", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--min-frames", type=int, default=100)
    ap.add_argument("--list", action="store_true", help="only show the layout, extract nothing")
    a = ap.parse_args()

    with zipfile.ZipFile(a.zip) as zf:
        seqs, exts = scan(zf)
        big = {k: v for k, v in seqs.items() if len(v) >= a.min_frames}
        print(f"{Path(a.zip).name}: {sum(exts.values())} files; extensions: {dict(exts.most_common(6))}")
        print(f"JPEG folders with >= {a.min_frames} frames (= sequences): {len(big)}")
        for k in sorted(big)[:12]:
            names = [PurePosixPath(m).name for m in big[k]]
            print(f"   {k}   {len(names)} frames   {names[0]} ... {names[-1]}")
        if len(big) > 12:
            print(f"   ... and {len(big) - 12} more")
        if not big:
            non = {e: c for e, c in exts.items() if e not in (".jpg", ".jpeg")}
            raise SystemExit("No JPEG sequences found. Other file types present: "
                             f"{non}. Paste this output to your teammate/me — the layout may differ from what this expects.")
        if a.list:
            return

        folders = sorted(big)
        step = max(1, len(folders) // a.n_seq)
        chosen = folders[::step][: a.n_seq]
        out = Path(a.out)
        for folder in chosen:
            members = big[folder][: a.max_frames]
            dst = out / PurePosixPath(folder).name
            dst.mkdir(parents=True, exist_ok=True)
            for m in members:
                with zf.open(m) as src, open(dst / PurePosixPath(m).name, "wb") as f:
                    shutil.copyfileobj(src, f)
            print(f"extracted {len(members)} frames -> {dst}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"done: {total:.0f} MB in {out}")
    print(f"next:  python p1/run_all.py --tag <tag> --seq-root {out}")


if __name__ == "__main__":
    main()
