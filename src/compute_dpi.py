#!/usr/bin/env python3
"""Estimate the spatial resolution (DPI) of the BarkNet 1.0 images and of the patches
cut from them.

RUNS LOCALLY on the raw BarkNet folder -- no cluster, no GPU. It reads only each JPEG's
HEADER (PIL gives .size without decoding pixels), so ~23k images take seconds even off a
spinning disk. Point it at the folder that holds the per-species subfolders.

    python compute_dpi.py G:\\data\\barknet_raw --csv image_dpi.csv

FILENAME FORMAT
    <tree_id>_<class>_<circumference_cm>_<device>_<yyyymmdd>_<hhmmss>_<crop>.jpg
    e.g.  41_CHR_83_GalaxyS5_20170607_134920_2.jpg
Note the README's numbered list omits the device token, but it is present in the actual
names (field 4), so the circumference is parts[2].

HOW THE DPI IS DERIVED
    diameter_cm = circumference_cm / pi                (DBH from girth)
    trunk_px    = image_width_px * TRUNK_FILL          (see the assumption below)
    px_per_cm   = trunk_px / diameter_cm
    DPI         = px_per_cm * 2.54

>>> THE LOAD-BEARING ASSUMPTION <<<
This assumes the trunk spans TRUNK_FILL of the image's horizontal extent (default 1.0,
i.e. the trunk fills the frame edge to edge). BarkNet images are close-ups, so if a photo
actually frames only part of the trunk, the true px_per_cm is HIGHER than computed here
and the reported DPI is an UNDER-estimate. Conversely, if the trunk is smaller than the
frame, DPI is over-estimated. Because the fraction is not recorded in the dataset, treat
these numbers as an order-of-magnitude scale, not a calibrated measurement, and report the
assumption alongside them. Use --trunk-fill to test sensitivity.

Width is used (not height) because the trunk's diameter runs horizontally in a standing
photograph; --use-min-side uses min(w, h) instead if the images are rotated.

PATCH DPI
    A patch of P pixels cut from the original covers  P / px_per_cm  cm of real bark.
    The network then resizes it to INPUT (224), so the effective sampling density is
        patch_DPI = DPI_image * INPUT / P
    Note P > INPUT is a DOWN-scale (real detail discarded) while P < INPUT is an UP-scale:
    the nominal DPI rises but no new information is added, which matters when interpreting
    the small-patch arms of the ablation.
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these are big phone photos; the decompression-bomb guard
# is irrelevant here and only produces warnings.

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
CM_PER_INCH = 2.54


def parse_name(path: Path):
    """-> (tree_id, species, circumference_cm, device) or None if unparseable."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        circ = float(parts[2])
    except ValueError:
        return None
    if circ <= 0:
        return None
    device = parts[3] if len(parts) > 3 else "unknown"
    return parts[0], parts[1], circ, device


def measure(path_str: str):
    """Read ONLY the image header for its size -- no pixel decode, so this is fast."""
    p = Path(path_str)
    meta = parse_name(p)
    if meta is None:
        return None
    tree, species, circ, device = meta
    try:
        with Image.open(p) as im:
            w, h = im.size          # lazy: PIL does not decode the pixel data here
    except Exception:
        return None
    return {"path": str(p), "tree_id": tree, "species": species,
            "circumference_cm": circ, "device": device, "width": w, "height": h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder of raw BarkNet images (searched recursively)")
    ap.add_argument("--trunk-fill", type=float, default=1.0,
                    help="fraction of the image's horizontal extent occupied by the trunk "
                         "(default 1.0 = fills the frame). See the assumption note.")
    ap.add_argument("--use-min-side", action="store_true",
                    help="use min(width,height) instead of width, for rotated images")
    ap.add_argument("--input-size", type=int, default=224,
                    help="network input size the patches are resized to (default 224)")
    ap.add_argument("--patch-sizes", type=int, nargs="+",
                    default=[160, 224, 288, 384, 512])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="parallel header readers (default: all cores but one)")
    ap.add_argument("--csv", default=None, help="optional per-image CSV output")
    args = ap.parse_args()

    root = Path(args.root)
    files = [str(p) for p in root.rglob("*") if p.suffix in EXTS]
    print(f"Found {len(files)} candidate images under {root}")
    if not files:
        return

    recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(measure, f) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                recs.append(r)
            if i % 2000 == 0:
                print(f"  ...{i}/{len(files)}")

    print(f"Parsed {len(recs)} images ({len(files) - len(recs)} skipped: bad name or unreadable)")
    if not recs:
        return

    circ = np.array([r["circumference_cm"] for r in recs], dtype=float)
    W = np.array([r["width"] for r in recs], dtype=float)
    H = np.array([r["height"] for r in recs], dtype=float)
    side = np.minimum(W, H) if args.use_min_side else W

    diameter_cm = circ / math.pi
    trunk_px = side * args.trunk_fill
    px_per_cm = trunk_px / diameter_cm
    dpi = px_per_cm * CM_PER_INCH

    for r, d, pc, dp in zip(recs, diameter_cm, px_per_cm, dpi):
        r["diameter_cm"] = float(d)
        r["px_per_cm"] = float(pc)
        r["dpi"] = float(dp)

    def stat(name, a, unit=""):
        print(f"  {name:<28} {a.mean():10.2f} +/- {a.std(ddof=1):8.2f} {unit}"
              f"   [min {a.min():.1f}, median {np.median(a):.1f}, max {a.max():.1f}]")

    print("\n" + "=" * 78)
    print(f"DATASET SCALE   (trunk_fill={args.trunk_fill}, "
          f"side={'min(w,h)' if args.use_min_side else 'width'})")
    print("=" * 78)
    stat("circumference (cm)", circ)
    stat("DBH diameter (cm)", diameter_cm)
    stat("image width (px)", W)
    stat("image height (px)", H)
    stat("pixels per cm", px_per_cm)
    stat("IMAGE DPI", dpi)

    print("\n" + "=" * 78)
    print(f"PATCH SCALE     (patches resized to {args.input_size} px for the network)")
    print("=" * 78)
    print(f"  {'patch':>6} | {'covers (cm)':>18} | {'scale':>7} | {'effective patch DPI':>24}")
    print("  " + "-" * 68)
    for P in args.patch_sizes:
        extent_cm = P / px_per_cm                      # real bark spanned by the patch
        scale = args.input_size / P                    # >1 = upscale, <1 = downscale
        patch_dpi = dpi * scale
        tag = "up" if scale > 1 else ("1:1" if abs(scale - 1) < 1e-9 else "down")
        print(f"  {P:>6} | {extent_cm.mean():8.2f} +/- {extent_cm.std(ddof=1):5.2f} | "
              f"{scale:6.3f}{tag:>4} | {patch_dpi.mean():12.1f} +/- {patch_dpi.std(ddof=1):7.1f}")

    print("\n  'covers (cm)' is the physical bark extent per patch -- often more"
          "\n  interpretable for the paper than DPI. Upscaled arms (patch < "
          f"{args.input_size}) gain"
          "\n  nominal DPI without gaining real detail.")

    # per-device, since different phones have different sensors
    print("\n" + "=" * 78)
    print("BY CAPTURE DEVICE")
    print("=" * 78)
    devs = {}
    for r in recs:
        devs.setdefault(r["device"], []).append(r["dpi"])
    for d, vals in sorted(devs.items(), key=lambda kv: -len(kv[1])):
        v = np.array(vals)
        sd = v.std(ddof=1) if v.size > 1 else 0.0
        print(f"  {d:<18} n={v.size:>6}   DPI {v.mean():8.1f} +/- {sd:7.1f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"\nPer-image values -> {args.csv}")


if __name__ == "__main__":
    main()
