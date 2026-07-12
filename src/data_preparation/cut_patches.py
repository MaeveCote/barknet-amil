"""Cut bark images into square patches for the BarkNet-AMIL pipeline.

Input layout (source)::

    src/
        <SPECIES>/
            <tree>_<class>_<circ>_<device>_<date>_<time>_<crop>.jpg
            ...

Output layout (destination) is split into two independent, self-contained roots::

    dst/
        train/<SPECIES>/<treeID>_<imageUID>_<className>_<patchNo>.jpg
        train/patch_index.csv
        test/<SPECIES>/...
        test/patch_index.csv
        tree_split.csv          # which tree went where, for reproducibility

The split is done at the TREE level, stratified per species, before any cutting
happens -- the same leakage-safe principle used for the train/val split at training
time. A tree's photos always land entirely in train or entirely in test, never both.
``train/`` becomes ``data.patch_root`` and ``test/`` becomes ``test.data_root`` in
the pipeline config; the dynamic train/val split at training time then further
divides ``train/`` on every run, while ``test/`` stays fixed and untouched until
final evaluation.

Patches are renamed ``<treeID>_<imageUID>_<className>_<patchNo>.jpg`` so the data
loader can recover the tree (split granularity) and the source photo (bag
granularity) directly from the filename. ``imageUID`` is a per-tree running index
over distinct photos, so two phones reusing the same crop number on one tree still
produce two separate bags.

Two cutting methods:

* ``minimal_overlap`` (default, recommended): ceil(dim / patch) patches per axis, with
  the overlap spread evenly so the whole image is covered and no border is wasted.
  This matches the Carbotech production cutting logic.
* ``nonoverlap``: floor(dim / patch) patches starting at the origin, leaving an unused
  border on the right and bottom.

Usage::

    python cut_patches.py SRC DST --patch-size 224 --test-ratio 0.15
    python cut_patches.py SRC DST --analyze        # compare cutting methods, cut nothing
"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is unavailable
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def compute_starts(length: int, patch: int, method: str):
    """Top-left start coordinates of patches along one axis of size ``length``."""
    if length <= patch:
        return [0]  # caller resizes images smaller than a patch up to patch size

    if method == "nonoverlap":
        n = length // patch
        return [i * patch for i in range(n)]  # border left unused at the far edge

    # minimal_overlap: cover the whole axis, overlap shared evenly across patches.
    n = math.ceil(length / patch)
    if n == 1:
        return [0]
    step = (length - patch) / (n - 1)
    starts = [int(round(i * step)) for i in range(n)]
    # Clamp into valid range (defensive against rounding at the edges).
    return [max(0, min(s, length - patch)) for s in starts]


def patch_grid(width: int, height: int, patch: int, method: str):
    """Yield (patch_no, x, y) for every patch in row-major order."""
    xs = compute_starts(width, patch, method)
    ys = compute_starts(height, patch, method)
    n = 0
    for y in ys:
        for x in xs:
            yield n, x, y
            n += 1


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_filename(path: Path):
    """Extract metadata from a BarkNet-style filename.

    Robust to the optional device token: tree id is the first token, class the second,
    circumference the third, and the crop id is always the last token.
    """
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name}")

    record = {
        "tree_id": parts[0],
        "class_name": parts[1],
        "circumference_cm": parts[2],
        "crop_id": parts[-1],
        "date": parts[-3] if len(parts) >= 6 else "",
        "time": parts[-2] if len(parts) >= 6 else "",
        "device": parts[3] if len(parts) >= 7 else "",
    }
    return record


def dbh_from_circumference(circ: str):
    """Diameter at breast height (cm) = circumference / pi, if parseable."""
    try:
        return round(float(circ) / math.pi, 2)
    except (ValueError, TypeError):
        return ""


# --------------------------------------------------------------------------- #
# Scanning + integrity
# --------------------------------------------------------------------------- #
def scan(src: Path):
    """Build a record per source image and check basic integrity."""
    records = []
    mismatches = 0
    for species_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        species = species_dir.name
        for path in sorted(species_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                meta = parse_filename(path)
            except ValueError as exc:
                print(f"  skip (unparseable): {exc}")
                continue

            if meta["class_name"] != species:
                mismatches += 1
            meta["class_name"] = species
            meta["path"] = path
            meta["species"] = species
            records.append(meta)

    if mismatches:
        print(f"  note: {mismatches} files had a class token != folder name (folder used).")

    # Assign a unique bag id per SOURCE PHOTO, scoped to (species, tree).
    # BarkNet shot many trees with two phones whose crop numbers both start at 1,
    # so the crop number is NOT unique per photo within a tree. A per-tree running
    # index over distinct source files guarantees one bag == one photo.
    by_tree = defaultdict(list)
    for rec in records:
        by_tree[(rec["species"], rec["tree_id"])].append(rec)
    for (_species, _tree), recs in by_tree.items():
        recs.sort(key=lambda r: r["path"].name)  # deterministic across runs
        for image_uid, rec in enumerate(recs):
            rec["image_uid"] = str(image_uid)

    return records


# --------------------------------------------------------------------------- #
# Train/test split (tree-level, stratified per species)
# --------------------------------------------------------------------------- #
def split_trees(records, test_ratio: float, seed: int):
    """Assign every (species, tree) to 'train' or 'test'.

    Stratified per species so a rare species isn't accidentally dropped entirely
    into one side. Returns the assignment dict plus the (species, tree) -> records
    grouping, for the reproducibility manifest.
    """
    by_tree = defaultdict(list)
    for rec in records:
        by_tree[(rec["species"], rec["tree_id"])].append(rec)

    by_species = defaultdict(list)
    for key in by_tree:
        by_species[key[0]].append(key)

    rng = random.Random(seed)
    assignment = {}
    for species, tree_keys in sorted(by_species.items()):
        keys = sorted(tree_keys)  # deterministic order before shuffling
        rng.shuffle(keys)
        n_test = round(len(keys) * test_ratio)
        test_keys = set(keys[:n_test])
        for key in keys:
            assignment[key] = "test" if key in test_keys else "train"

    return assignment, by_tree


def print_split_summary(assignment):
    counts = defaultdict(lambda: {"train": 0, "test": 0})
    for (species, _tree), split in assignment.items():
        counts[species][split] += 1

    print("\nTree-level train/test split (stratified per species):")
    print(f"  {'species':<8}{'train_trees':>12}{'test_trees':>12}")
    for species in sorted(counts):
        c = counts[species]
        print(f"  {species:<8}{c['train']:>12}{c['test']:>12}")


# --------------------------------------------------------------------------- #
# Cutting (one image -> many patches); top-level for multiprocessing
# --------------------------------------------------------------------------- #
def cut_image(args):
    record, dst_root, patch_size, method, fmt, quality = args
    src_path = record["path"]
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)  # BGR; loader converts to RGB
    if img is None:
        return record["species"], record["tree_id"], record["split"], src_path.name, 0, []

    h, w = img.shape[:2]

    # Resize up if the image is smaller than a patch on the limiting dimension.
    if min(h, w) < patch_size:
        scale = patch_size / min(h, w)
        img = cv2.resize(
            img, (max(patch_size, round(w * scale)), max(patch_size, round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = img.shape[:2]

    out_dir = dst_root / record["split"] / record["species"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tree, cls, split = record["tree_id"], record["class_name"], record["split"]
    image_uid, crop = record["image_uid"], record["crop_id"]
    dbh = dbh_from_circumference(record["circumference_cm"])
    save_params = (
        [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt in ("jpg", "jpeg") else []
    )

    rows = []
    count = 0
    for patch_no, x, y in patch_grid(w, h, patch_size, method):
        patch = img[y : y + patch_size, x : x + patch_size]
        # bag token is image_uid (unique per source photo within a tree), not crop
        name = f"{tree}_{image_uid}_{cls}_{patch_no}.{fmt}"
        cv2.imwrite(str(out_dir / name), patch, save_params)
        rows.append(
            [name, cls, split, tree, image_uid, crop, patch_no, x, y, patch_size,
             src_path.name, record["circumference_cm"], dbh, record["date"],
             record["time"], record["device"]]
        )
        count += 1
    return record["species"], record["tree_id"], split, src_path.name, count, rows


# --------------------------------------------------------------------------- #
# Analysis mode (compare methods without writing anything)
# --------------------------------------------------------------------------- #
def analyze(records, patch_size):
    tot = {"minimal_overlap": 0, "nonoverlap": 0}
    waste_sum = 0.0          # non-overlap: fraction of pixels discarded
    overlap_sum = 0.0        # minimal-overlap: redundant area fraction
    counted = 0

    for rec in tqdm(records, desc="Analyzing"):
        try:
            w, h = Image.open(rec["path"]).size
        except Exception:
            continue
        area = w * h
        for method in tot:
            nx = len(compute_starts(w, patch_size, method))
            ny = len(compute_starts(h, patch_size, method))
            tot[method] += nx * ny
            covered = nx * patch_size * ny * patch_size
            if method == "nonoverlap":
                used = min(nx * patch_size, w) * min(ny * patch_size, h)
                waste_sum += 1.0 - used / area
            else:
                overlap_sum += max(0.0, covered / area - 1.0)
        counted += 1

    if not counted:
        print("No readable images found.")
        return

    print("\n================ patch-cutting comparison ================")
    print(f"images analyzed              : {counted}")
    print(f"patch size                   : {patch_size}")
    print(f"total patches (minimal_overlap): {tot['minimal_overlap']:>10,}  "
          f"({tot['minimal_overlap']/counted:.1f}/image)")
    print(f"total patches (nonoverlap)     : {tot['nonoverlap']:>10,}  "
          f"({tot['nonoverlap']/counted:.1f}/image)")
    print(f"mean discarded border (nonoverlap)   : {100*waste_sum/counted:5.1f}% of pixels")
    print(f"mean redundant overlap (minimal)     : {100*overlap_sum/counted:5.1f}% of area")
    print("=========================================================")
    print("Recommendation: minimal_overlap — full coverage, no wasted bark,")
    print("consistent with the validated Carbotech cutting logic.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Cut bark images into patches, split by tree into train/test.")
    parser.add_argument("src", type=str, help="source dir with species sub-folders")
    parser.add_argument("dst", type=str, help="destination dir (will contain train/ and test/)")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument(
        "--method", choices=["minimal_overlap", "nonoverlap"], default="minimal_overlap"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="fraction of TREES per species held out for test (0 = all train)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="seed for the tree-level train/test split (deterministic)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--analyze", action="store_true", help="compare cutting methods, cut nothing")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        parser.error(f"source directory not found: {src}")

    print(f"Scanning {src} ...")
    records = scan(src)
    print(f"Found {len(records)} source images across "
          f"{len(set(r['species'] for r in records))} species.")

    if args.analyze:
        analyze(records, args.patch_size)
        return

    assignment, by_tree = split_trees(records, args.test_ratio, args.split_seed)
    for rec in records:
        rec["split"] = assignment[(rec["species"], rec["tree_id"])]
    print_split_summary(assignment)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)

    header = ["patch_filename", "species", "split", "tree_id", "image_uid", "crop_id",
              "patch_no", "x", "y", "patch_size", "src_image", "circumference_cm",
              "dbh_cm", "date", "time", "device"]

    tasks = [(rec, dst, args.patch_size, args.method, args.format, args.quality)
             for rec in records]
    patches_per_split = {"train": 0, "test": 0}
    patches_per_tree = defaultdict(int)

    files = {
        "train": open(dst / "train" / "patch_index.csv", "w", newline=""),
        "test": open(dst / "test" / "patch_index.csv", "w", newline=""),
    }
    writers = {split: csv.writer(f) for split, f in files.items()}
    for w_ in writers.values():
        w_.writerow(header)

    try:
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(cut_image, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Cutting"):
                    species, tree, split, _src_name, count, rows = fut.result()
                    writers[split].writerows(rows)
                    patches_per_split[split] += count
                    patches_per_tree[(species, tree)] += count
        else:
            for t in tqdm(tasks, desc="Cutting"):
                species, tree, split, _src_name, count, rows = cut_image(t)
                writers[split].writerows(rows)
                patches_per_split[split] += count
                patches_per_tree[(species, tree)] += count
    finally:
        for f in files.values():
            f.close()

    # Reproducibility manifest: which tree went where, and how much it produced.
    tree_split_path = dst / "tree_split.csv"
    with open(tree_split_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "tree_id", "split", "num_source_images", "num_patches"])
        for (species, tree), recs in sorted(by_tree.items()):
            split = assignment[(species, tree)]
            writer.writerow([species, tree, split, len(recs), patches_per_tree[(species, tree)]])

    total = patches_per_split["train"] + patches_per_split["test"]
    print(f"\nDone. Wrote {total:,} patches to {dst}")
    print(f"  train: {patches_per_split['train']:,} patches -> {dst / 'train'}")
    print(f"  test : {patches_per_split['test']:,} patches -> {dst / 'test'}")
    print(f"Tree-level split manifest: {tree_split_path}")


if __name__ == "__main__":
    main()
