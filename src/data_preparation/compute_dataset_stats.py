"""Compute per-channel mean and std over the bark patch dataset.

The values are produced in the exact form the training pipeline expects:
  * RGB channel order (the loader does BGR -> RGB before ToTensor),
  * [0, 1] scale (ToTensor divides by 255), so they plug straight into
    ``augmentation.normalize`` in the config.

The held-out test set should live elsewhere and is never included.

Usage:
    python compute_dataset_stats.py -c configs/config.yaml
    python compute_dataset_stats.py --root ./data/barknet_patches/train
    python compute_dataset_stats.py --root ./data/patches --sample-fraction 0.15
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def accumulate_chunk(paths):
    """Per-channel (RGB, 0-1) sum, sum-of-squares, and pixel count for a list of files."""
    s = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    n = 0
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        s += flat.sum(axis=0)
        sq += (flat * flat).sum(axis=0)
        n += flat.shape[0]
    return s, sq, n

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def collect_paths(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

def main():
    parser = argparse.ArgumentParser(description="Compute dataset RGB mean/std.")
    parser.add_argument("-c", "--config", default=None, help="read patch_root from this config")
    parser.add_argument("--root", default=None, help="patch directory (overrides config)")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="fraction of patches to sample (stats converge fast; 0.1-0.2 is plenty)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=256, help="files per worker task")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).resolve()
    elif args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["data"]["patch_root"]).resolve()
    else:
        parser.error("provide --root or --config")

    if not root.is_dir():
        parser.error(f"patch directory not found: {root}")

    print(f"Scanning {root} ...")
    paths = collect_paths(root)
    if not paths:
        parser.error("no image files found")

    if args.sample_fraction < 1.0:
        random.seed(args.seed)
        random.shuffle(paths)
        keep = max(1, int(len(paths) * args.sample_fraction))
        paths = paths[:keep]
        print(f"Sampling {keep:,} of the patches ({args.sample_fraction:.0%}).")
    else:
        print(f"Using all {len(paths):,} patches.")

    total_s = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    total_n = 0

    tasks = list(chunked(paths, args.chunk))
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(accumulate_chunk, c) for c in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Accumulating"):
                s, sq, n = fut.result()
                total_s += s
                total_sq += sq
                total_n += n
    else:
        for c in tqdm(tasks, desc="Accumulating"):
            s, sq, n = accumulate_chunk(c)
            total_s += s
            total_sq += sq
            total_n += n

    if total_n == 0:
        parser.error("no readable pixels")

    mean = total_s / total_n
    var = total_sq / total_n - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))

    print(f"\nPixels per channel : {total_n:,}")
    print(f"mean (RGB, 0-1)    : [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std  (RGB, 0-1)    : [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("\nPaste into configs/config.yaml under augmentation.normalize:\n")
    print("  normalize:")
    print(f"    mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"    std: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")


if __name__ == "__main__":
    main()