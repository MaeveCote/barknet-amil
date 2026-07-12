"""vram_probe.py — how much GPU memory does one board-level AMIL step need?

At the board (Stage-2) level, the unit that must fit in GPU memory is ONE whole
image's bag of patches, pushed through the backbone in a single forward+backward
pass. That size is set by the data, not by a tunable batch size -- so the only way
to know whether an 8 / 10 / 12 / 16 / 24 GB card can run Stage 2 is to measure the
worst-case bag directly.

This script needs NO config file. Point it at a patch directory (it finds the
largest bag automatically) or just give it a patch count:

    python vram_probe.py PATCH_DIR
    python vram_probe.py PATCH_DIR --sizes pico tiny small
    python vram_probe.py PATCH_DIR --input-size 288
    python vram_probe.py --num-patches 85               # skip the data entirely

Because a bag too large to fit can't have its peak measured (it just OOMs), the
probe measures peak memory at several SMALLER bag sizes that do fit, fits a line,
and PROJECTS the requirement at your true worst-case bag -- so even an 8 GB card can
tell you whether a bigger card would clear it.

Adapt the import below to your package name (e.g. `helper.model_wrapper`).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from helper.model_wrapper import ConvNeXtAMIL  # adapt to your renamed package

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CARD_SIZES_GB = [8, 10, 12, 16, 24]
CONTEXT_OVERHEAD_GB = 0.8  # CUDA context lives outside the caching allocator's reserved pool


# --------------------------------------------------------------------------- #
# Find the largest bag (no torch needed -- kept pure for easy testing)
# --------------------------------------------------------------------------- #
def find_largest_bag(patch_dir):
    """Return (num_patches, key) for the image with the most patches.

    A bag is identified by (tree_id, image_uid, species) -- the first three tokens of
    each patch filename ``<tree>_<imageUID>_<class>_<patchNo>``. Uses patch_index.csv
    if present (fast), otherwise scans filenames.
    """
    patch_dir = Path(patch_dir)
    counts = Counter()

    csv_path = patch_dir / "patch_index.csv"
    if csv_path.exists():
        import csv
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                counts[(row["tree_id"], row["image_uid"], row["species"])] += 1
    else:
        for p in patch_dir.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = p.stem.split("_")
            if len(parts) < 3:
                continue
            counts[(parts[0], parts[1], parts[2])] += 1

    if not counts:
        raise RuntimeError(f"No patches found under {patch_dir}")
    key, n = counts.most_common(1)[0]
    return n, key


# --------------------------------------------------------------------------- #
# Memory measurement
# --------------------------------------------------------------------------- #
def measure_peak_reserved(model, wrapper, criterion, optimizer, scaler,
                          n, input_size, device, steps=3):
    """Run a few real training steps on a synthetic [n,3,H,W] bag; return peak
    reserved GB. Pixel content is irrelevant to memory, so random data is fine.
    Raises RuntimeError on OOM."""
    label = torch.zeros(1, dtype=torch.long, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(steps):
        x = torch.randn(n, 3, input_size, input_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=wrapper.amp_device, dtype=torch.float16):
            image_logits, patch_logits = model(x)
            loss = wrapper._combined_loss(image_logits, patch_logits, label, criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        del x, image_logits, patch_logits, loss

    return torch.cuda.max_memory_reserved() / 1e9


def probe_model_size(model_size, max_n, input_size, num_classes, device):
    """Sweep bag sizes up to max_n, measure peak reserved at each, return the
    (n, peak_gb) points actually achieved before any OOM."""
    species = [f"c{i}" for i in range(num_classes)]
    wrapper = ConvNeXtAMIL(
        device=device, species=species, model_size=model_size,
        weights="none", instance_loss_weight=0.5, save_data=False,
    )
    model, criterion = wrapper.get_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device=wrapper.amp_device)

    # A spread of bag sizes from small up to the true worst case, for the linear fit.
    fractions = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    sweep = sorted({max(2, round(f * max_n)) for f in fractions})

    points = []
    for n in sweep:
        try:
            peak = measure_peak_reserved(model, wrapper, criterion, optimizer, scaler,
                                          n, input_size, device)
            points.append((n, peak))
            print(f"    bag={n:>4} patches -> {peak:6.2f} GB reserved")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"    bag={n:>4} patches -> OOM (exceeds this card; using smaller bags to project)")
                break
            raise

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return points


def project(points, max_n):
    """Linear fit peak ~= slope*n + intercept; return projected GB at max_n and
    whether it's an exact measurement or an extrapolation."""
    if not points:
        return None, False
    measured = next((gb for n, gb in points if n == max_n), None)
    if measured is not None:
        return measured, True  # the worst case actually fit -- exact
    if len(points) < 2:
        return None, False
    ns = np.array([n for n, _ in points], dtype=float)
    gbs = np.array([gb for _, gb in points], dtype=float)
    slope, intercept = np.polyfit(ns, gbs, 1)
    return float(slope * max_n + intercept), False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Probe peak GPU memory for one board-level AMIL step.")
    parser.add_argument("patch_dir", nargs="?", default=None,
                        help="patch directory (finds the largest bag automatically)")
    parser.add_argument("--num-patches", type=int, default=None,
                        help="worst-case bag size; skips data scan if given")
    parser.add_argument("--sizes", nargs="+", default=["pico", "tiny", "small"],
                        help="model sizes to probe")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("This probe measures GPU memory and needs CUDA. No CUDA device found.")
        sys.exit(1)

    device = args.device
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    print(f"Device: {props.name}  ({total_gb:.1f} GB total)")
    print(f"Input size: {args.input_size}x{args.input_size}  |  classes: {args.num_classes}\n")

    # Determine worst-case bag size.
    if args.num_patches is not None:
        max_n = args.num_patches
        print(f"Worst-case bag: {max_n} patches (from --num-patches)\n")
    elif args.patch_dir:
        n, key = find_largest_bag(args.patch_dir)
        max_n = n
        print(f"Worst-case bag: {max_n} patches  (tree={key[0]}, image={key[1]}, species={key[2]})\n")
    else:
        parser.error("provide a patch directory or --num-patches")

    results = {}
    for size in args.sizes:
        print(f"[{size}] sweeping bag sizes ...")
        try:
            points = probe_model_size(size, max_n, args.input_size, args.num_classes, device)
        except Exception as exc:  # e.g. invalid size name
            print(f"    skipped: {exc}\n")
            continue
        projected, exact = project(points, max_n)
        results[size] = (projected, exact)
        print()

    # Verdict table.
    print("=" * 72)
    print(f"Projected peak VRAM for the worst-case bag ({max_n} patches), "
          f"input {args.input_size}px:")
    print(f"(+{CONTEXT_OVERHEAD_GB} GB added for CUDA context when judging card fit)\n")
    header = "  size      need(GB)   " + "".join(f"{c}GB ".rjust(7) for c in CARD_SIZES_GB)
    print(header)
    for size, (projected, exact) in results.items():
        if projected is None:
            print(f"  {size:<8}  (could not measure -- even the smallest bag OOM'd)")
            continue
        tag = "exact" if exact else "proj."
        need = projected + CONTEXT_OVERHEAD_GB
        verdicts = "".join(("  ok  " if need <= c else "  --  ") for c in CARD_SIZES_GB)
        print(f"  {size:<8}  {projected:6.2f} {tag}   {verdicts}")
    print("=" * 72)
    print("ok = worst-case bag should fit with headroom;  -- = will OOM.")
    print("'proj.' = extrapolated from smaller bags (worst case didn't fit on THIS card).")


if __name__ == "__main__":
    main()