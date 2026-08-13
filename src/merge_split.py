#!/usr/bin/env python3
r"""Merge a patch dataset's train/ and test/ splits into a single folder.

WHY: cut_patches.py writes  <root>/train/<species>/*.jpg  and  <root>/test/<species>/*.jpg
when --test-ratio > 0. The fold-based loader does its own tree-level split at load time and
expects ONE directory of species subfolders, so a pre-split dataset has to be recombined
before it can be used for cross-validation. (The cluster copies were cut with
--test-ratio 0, which is why they only have train/.)

RESULT:  <root>/train/<species>/*.jpg   containing every patch   (default, in-place)
    or   <dest>/<species>/*.jpg          with --dest

Patch names are <treeID>_<imageUID>_<class>_<patchNo>.jpg and the original split was by
TREE, so train/ and test/ hold disjoint trees and name collisions should be impossible.
The script checks anyway and refuses to clobber anything if that turns out to be wrong.

    python merge_splits.py "G:\data\patches_224"
    python merge_splits.py "G:\data\patches_224" --mode hardlink --dest "G:\data\patches_224_all"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def species_map(split_dir: Path) -> dict[str, list[Path]]:
    """species -> list of image files directly under <split_dir>/<species>/."""
    out: dict[str, list[Path]] = defaultdict(list)
    if not split_dir.is_dir():
        return out
    for sp_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        files = [f for f in sp_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        if files:
            out[sp_dir.name] = files
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help=r"folder containing train\ and test\ (e.g. G:\data\patches_224)")
    ap.add_argument("--dest", default=None,
                    help="write the merged set here instead of merging into train/ in place")
    ap.add_argument("--mode", choices=["move", "copy", "hardlink"], default="move",
                    help="move = fastest, no extra space, empties test/ (default). "
                         "copy = safest, needs 2x space. "
                         "hardlink = no extra space AND non-destructive, but both paths "
                         "must be on the SAME volume (fine on one NTFS drive).")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    root = Path(args.root)
    train_dir, test_dir = root / "train", root / "test"
    if not train_dir.is_dir() and not test_dir.is_dir():
        print(f"Neither {train_dir} nor {test_dir} exists -- is {root} the right folder?")
        return 1

    tr, te = species_map(train_dir), species_map(test_dir)
    n_tr = sum(len(v) for v in tr.values())
    n_te = sum(len(v) for v in te.values())
    all_species = sorted(set(tr) | set(te))

    print(f"train/ : {n_tr:>9,} patches across {len(tr)} species")
    print(f"test/  : {n_te:>9,} patches across {len(te)} species")
    print(f"total  : {n_tr + n_te:>9,} patches across {len(all_species)} species")

    if n_te == 0:
        print("\ntest/ is empty -- nothing to merge. This dataset is already fold-ready.")
        return 0

    dest_root = Path(args.dest) if args.dest else train_dir
    in_place = args.dest is None

    # ---- pre-flight: collisions --------------------------------------------------------
    # Split was by tree, so the same patch name must not appear on both sides.
    collisions = []
    for sp in all_species:
        existing = {f.name for f in tr.get(sp, [])}
        for f in te.get(sp, []):
            if f.name in existing:
                collisions.append((sp, f.name))
    if collisions:
        print(f"\nABORT: {len(collisions)} filename collision(s) between train/ and test/.")
        for sp, name in collisions[:10]:
            print(f"   {sp}/{name}")
        print("The split was supposed to be tree-level, so this should not happen."
              "\nNothing was moved. Investigate before merging.")
        return 1
    print("\nNo filename collisions (as expected for a tree-level split).")

    # ---- plan --------------------------------------------------------------------------
    verb = {"move": "MOVE", "copy": "COPY", "hardlink": "HARDLINK"}[args.mode]
    print(f"\nPlan: {verb} {n_te:,} files from test/ -> "
          f"{'train/ (in place)' if in_place else dest_root}")
    if not in_place:
        print(f"      and {verb} {n_tr:,} files from train/ -> {dest_root}")
    if args.mode == "move":
        print("      test/ will be left empty (the files are relocated, not duplicated).")
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted; nothing changed.")
            return 0

    # ---- transfer ----------------------------------------------------------------------
    def transfer(src: Path, dst: Path) -> None:
        if args.mode == "copy":
            shutil.copy2(src, dst)
        elif args.mode == "hardlink":
            os.link(src, dst)
        else:
            # os.replace is a metadata-only rename within a volume, far faster than a copy;
            # shutil.move falls back to copy+delete across volumes.
            try:
                os.replace(src, dst)
            except OSError:
                shutil.move(str(src), str(dst))

    jobs: list[tuple[Path, Path]] = []
    for sp in all_species:
        (dest_root / sp).mkdir(parents=True, exist_ok=True)
        if not in_place:
            jobs += [(f, dest_root / sp / f.name) for f in tr.get(sp, [])]
        jobs += [(f, dest_root / sp / f.name) for f in te.get(sp, [])]

    total = len(jobs)
    print(f"\n{verb}ing {total:,} files ...")
    done = 0
    for src, dst in jobs:
        transfer(src, dst)
        done += 1
        if done % 20000 == 0 or done == total:
            pct = 100.0 * done / total
            print(f"  {done:>9,}/{total:,}  ({pct:5.1f}%)")
            sys.stdout.flush()

    # ---- verify ------------------------------------------------------------------------
    final = species_map(dest_root)
    n_final = sum(len(v) for v in final.values())
    expected = n_tr + n_te
    print(f"\nVerification: {n_final:,} patches now under {dest_root}"
          f"  (expected {expected:,})")
    if n_final != expected:
        print("  !! COUNT MISMATCH -- check for errors above before using this dataset.")
        return 1
    print(f"  species dirs: {len(final)}")

    # merge the split manifests if present (not required -- the loader scans the tree)
    for name in ("patch_index.csv",):
        a, b = train_dir / name, test_dir / name
        if a.exists() and b.exists():
            out = dest_root / name if not in_place else a
            try:
                header_written = False
                lines = []
                for src_csv in (a, b):
                    with open(src_csv) as fh:
                        for i, line in enumerate(fh):
                            if i == 0 and header_written:
                                continue
                            lines.append(line)
                    header_written = True
                with open(out, "w") as fh:
                    fh.writelines(lines)
                print(f"  merged {name} -> {out}")
            except Exception as exc:
                print(f"  (could not merge {name}: {exc} -- harmless, the loader scans "
                      f"the folder tree, not this CSV)")

    if args.mode == "move" and in_place:
        left = sum(len(v) for v in species_map(test_dir).values())
        print(f"\ntest/ now holds {left} patches; you can delete the empty folders.")

    print(f"\nDone. Point the loader's patch_root at:  {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())