"""Build the IMAGE-level fold file for the v2 leakage experiment (CPU only, run once).

Scans the RAW BarkNet photos (``<root>/<SPECIES>/<tree>_<class>_..._<crop>.jpg``) for the
species in the config, gives every photo the same ``<SPECIES>:<tree>:<uid>`` id that the
patch pipeline uses, and assigns 5 folds with ``StratifiedKFold`` over images (stratified
by species, shuffled, fixed seed, tree ids ignored). Then it reports, per fold, the share
of test images whose tree also appears in training -- the leakage the experiment exists to
measure, which should be close to 100%.

Outputs (in --out-dir):
    image_folds.csv          image_id, tree_id, species, fold, src_image
    image_folds_leakage.csv  per-fold sanity check (and .json with per-species counts)

Usage:
    python src/data_preparation/make_image_folds.py --raw-root $SCRATCH/data/barknet_raw \
        -c configs/config.yaml --out-dir $SCRATCH/results/v2/folds
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from helper import image_folds  # noqa: E402
from helper.data_loader import _scan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", required=True, help="folder of <SPECIES>/<raw photo>.jpg")
    ap.add_argument("-c", "--config", default="configs/config.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-folds", type=int, default=None, help="default: data.split.n_folds")
    ap.add_argument("--seed", type=int, default=None, help="default: project.seed")
    ap.add_argument("--expect-images", type=int, default=None,
                    help="fail unless exactly this many images are found (23271 for BarkNet-20)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing fold file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    species = cfg["data"]["species"]
    seed = args.seed if args.seed is not None else int(cfg["project"]["seed"])
    n_folds = args.n_folds or int(cfg["data"]["split"]["n_folds"])
    val_ratio = float(cfg["data"].get("val_ratio", 0.15))

    out_dir = Path(args.out_dir)
    fold_path = out_dir / "image_folds.csv"
    if fold_path.exists() and not args.force:
        sys.exit(f"{fold_path} already exists -- refusing to overwrite (pass --force to rebuild).")

    fcfg = {**cfg["data"].get("filename", {}), "layout": "raw"}
    trees, _ = _scan(args.raw_root, species, fcfg)
    images = []
    for tree_imgs in trees.values():
        for key, bag in tree_imgs.items():
            label, tree, uid = key.split(":")
            assert len(bag) == 1, f"raw layout should give one file per image, got {key}"
            sp = species[int(label)]
            images.append({"image_id": image_folds.image_id(sp, tree, uid), "tree_id": tree,
                           "species": sp, "src_image": bag[0][0].name})
    n_trees = len({(r["species"], r["tree_id"]) for r in images})
    print(f"{len(images)} images, {n_trees} trees, {len({r['species'] for r in images})} species")
    if args.expect_images is not None and len(images) != args.expect_images:
        sys.exit(f"expected {args.expect_images} images, found {len(images)} -- wrong raw root?")

    rows = image_folds.make_folds(images, n_folds=n_folds, seed=seed)
    image_folds.write_fold_file(rows, fold_path)
    print(f"wrote {fold_path}  (StratifiedKFold n={n_folds}, shuffle=True, seed={seed})")

    report = []
    for f in range(n_folds):
        rep = image_folds.leakage_report(rows, image_folds.partition(rows, f, val_ratio, seed))
        report.append({"fold": f, **rep})
        print(f"  fold {f}: train {rep['n_train']:>5} | val {rep['n_val']:>4} | test "
              f"{rep['n_test']:>4} | test images whose tree is in train "
              f"{100 * rep['test_images_tree_in_train']:6.2f}% | test trees seen in train "
              f"{100 * rep['test_trees_in_train']:6.2f}%")
    flat_cols = [k for k in report[0] if k != "test_species_counts"]
    with open(out_dir / "image_folds_leakage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(report)
    meta = {"n_images": len(images), "n_trees": n_trees, "n_folds": n_folds, "seed": seed,
            "val_ratio": val_ratio, "species": species, "raw_root": str(args.raw_root),
            "per_fold": report}
    with open(out_dir / "image_folds_leakage.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out_dir / 'image_folds_leakage.csv'} (+ .json)")


if __name__ == "__main__":
    main()
