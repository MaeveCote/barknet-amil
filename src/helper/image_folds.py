"""Image-level folds for the v2 leakage experiment (deliberately NOT leakage-safe).

The paper's evaluation splits by TREE. The v2 experiment measures how much an
image-level split inflates accuracy, so here the unit is the IMAGE and tree ids are
ignored: images of one tree land on both sides of the split.

Image identity
--------------
An image is identified as ``<SPECIES>:<tree>:<uid>``, where ``uid`` is the image's index
among its tree's source files sorted by filename -- exactly the ``imageUID`` that
``data_preparation/cut_patches.py`` writes into every patch filename
(``<tree>_<uid>_<class>_<patch>.jpg``). The same id can therefore be computed from the
raw images (whole-image baseline) and from any patch cut (patch pipeline), so every
pipeline shares ONE fold assignment, stored in a CSV and never recomputed per run.

Folds: ``StratifiedKFold(n_folds, shuffle=True, random_state=seed)`` over images sorted by
id, stratified by species. The inner validation split is also image-level: a stratified
``val_ratio`` of the non-test images, drawn from the fold CSV alone (not from whatever a
given patch root contains), so it is identical across pipelines too.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

FOLD_COLUMNS = ["image_id", "tree_id", "species", "fold", "src_image"]

# Same set as helper.data_loader.IMAGE_EXTENSIONS (which imports it from here). This module
# deliberately needs only the standard library (+ scikit-learn for the folds), so the fold
# file can be built on a login node without OpenCV / torch.
IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_species_files(cls_dir: Path) -> List[Path]:
    """Image files of one species folder, in the order helper.data_loader._scan walks them."""
    return [p for p in sorted(cls_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]


def scan_raw(root, species: List[str]) -> List[Dict]:
    """One record per raw photo of the given species: image_id, tree_id, species, src_image.
    Identical ids to helper.data_loader._scan with ``data.filename.layout=raw``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"raw image root does not exist: {root}")
    records = []
    for sp in species:
        cls_dir = root / sp
        if not cls_dir.exists():
            print(f"Warning: missing class directory {cls_dir}")
            continue
        files = [p for p in list_species_files(cls_dir) if is_raw_name(p)]
        uids = assign_raw_uids(files)
        for p in files:
            tree = p.stem.split("_")[0]
            records.append({"image_id": image_id(sp, tree, uids[p]), "tree_id": tree,
                            "species": sp, "src_image": p.name})
    return records


def image_id(species: str, tree: str, uid) -> str:
    return f"{species}:{tree}:{uid}"


def is_raw_name(path: Path) -> bool:
    """cut_patches.scan() skips files with fewer than 4 '_' tokens before numbering."""
    return len(path.stem.split("_")) >= 4


def assign_raw_uids(paths: Iterable[Path]) -> Dict[Path, str]:
    """``imageUID`` per raw file of ONE species folder, reproducing cut_patches.scan():
    within a tree the source files are sorted by name and numbered from 0."""
    by_tree: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        by_tree[p.stem.split("_")[0]].append(p)
    uids = {}
    for _, files in by_tree.items():
        for uid, p in enumerate(sorted(files, key=lambda q: q.name)):
            uids[p] = str(uid)
    return uids


def make_folds(images: List[Dict], n_folds: int, seed: int) -> List[Dict]:
    """``images``: dicts with image_id, tree_id, species, src_image. Returns them with a
    ``fold`` key, sorted by image_id (the order the stratified split is drawn in)."""
    from sklearn.model_selection import StratifiedKFold

    rows = sorted(images, key=lambda r: r["image_id"])
    ids = list(range(len(rows)))
    y = [r["species"] for r in rows]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, test_idx) in enumerate(skf.split(ids, y)):
        for i in test_idx:
            rows[i]["fold"] = f
    return rows


def write_fold_file(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FOLD_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_fold_file(path) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fold"] = int(r["fold"])
    if not rows or set(FOLD_COLUMNS) - set(rows[0]):
        raise ValueError(f"{path} is not an image-fold file (need columns {FOLD_COLUMNS}).")
    return rows


def partition(rows: List[Dict], fold_index: int, val_ratio: float, seed: int
              ) -> Dict[str, Set[str]]:
    """image ids per split for one fold: test = that fold; val = a stratified val_ratio
    of the rest (image-level); train = the remainder."""
    from sklearn.model_selection import train_test_split

    test = {r["image_id"] for r in rows if r["fold"] == fold_index}
    if not test:
        raise ValueError(f"fold {fold_index} is empty in the fold file")
    rest = sorted((r for r in rows if r["fold"] != fold_index), key=lambda r: r["image_id"])
    if val_ratio and val_ratio > 0:
        tr, va = train_test_split(
            [r["image_id"] for r in rest], test_size=val_ratio,
            stratify=[r["species"] for r in rest], random_state=seed)
        train, val = set(tr), set(va)
    else:
        train, val = {r["image_id"] for r in rest}, set()
    return {"train": train, "val": val, "test": test}


def leakage_report(rows: List[Dict], parts: Dict[str, Set[str]]) -> Dict:
    """Share of test images whose tree also appears in training (expected ~100%)."""
    tree_of = {r["image_id"]: f"{r['species']}:{r['tree_id']}" for r in rows}
    train_trees = {tree_of[i] for i in parts["train"]}
    seen_trees = train_trees | {tree_of[i] for i in parts["val"]}
    test = parts["test"]
    n = max(len(test), 1)
    return {
        "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(test),
        "test_images_tree_in_train": sum(tree_of[i] in train_trees for i in test) / n,
        "test_images_tree_in_train_or_val": sum(tree_of[i] in seen_trees for i in test) / n,
        "test_trees_in_train": (len({tree_of[i] for i in test} & train_trees)
                                / max(len({tree_of[i] for i in test}), 1)),
        "test_species_counts": dict(sorted(Counter(i.split(":")[0] for i in test).items())),
    }
