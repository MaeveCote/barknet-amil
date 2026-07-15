"""Data pipeline for pre-cut bark patches.

Directory layout expected on disk (produced by the patch-cutting step)::

    patch_root/
        <SPECIES_CODE>/                 # folder name == class label
            <tree>_<image>_<patch>.jpg  # filename encodes tree id, image id, patch index
            ...

Two grouping keys matter and they are *different*:

* **Split granularity = tree.** Every patch of a given tree lands entirely in train OR
  val OR test. This prevents the same-tree leakage that inflates image-level random
  splits (the methodology issue we are correcting relative to prior work).
* **Bag granularity = image.** The unit the model sees as ``[N, C, H, W]`` is the set of
  patches cut from a single image.

Tree/image ids are parsed from the filename via the ``data.filename`` config block
(delimiter + token indices), so the convention can change without touching code.

Splitting happens **at run time from a single patch root**. The patches do not need to
be pre-partitioned into train/ and test/ folders on disk -- cut everything once (e.g.
with ``--test-ratio 0``, so everything lands in ``train/``) and let ``data.split`` carve
out the test trees here. Two modes:

* ``holdout`` : per class, ``test_ratio`` of the trees become test, then ``val_ratio``
                of what remains becomes val.
* ``kfold``   : per class, trees are shuffled into ``n_folds`` folds; fold
                ``fold_index`` is the test set and ``val_ratio`` of the rest is val.
                Rotating ``fold_index`` gives leakage-safe 5-fold CV without re-cutting.

Legacy mode still works: point ``test.data_root`` at a separate pre-cut patch tree and
that whole directory is used as the test set (see ``build_test_loader``).
"""
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# One OpenCV thread per process. Each DataLoader worker is a process; letting OpenCV
# also spawn a thread pool inside each one oversubscribes the CPUs allocated by SLURM and
# makes decoding slower, not faster.
cv2.setNumThreads(0)

IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

# A bag is a list of (patch_path, label) tuples for a single image.
Bag = List[Tuple[Path, int]]

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_train_transform(aug: Dict) -> Callable:
    """Per-patch training augmentation.

    Patches arrive pre-cut and square, so we Resize to the backbone input instead of
    RandomResizedCrop-upscaling (which silently blew small crops up in the legacy code).
    """
    size = aug["input_size"]
    ops: List[Callable] = [transforms.Resize((size, size))]

    if aug.get("horizontal_flip", 0):
        ops.append(transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]))
    if aug.get("vertical_flip", 0):
        ops.append(transforms.RandomVerticalFlip(p=aug["vertical_flip"]))

    rot = aug.get("rotation", {})
    if rot.get("prob", 0) > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomRotation(rot["degrees"])], p=rot["prob"]
            )
        )

    cj = aug.get("color_jitter", {})
    if cj.get("prob", 0) > 0:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=cj.get("brightness", 0),
                        contrast=cj.get("contrast", 0),
                        saturation=cj.get("saturation", 0),
                        hue=cj.get("hue", 0),
                    )
                ],
                p=cj["prob"],
            )
        )

    gb = aug.get("gaussian_blur", {})
    if gb.get("prob", 0) > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=gb.get("kernel_size", 5))],
                p=gb["prob"],
            )
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(aug["normalize"]["mean"], aug["normalize"]["std"]))

    # RandomErasing (Cutout) operates on tensors, so it comes after ToTensor/Normalize.
    re = aug.get("random_erasing", {})
    if re.get("prob", 0) > 0:
        for _ in range(int(re.get("count", 1))):
            ops.append(
                transforms.RandomErasing(
                    p=re["prob"], scale=tuple(re.get("scale", (0.02, 0.06))), value="random"
                )
            )

    return transforms.Compose(ops)


def build_eval_transform(aug: Dict) -> Callable:
    """Deterministic transform for validation / test."""
    size = aug["input_size"]
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(aug["normalize"]["mean"], aug["normalize"]["std"]),
        ]
    )


# --------------------------------------------------------------------------- #
# Scanning + splitting
# --------------------------------------------------------------------------- #
def _parse_ids(filename: str, fcfg: Dict) -> Tuple[str, str]:
    """Extract (tree_id, image_id) tokens from a patch filename."""
    parts = Path(filename).stem.split(fcfg.get("delimiter", "_"))
    tree = parts[fcfg.get("tree_id_index", 0)]
    image = parts[fcfg.get("image_id_index", 1)]
    return tree, image


def _scan(patch_root: str, species: List[str], fcfg: Dict):
    """Walk ``patch_root`` once and group patches tree -> image -> [(path, label)]."""
    root = Path(patch_root)
    if not root.exists():
        raise FileNotFoundError(f"data.patch_root does not exist: {root}")

    class_to_idx = {name: i for i, name in enumerate(species)}
    trees: Dict[str, Dict[str, Bag]] = defaultdict(lambda: defaultdict(list))
    tree_label: Dict[str, int] = {}

    print(f"Scanning patches under {root} and grouping by tree -> image ...")
    n_patches = 0
    for name in species:
        cls_dir = root / name
        if not cls_dir.exists():
            print(f"Warning: missing class directory {cls_dir}")
            continue
        label = class_to_idx[name]
        for path in sorted(cls_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            tree, image = _parse_ids(path.name, fcfg)
            # Namespace by label so identical raw ids across species never collide.
            tree_key = f"{label}:{tree}"
            image_key = f"{label}:{tree}:{image}"
            trees[tree_key][image_key].append((path, label))
            tree_label[tree_key] = label
            n_patches += 1

    if not n_patches:
        raise RuntimeError(
            f"No patches found under {root}. Check data.patch_root and data.species "
            f"(one sub-directory per species code is expected)."
        )
    print(f"Found {n_patches} patches across {len(tree_label)} trees.")
    return trees, tree_label


def _take(n_items: int, ratio: float) -> int:
    """How many trees a ratio claims, never emptying the pool it draws from."""
    if ratio <= 0 or n_items == 0:
        return 0
    if ratio >= 1.0:
        return n_items
    return min(max(1, int(round(n_items * ratio))), max(n_items - 1, 0))


def _chunks(items: List[str], n: int) -> List[List[str]]:
    """Split a list into ``n`` near-equal contiguous chunks."""
    k, r = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        stop = start + k + (1 if i < r else 0)
        out.append(items[start:stop])
        start = stop
    return out


def split_trees(
        patch_root: str,
        species: List[str],
        fcfg: Dict,
        val_ratio: float,
        test_ratio: float = 0.0,
        seed: int = 42,
        n_folds: Optional[int] = None,
        fold_index: int = 0,
        verbose: bool = True,
) -> Dict[str, List[Bag]]:
    """Tree-level train/val/test split over a single patch root.

    Returns ``{"train": [...], "val": [...], "test": [...]}`` of image-level bags.

    The split is done **per class** so every species is represented in every split, and
    each class's RNG is seeded from ``(seed, label)`` -- so one class's membership does
    not depend on how many classes were scanned before it, and adding or dropping a
    species leaves the other classes' splits untouched.

    ``test_ratio >= 1.0`` sends everything to test (the legacy ``test.data_root`` path,
    where the whole directory *is* the test set).
    """
    trees, tree_label = _scan(patch_root, species, fcfg)

    by_label: Dict[int, List[str]] = defaultdict(list)
    for tree_key, label in tree_label.items():
        by_label[label].append(tree_key)

    splits: Dict[str, List[Bag]] = {name: [] for name in SPLIT_NAMES}
    tree_counts: Dict[str, int] = {name: 0 for name in SPLIT_NAMES}

    for label in sorted(by_label):
        keys = sorted(by_label[label])
        random.Random(f"{seed}:{label}").shuffle(keys)

        if test_ratio >= 1.0:
            test_keys, rest = keys, []
        elif n_folds and int(n_folds) > 1:
            n_folds = int(n_folds)
            folds = _chunks(keys, n_folds)
            f = int(fold_index) % n_folds
            test_keys = folds[f]
            # Rotate the remaining folds so the val trees move with the fold too.
            rest = [k for i in range(1, n_folds) for k in folds[(f + i) % n_folds]]
        else:
            n_test = _take(len(keys), test_ratio)
            test_keys, rest = keys[:n_test], keys[n_test:]

        n_val = _take(len(rest), val_ratio)
        val_keys, train_keys = rest[:n_val], rest[n_val:]

        for name, key_list in (("train", train_keys), ("val", val_keys), ("test", test_keys)):
            tree_counts[name] += len(key_list)
            for tree_key in key_list:
                splits[name].extend(trees[tree_key].values())

    if verbose:
        mode = (f"kfold(n={n_folds}, fold={fold_index})"
                if (n_folds and int(n_folds) > 1) else f"holdout(test_ratio={test_ratio})")
        print(f"Tree-level split [{mode}, val_ratio={val_ratio}, seed={seed}]:")
        for name in SPLIT_NAMES:
            n_bags = len(splits[name])
            n_pat = sum(len(b) for b in splits[name])
            print(f"  {name:<5} {tree_counts[name]:>4} trees | {n_bags:>6} images "
                  f"| {n_pat:>9} patches")
    _assert_disjoint(splits, fcfg)
    return splits


def _assert_disjoint(splits: Dict[str, List[Bag]], fcfg: Dict) -> None:
    """Hard guarantee against the exact failure mode this project exists to correct."""
    tree_sets = {}
    for name, bags in splits.items():
        keys = set()
        for bag in bags:
            tree, _ = _parse_ids(bag[0][0].name, fcfg)
            keys.add(f"{bag[0][1]}:{tree}")
        tree_sets[name] = keys
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = tree_sets[a] & tree_sets[b]
        if overlap:
            raise AssertionError(
                f"TREE LEAKAGE: {len(overlap)} tree(s) in both '{a}' and '{b}', "
                f"e.g. {sorted(overlap)[:5]}"
            )


def split_config(cfg: Dict) -> Dict:
    """Read the ``data.split`` block, with backwards-compatible defaults."""
    data_cfg = cfg["data"]
    sp = data_cfg.get("split") or {}
    return {
        "val_ratio": data_cfg.get("val_ratio", 0.15),
        "test_ratio": float(sp.get("test_ratio") or 0.0),
        "n_folds": sp.get("n_folds"),
        "fold_index": int(sp.get("fold_index") or 0),
    }


def get_splits(cfg: Dict, seed: int = 42, patch_root: Optional[str] = None) -> Dict[str, List[Bag]]:
    """All three splits, driven by the config (single source of truth)."""
    data_cfg = cfg["data"]
    sc = split_config(cfg)
    return split_trees(
        patch_root=patch_root or data_cfg["patch_root"],
        species=data_cfg["species"],
        fcfg=data_cfg.get("filename", {}),
        val_ratio=sc["val_ratio"],
        test_ratio=sc["test_ratio"],
        seed=seed,
        n_folds=sc["n_folds"],
        fold_index=sc["fold_index"],
    )


def split_by_tree(
        patch_root: str,
        species: List[str],
        fcfg: Dict,
        val_ratio: float,
        seed: int = 42,
) -> Tuple[List[Bag], List[Bag]]:
    """Backwards-compatible two-way split (train, val), no test holdout.

    Kept for hyperparameter_tuning.py. The tree assignment is NOT bit-identical to the
    pre-2026 implementation (the RNG is now per-class instead of one global stream), so
    a split produced here will not reproduce an old run's exact membership.
    """
    splits = split_trees(patch_root, species, fcfg, val_ratio, test_ratio=0.0, seed=seed)
    return splits["train"], splits["val"]


def _stratified_subset(bags: List[Bag], ratio: float, seed: int) -> List[Bag]:
    """Keep a per-class fraction of the bags (used to speed up hyperparameter search)."""
    if ratio >= 1.0:
        return bags
    by_label: Dict[int, List[Bag]] = defaultdict(list)
    for bag in bags:
        by_label[bag[0][1]].append(bag)

    subset: List[Bag] = []
    for label, label_bags in by_label.items():
        random.seed(seed)
        random.shuffle(label_bags)
        keep = max(1, int(len(label_bags) * ratio))
        subset.extend(label_bags[:keep])

    random.shuffle(subset)
    print(f"Subset: reduced to {len(subset)} stratified training images.")
    return subset


def _select_patches(bag: Bag, max_patches, seed: int, epoch: int = 0,
                    stochastic: bool = False) -> Bag:
    """Subsample one bag down to ``max_patches``, or return it unchanged.

    Two modes:

    * **deterministic** (``stochastic=False``) -- the RNG is seeded from the bag's first
      filename alone, so the image always yields the SAME subset. Use for val/test, where
      a bag that changes between epochs would make the val curve noisy and early stopping
      jumpy.
    * **stochastic** (``stochastic=True``) -- the epoch is mixed into the seed, so every
      epoch sees a DIFFERENT subset of the same image. Over a long run the model sees all
      of the image's patches, and the cap becomes free augmentation instead of throwing
      ~40% of the training patches away permanently. Use for Stage-2 training.

    Still reproducible either way: same (seed, filename, epoch) -> same subset.
    """
    if not max_patches or max_patches <= 0 or len(bag) <= max_patches:
        return bag
    key = zlib.crc32(Path(bag[0][0]).stem.encode())
    salt = (epoch * 0x9E3779B1) if stochastic else 0
    rng = random.Random((seed ^ key ^ salt) & 0xFFFFFFFF)
    return rng.sample(bag, max_patches)


def cap_report(bags: List[Bag], max_patches, label: str = "") -> None:
    """Print how many bags the cap actually bites on. Worth logging per split: the
    cap-hit rate is patch-size dependent (a 224 cut yields ~3x more patches per image
    than a 384 cut), so an ablation across patch sizes is also, silently, an ablation
    across how much of each bag the model ever sees."""
    if not max_patches or max_patches <= 0:
        print(f"Bag cap{label}: disabled (full bags).")
        return
    n_over = sum(1 for b in bags if len(b) > max_patches)
    pct = 100.0 * n_over / max(len(bags), 1)
    print(f"Bag cap{label}: {max_patches} patches | {n_over}/{len(bags)} bags "
          f"({pct:.1f}%) exceed it and get subsampled.")


def _cap_bags(bags: List[Bag], max_patches, seed: int) -> List[Bag]:
    """Split-time deterministic cap (test loader, and the legacy Stage-2 path).

    Stage-2 *training* no longer uses this -- it caps inside the dataset so the subset can
    be resampled every epoch. Kept for the deterministic paths and for hyperparameter_tuning.
    """
    if not max_patches or max_patches <= 0:
        return bags
    capped = [_select_patches(b, max_patches, seed) for b in bags]
    n_reduced = sum(1 for b in bags if len(b) > max_patches)
    if n_reduced:
        print(f"Bag cap: {n_reduced} bag(s) exceeded {max_patches} patches and were "
              f"deterministically subsampled to {max_patches}.")
    return capped


def bag_size_stats(bags: List[Bag]) -> Dict[str, int]:
    """min / median / p99 / max patches per bag -- the numbers that drive peak VRAM."""
    sizes = sorted(len(b) for b in bags)
    if not sizes:
        return {}
    return {
        "n_bags": len(sizes),
        "min": sizes[0],
        "median": sizes[len(sizes) // 2],
        "p99": sizes[min(len(sizes) - 1, int(0.99 * len(sizes)))],
        "max": sizes[-1],
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class PatchBagDataset(Dataset):
    """Returns one image's patches stacked as ``[N, C, H, W]`` plus its label.

    Set ``return_id=True`` for inference to also yield the ``"tree:image"`` identifier.
    """

    def __init__(
            self,
            bags: List[Bag],
            transform: Optional[Callable] = None,
            return_id: bool = False,
            fcfg: Optional[Dict] = None,
            max_patches: Optional[int] = None,
            stochastic_cap: bool = False,
            seed: int = 42,
    ):
        self.bags = bags
        self.transform = transform
        self.return_id = return_id
        self.fcfg = fcfg or {}
        # Capping happens HERE rather than at split time so that a stochastic cap can draw
        # a fresh subset every epoch. The full bag is always retained in self.bags.
        self.max_patches = max_patches
        self.stochastic_cap = stochastic_cap
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch so a stochastic cap draws a new subset.

        NOTE: with ``persistent_workers=True`` the worker processes hold their own copy of
        this dataset and would never see the update -- so ``build_dataloaders`` disables
        persistent workers on the Stage-2 train loader. Do not re-enable them without
        replacing this with a shared-memory counter.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = _select_patches(
            self.bags[idx], self.max_patches, self.seed, self.epoch, self.stochastic_cap
        )
        label = bag[0][1]

        tensors = []
        for path, _ in bag:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if self.transform:
                pil = self.transform(pil)
            tensors.append(pil)

        if not tensors:
            raise RuntimeError(f"Every patch of the bag at {bag[0][0]} failed to decode.")

        patches = torch.stack(tensors)
        if self.return_id:
            # Identify by the FULL bag's first patch: a stochastic cap changes which patch
            # lands at index 0, and the image id must not wobble between epochs.
            tree, image = _parse_ids(self.bags[idx][0][0].name, self.fcfg)
            return patches, label, f"{tree}:{image}"
        return patches, label


class PatchDataset(Dataset):
    """Flat per-patch dataset for Stage-1 pretraining (one patch -> one label)."""

    def __init__(self, patches: List[Tuple[Path, int]], transform: Optional[Callable] = None):
        self.patches = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        path, label = self.patches[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read patch: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self.transform:
            pil = self.transform(pil)
        return pil, label


def flatten_bags(bags: List[Bag]) -> List[Tuple[Path, int]]:
    """Flatten image-level bags into a flat list of (patch_path, label)."""
    return [pair for bag in bags for pair in bag]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _loader_kwargs(data_cfg: Dict, persistent: bool = True) -> Dict:
    num_workers = int(data_cfg.get("num_workers", 8) or 0)
    kwargs: Dict = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return kwargs


def build_dataloaders(cfg: Dict, seed: int = 42, train_subset_ratio: float = 1.0):
    """Stage-2 train/val loaders. Batch size is fixed to 1 (one bag); the effective
    batch size is controlled by gradient accumulation in the training loop.

    The bag cap is applied inside the dataset, not at split time:

    * **train** -- stochastic. Each epoch draws a different ``max_patches_per_bag`` subset
      of every oversized image, so over a full run the model sees all of its patches.
      A deterministic cap would permanently discard ~40% of the training patches at 224.
    * **val** -- deterministic. A val bag that changed between epochs would add noise to
      the exact curve early stopping and checkpointing read.
    """
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]

    splits = get_splits(cfg, seed=seed)
    train_bags = _stratified_subset(splits["train"], train_subset_ratio, seed)
    val_bags = splits["val"]

    max_patches = data_cfg.get("max_patches_per_bag")
    cap_report(train_bags, max_patches, label=" [train, stochastic]")
    cap_report(val_bags, max_patches, label=" [val, deterministic]")

    train_ds = PatchBagDataset(
        train_bags, build_train_transform(aug),
        max_patches=max_patches, stochastic_cap=True, seed=seed,
    )
    val_ds = PatchBagDataset(
        val_bags, build_eval_transform(aug),
        max_patches=max_patches, stochastic_cap=False, seed=seed,
    )

    # persistent_workers=False on train: workers hold their own copy of the dataset, so a
    # persistent worker would keep serving epoch 0's subsets forever and the stochastic cap
    # would silently degrade into the deterministic one. Respawning workers costs a couple
    # of seconds against a ~20-minute epoch.
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        **_loader_kwargs(data_cfg, persistent=False),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        **_loader_kwargs(data_cfg, persistent=True),
    )
    return train_loader, val_loader


def build_patch_dataloaders(cfg: Dict, batch_size: int, seed: int = 42):
    """Stage-1 loaders: standard batched patch classification with a tree-level split.

    The tree-level holdout is preserved by splitting first, then flattening each side's
    bags into individual patches. The test trees are excluded here as well, so neither
    stage ever sees a single patch of a test tree.
    """
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]

    splits = get_splits(cfg, seed=seed)
    train_patches = flatten_bags(splits["train"])
    val_patches = flatten_bags(splits["val"])
    print(f"Stage-1 patches: {len(train_patches)} train / {len(val_patches)} val")

    kwargs = _loader_kwargs(data_cfg)
    train_loader = DataLoader(
        PatchDataset(train_patches, build_train_transform(aug)),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs,
    )
    val_loader = DataLoader(
        PatchDataset(val_patches, build_eval_transform(aug)),
        batch_size=batch_size, shuffle=False, **kwargs,
    )
    return train_loader, val_loader


def build_bag_loader(cfg: Dict, split: str = "test", seed: int = 42, shuffle: bool = False):
    """Bag loader over any one split, yielding ``(patches, label, image_id)``.

    Used by test_model.py (AMIL image-level eval) and by the majority-voting baseline,
    which must see exactly the same bags.
    """
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]
    test_cfg = cfg.get("test", {}) or {}

    data_root = test_cfg.get("data_root")
    if split == "test" and data_root and Path(data_root).exists():
        print(f"Test set: legacy folder mode ({data_root}) -- every tree there is test.")
        bags = split_trees(
            patch_root=data_root,
            species=data_cfg["species"],
            fcfg=data_cfg.get("filename", {}),
            val_ratio=0.0,
            test_ratio=1.0,
            seed=seed,
        )["test"]
    else:
        sc = split_config(cfg)
        if split == "test" and sc["test_ratio"] <= 0 and not sc["n_folds"]:
            raise ValueError(
                "No test set available: test.data_root does not exist and data.split "
                "defines no holdout. Set data.split.test_ratio (e.g. 0.2) or "
                "data.split.n_folds, or point test.data_root at a pre-cut test folder."
            )
        bags = get_splits(cfg, seed=seed)[split]

    print(f"{split} bag sizes: {bag_size_stats(bags)}")

    # Test bags are uncapped by default: inference chunks each bag through the extractor
    # (PatchAttentionMIL.infer), so peak memory is bounded by test.chunk_size rather than
    # by bag size. Set test.max_patches_per_bag only if you deliberately want the test
    # bag-size distribution to match the capped training one.
    max_patches = test_cfg.get("max_patches_per_bag")
    if max_patches:
        bags = _cap_bags(bags, max_patches, seed)

    dataset = PatchBagDataset(
        bags, build_eval_transform(aug), return_id=True, fcfg=data_cfg.get("filename", {})
    )
    kwargs = _loader_kwargs(data_cfg)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, **kwargs)


def build_test_loader(cfg: Dict, data_root: Optional[str] = None, seed: int = 42):
    """Backwards-compatible entry point used by test_model.py."""
    if data_root:
        cfg = dict(cfg)
        cfg["test"] = {**(cfg.get("test") or {}), "data_root": data_root}
    return build_bag_loader(cfg, split="test", seed=seed)
