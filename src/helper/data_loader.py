"""Data pipeline for pre-cut bark patches.

Directory layout expected on disk (produced by the patch-cutting step, rebuilt later)::

    patch_root/
        <SPECIES_CODE>/                 # folder name == class label
            <tree>_<image>_<patch>.png  # filename encodes tree id, image id, patch index
            ...

Two grouping keys matter and they are *different*:

* **Split granularity = tree.** Every patch of a given tree lands entirely in train OR
  in val/test. This prevents the same-tree leakage that inflates image-level random
  splits (the methodology issue we are correcting relative to prior work).
* **Bag granularity = image.** The unit the model sees as ``[N, C, H, W]`` is the set of
  patches cut from a single image.

Tree/image ids are parsed from the filename via the ``data.filename`` config block
(delimiter + token indices), so the convention can change without touching code.
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

IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

# A bag is a list of (patch_path, label) tuples for a single image.
Bag = List[Tuple[Path, int]]


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
# Splitting
# --------------------------------------------------------------------------- #
def _parse_ids(filename: str, fcfg: Dict) -> Tuple[str, str]:
    """Extract (tree_id, image_id) tokens from a patch filename."""
    parts = Path(filename).stem.split(fcfg.get("delimiter", "_"))
    tree = parts[fcfg.get("tree_id_index", 0)]
    image = parts[fcfg.get("image_id_index", 1)]
    return tree, image


def split_by_tree(
        patch_root: str,
        species: List[str],
        fcfg: Dict,
        val_ratio: float,
        seed: int = 42,
) -> Tuple[List[Bag], List[Bag]]:
    """Scan ``patch_root`` and split into image-level bags with a tree-level holdout.

    Returns ``(train_bags, val_bags)`` where each bag is the patch list of one image.
    """
    random.seed(seed)
    root = Path(patch_root)
    class_to_idx = {name: i for i, name in enumerate(species)}

    # tree_key -> image_key -> [(path, label), ...]
    trees: Dict[str, Dict[str, Bag]] = defaultdict(lambda: defaultdict(list))
    tree_label: Dict[str, int] = {}

    print("Scanning patches and grouping by tree -> image ...")
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

    by_label: Dict[int, List[str]] = defaultdict(list)
    for tree_key, label in tree_label.items():
        by_label[label].append(tree_key)

    train_bags: List[Bag] = []
    val_bags: List[Bag] = []
    for label, tree_keys in by_label.items():
        random.shuffle(tree_keys)
        n_val = int(len(tree_keys) * val_ratio)
        val_set = set(tree_keys[:n_val])
        for tree_key in tree_keys:
            target = val_bags if tree_key in val_set else train_bags
            for _, bag in trees[tree_key].items():
                target.append(bag)

    print(
        f"Split complete: {len(train_bags)} train images / {len(val_bags)} val images "
        f"across {len(tree_label)} trees."
    )
    return train_bags, val_bags


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


def _cap_bags(bags: List[Bag], max_patches, seed: int) -> List[Bag]:
    """Limit each bag to at most ``max_patches`` patches (Stage-2 memory control).

    Bags at or under the cap are returned unchanged. Oversized bags are subsampled
    deterministically -- the per-bag RNG is seeded from the bag's first filename, so a
    given image always yields the SAME subset across epochs, runs, and the train/val
    split. This keeps results reproducible and is safe for dense bark texture, where
    every patch carries similar signal. Pass ``None`` / 0 / negative to disable.
    """
    if not max_patches or max_patches <= 0:
        return bags

    capped: List[Bag] = []
    n_reduced = 0
    for bag in bags:
        if len(bag) <= max_patches:
            capped.append(bag)
            continue
        key = Path(bag[0][0]).stem.encode()
        rng = random.Random(seed ^ zlib.crc32(key))
        capped.append(rng.sample(bag, max_patches))
        n_reduced += 1

    if n_reduced:
        print(f"Bag cap: {n_reduced} bag(s) exceeded {max_patches} patches and were "
              f"deterministically subsampled to {max_patches}.")
    return capped


# --------------------------------------------------------------------------- #
# Dataset
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
    ):
        self.bags = bags
        self.transform = transform
        self.return_id = return_id
        self.fcfg = fcfg or {}

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = self.bags[idx]
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

        patches = torch.stack(tensors)
        if self.return_id:
            tree, image = _parse_ids(bag[0][0].name, self.fcfg)
            return patches, label, f"{tree}:{image}"
        return patches, label


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: Dict, seed: int = 42, train_subset_ratio: float = 1.0):
    """Build train/val loaders. Batch size is fixed to 1 (one bag); the effective
    batch size is controlled by gradient accumulation in the training loop."""
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]

    train_bags, val_bags = split_by_tree(
        patch_root=data_cfg["patch_root"],
        species=data_cfg["species"],
        fcfg=data_cfg.get("filename", {}),
        val_ratio=data_cfg["val_ratio"],
        seed=seed,
    )
    train_bags = _stratified_subset(train_bags, train_subset_ratio, seed)

    # Stage-2 memory control: cap patches per bag so a single oversized image (e.g.
    # a high-res photo yielding hundreds of patches) can't dictate peak VRAM. Set
    # data.max_patches_per_bag to null/0 to disable. Applies to both the board-level
    # hyperparameter search and training (both go through this function).
    max_patches = data_cfg.get("max_patches_per_bag")
    train_bags = _cap_bags(train_bags, max_patches, seed)
    val_bags = _cap_bags(val_bags, max_patches, seed)

    num_workers = data_cfg.get("num_workers", 8)
    train_loader = DataLoader(
        PatchBagDataset(train_bags, build_train_transform(aug)),
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        PatchBagDataset(val_bags, build_eval_transform(aug)),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    return train_loader, val_loader


def build_test_loader(cfg: Dict, data_root: str, seed: int = 42):
    """Build a single loader over an entire patch root (val_ratio=1.0), yielding ids."""
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]
    _, test_bags = split_by_tree(
        patch_root=data_root,
        species=data_cfg["species"],
        fcfg=data_cfg.get("filename", {}),
        val_ratio=1.0,
        seed=seed,
    )
    dataset = PatchBagDataset(
        test_bags, build_eval_transform(aug), return_id=True, fcfg=data_cfg.get("filename", {})
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_cfg.get("num_workers", 4))


# --------------------------------------------------------------------------- #
# Stage-1 patch-level loaders (standard classification, no bags)
# --------------------------------------------------------------------------- #
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


def build_patch_dataloaders(cfg: Dict, batch_size: int, seed: int = 42):
    """Stage-1 loaders: standard batched patch classification with a tree-level split.

    The tree-level holdout is preserved by splitting first, then flattening the bags
    on each side into individual patches.
    """
    data_cfg = cfg["data"]
    aug = cfg["augmentation"]

    train_bags, val_bags = split_by_tree(
        patch_root=data_cfg["patch_root"],
        species=data_cfg["species"],
        fcfg=data_cfg.get("filename", {}),
        val_ratio=data_cfg["val_ratio"],
        seed=seed,
    )
    train_patches = flatten_bags(train_bags)
    val_patches = flatten_bags(val_bags)
    print(f"Stage-1 patches: {len(train_patches)} train / {len(val_patches)} val")

    num_workers = data_cfg.get("num_workers", 8)
    train_loader = DataLoader(
        PatchDataset(train_patches, build_train_transform(aug)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        PatchDataset(val_patches, build_eval_transform(aug)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    return train_loader, val_loader
