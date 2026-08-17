#!/usr/bin/env python3
r"""Standalone WHOLE-IMAGE evaluation -- bypasses the bag/MIL loader entirely.

WHY THIS EXISTS
test_model.py always groups files into bags (tree -> image -> patches) and there is no
reliable "one image = one prediction" path: the image_id override kept collapsing crops of
a tree into a single bag, so the reported number was tree-level voting, not whole-image
classification. This script sidesteps that completely: it loads each test image, resizes to
224, and runs it straight through the trained per-image classifier. One image -> one
prediction. No bags, no voting, no aggregation.

WHAT IT REPRODUCES
It calls the SAME split_trees() the training used (same seed, same n_folds, same fold), so
the held-out test trees are exactly the ones the backbone never saw. Then it flattens those
trees to their individual images rather than grouping them.

WHAT best_backbone.pth IS
A PatchClassifier state_dict: extractor.* (ConvNeXt-V2) + classifier.* (Linear -> 20). Its
forward(x) takes [N,C,H,W] and returns [N,20] logits. Feeding a whole resized image is
exactly the per-image classification we want.

USAGE (local, one fold or all)
    python eval_wholeimage.py \
        --image-root "G:\path\to\barknet\dataset" \
        --backbone   "G:\...\runs_wholeimage\wholeimg_f0\pretrain\best_backbone.pth" \
        --repo-root  "G:\Source\BarkNet_ML\BarkNet_ML" \
        --fold 0 --out fold0_preds.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_split_module(repo_root: Path):
    """Import the project's split + config utilities so the split matches training exactly."""
    src = repo_root / "src"
    sys.path.insert(0, str(src))
    import helper.data_loader as dl          # noqa: E402
    from helper.config_cli import load_config  # noqa: E402
    return dl, load_config


def build_classifier(repo_root: Path, model_size: str, num_classes: int,
                     drop_path: float, device: str):
    """Reconstruct the PatchClassifier the checkpoint was saved from."""
    sys.path.insert(0, str(repo_root / "src"))
    import timm
    from helper.model import PatchClassifier

    SIZE_TO_TAG = {
        "atto": "convnextv2_atto.fcmae_ft_in1k", "femto": "convnextv2_femto.fcmae_ft_in1k",
        "pico": "convnextv2_pico.fcmae_ft_in1k", "nano": "convnextv2_nano.fcmae_ft_in1k",
        "tiny": "convnextv2_tiny.fcmae_ft_in1k", "base": "convnextv2_base.fcmae_ft_in1k",
    }
    tag = SIZE_TO_TAG[model_size]
    extractor = timm.create_model(tag, pretrained=False, num_classes=0, drop_path_rate=drop_path)
    in_features = extractor.num_features
    model = PatchClassifier(extractor, in_features, num_classes, dropout_p_classifier=0.0)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True, help="folder of <species>/<image>.jpg (whole images)")
    ap.add_argument("--backbone", required=True, help="best_backbone.pth for this fold")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", default=None, help="defaults to <repo>/configs/config.yaml")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-size", default="nano")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--drop-path", type=float, default=0.257)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="per-image predictions CSV")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    cfg_path = Path(args.config) if args.config else repo / "configs" / "config.yaml"
    dl, load_config = load_split_module(repo)

    # ---- load config purely to get species list + normalization stats ---------------
    class A:  # minimal args object for load_config
        pass
    a = A()
    a.config = str(cfg_path)
    a.overrides = [f"data.split.n_folds={args.n_folds}", f"data.split.fold_index={args.fold}"]
    for k in ("patch_root", "output_dir", "epochs", "model_size", "input_size",
              "num_workers", "device", "seed", "fold", "backbone_checkpoint", "checkpoint"):
        setattr(a, k, None)
    a.patch_root = args.image_root
    a.fold = args.fold
    cfg = load_config(a, stage="pretrain")

    species = cfg["data"]["species"]
    num_classes = len(species)
    norm = cfg["augmentation"].get("normalize") or cfg["augmentation"]["normalization"]
    mean = torch.tensor(norm["mean"]).view(3, 1, 1)
    std = torch.tensor(norm["std"]).view(3, 1, 1)
    class_to_idx = {s: i for i, s in enumerate(species)}

    # ---- reproduce the EXACT split, then flatten test trees to individual images -----
    fcfg = cfg["data"]["filename"]
    # split_trees returns bags grouped tree->image->[patches]; we only need which files are
    # in the TEST split. Re-scan and select the test trees, then take every file individually.
    splits = dl.split_trees(
        args.image_root, species, fcfg,
        val_ratio=cfg["data"]["split"].get("val_ratio", 0.15),
        test_ratio=cfg["data"]["split"]["test_ratio"],
        seed=args.seed,
        n_folds=args.n_folds,
        fold_index=args.fold,
    )
    test_bags = splits["test"]
    # each bag is a list of (path, label); flatten to individual (path, label) images
    images = [(p, label) for bag in test_bags for (p, label) in bag]
    print(f"fold {args.fold}: {len(test_bags)} test trees -> {len(images)} whole images")

    if not images:
        print("No test images -- check image_root and split settings.")
        return

    # ---- build model + load checkpoint ----------------------------------------------
    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_classifier(repo, args.model_size, num_classes, args.drop_path, device)
    state = torch.load(args.backbone, map_location="cpu")
    # strip a possible torch.compile prefix
    state = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  ! {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    n_loaded = sum(1 for _ in state)
    print(f"  loaded {n_loaded} tensors from {Path(args.backbone).name}"
          f" (missing {len(missing)}, unexpected {len(unexpected)})")
    model.to(device).eval()

    # ---- inference: resize -> normalize -> forward -> argmax ------------------------
    S = args.input_size
    rows, correct = [], 0
    batch_imgs, batch_meta = [], []

    def flush():
        nonlocal correct
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            logits = model(x)                       # [B, num_classes]
            pred = logits.argmax(1).cpu().numpy()
        for (path, label), pr in zip(batch_meta, pred):
            pred_sp = species[pr]
            true_sp = species[label]
            ok = int(pr) == int(label)
            correct += ok
            rows.append({"image": path.name, "true_class": true_sp,
                         "pred_class": pred_sp, "correct": ok})
        batch_imgs.clear(); batch_meta.clear()

    for i, (path, label) in enumerate(images, 1):
        try:
            im = Image.open(path).convert("RGB").resize((S, S), Image.BILINEAR)
        except Exception as exc:
            print(f"  ! skip {path.name}: {exc}")
            continue
        t = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        t = (t - mean) / std
        batch_imgs.append(t)
        batch_meta.append((path, label))
        if len(batch_imgs) >= args.batch_size:
            flush()
        if i % 500 == 0:
            print(f"  ...{i}/{len(images)}")
    flush()

    n = len(rows)
    acc = correct / n if n else 0.0

    # macro-F1
    try:
        from sklearn.metrics import f1_score
        yt = [r["true_class"] for r in rows]
        yp = [r["pred_class"] for r in rows]
        macro_f1 = f1_score(yt, yp, labels=species, average="macro", zero_division=0)
    except Exception:
        macro_f1 = float("nan")

    print(f"\nfold {args.fold}: {n} images | accuracy {acc:.4f} | macro-F1 {macro_f1:.4f}")

    out = Path(args.out) if args.out else Path(f"wholeimage_fold{args.fold}_preds.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_class", "pred_class", "correct"])
        w.writeheader(); w.writerows(rows)
    # a one-line summary sidecar for easy aggregation
    with open(out.with_suffix(".summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n_images", "accuracy", "macro_f1"])
        w.writerow([args.fold, n, f"{acc:.6f}", f"{macro_f1:.6f}"])
    print(f"  wrote {out}  and  {out.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()