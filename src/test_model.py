"""Evaluate on the held-out test trees. Three image-level predictors, one forward pass.

* **amil**      - Stage-2 attention-MIL. Attention-weighted sum of patch EMBEDDINGS ->
                  shared head -> image logits. This is the method.
* **amil_vote** - the SAME Stage-2 model, but its per-patch logits (from the auxiliary
                  instance head) are aggregated by hard majority vote instead of by
                  attention. Same weights, same forward pass, same bags -- the ONLY
                  difference is the aggregation rule.
* **vote**      - Stage-1 patch classifier + hard majority vote. The prior-work pipeline
                  (Carpentier et al. classify crops and vote over them).

Why `amil_vote` exists, and why it is the headline comparison: `amil` vs `vote` confounds
*aggregation* with *extra training* -- the AMIL model has had more gradient steps than the
Stage-1 backbone, so a win could just mean "trained longer". `amil` vs `amil_vote` holds the
weights fixed and varies only the aggregation rule, which is the claim the paper makes.
Both comparisons are reported, each with a paired McNemar test.

BAG SIZE. Training caps bags at ``data.max_patches_per_bag`` (resampled every epoch); test
bags are much larger (median ~144, max 336 at 224px). Two evaluations are run:

* **full** (primary)      - uncapped. Uses all the information in the image. Inference
                            streams each bag through the extractor in chunks
                            (``test.chunk_size``), so peak VRAM is bounded by the chunk.
* **capped** (robustness) - bags capped to the training cap. Answers "does attention
                            pooling degrade once N exceeds anything it saw in training?"
                            -- a real risk, since a softmax over more instances is flatter
                            and drifts toward plain mean pooling.

Usage:
    python test_model.py -c config/config_cluster.yaml \
        --checkpoint $SCRATCH/runs/x/train/best_model.pth \
        --backbone-checkpoint $SCRATCH/runs/x/pretrain/best_backbone.pth \
        --output-dir $SCRATCH/runs/x/test
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import helper.data_loader as dt
from helper.config_cli import add_config_args, dump_config, load_config
from helper.model_wrapper import ConvNeXtAMIL

PREDICTORS = [
    ("amil", "attention pooling (Stage-2)"),
    ("amil_vote", "majority vote over Stage-2's OWN patch head"),
    ("vote", "majority vote over the Stage-1 backbone"),
]


def mcnemar(a_correct, b_correct):
    """Exact McNemar test on paired per-image decisions.

    Only disagreements carry information: images both predictors get right (or both get
    wrong) say nothing about which is better. Under the null ("equally good") the
    disagreements should split 50/50, so this is a two-sided binomial test on that split.
    """
    b = int(sum(1 for x, y in zip(a_correct, b_correct) if x and not y))   # A only
    c = int(sum(1 for x, y in zip(a_correct, b_correct) if y and not x))   # B only
    n = b + c
    out = {"a_only_correct": b, "b_only_correct": c, "discordant": n}
    if n == 0:
        out["p_value"] = 1.0
        return out
    try:
        from scipy.stats import binomtest
        out["p_value"] = float(binomtest(b, n, 0.5).pvalue)
    except Exception:  # scipy missing -- normal approximation, continuity corrected
        chi2 = (abs(b - c) - 1) ** 2 / n
        out["p_value"] = float(np.exp(-chi2 / 2))
        out["p_value_note"] = "normal approximation (scipy unavailable)"
    return out


def majority_vote(patch_logits, species):
    """Hard majority vote over per-patch argmax. Ties -> lowest class index (deterministic)."""
    preds = patch_logits.argmax(dim=1).cpu().tolist()
    counts = Counter(preds)
    top = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return species[top[0]], top[1] / max(len(preds), 1)


def evaluate(cfg, species, amil_model, vote_model, chunk_size, device, seed, cap, tag):
    """Score every available predictor over one bag distribution (capped or uncapped)."""
    cfg = {**cfg, "test": {**cfg["test"], "max_patches_per_bag": cap}}
    loader = dt.build_bag_loader(cfg, split="test", seed=seed)

    rows = []
    with torch.no_grad():
        for patches, true_label, image_id in tqdm(loader, desc=f"Testing [{tag}]"):
            patches = patches.squeeze(0).to(device)
            entry = {
                "image_id": image_id[0],
                "true_class": species[int(true_label.item())],
                "n_patches": int(patches.size(0)),
            }

            if amil_model is not None:
                # ONE forward pass gives both the attention-pooled image logits and the
                # per-patch logits from the auxiliary head, so `amil` and `amil_vote` are
                # strictly the same weights on the same activations.
                image_logits, patch_logits, attn = amil_model.infer(patches, chunk_size=chunk_size)

                probs = torch.softmax(image_logits.float(), dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                entry["amil_pred"] = species[pred]
                entry["amil_conf"] = float(probs[pred])
                entry["amil_attn_max"] = float(attn.max().item())
                # Attention entropy: low = concentrated on a few patches; high = spread out
                # (i.e. behaving like mean pooling). Diagnostic for the bag-size question
                # and material for the interpretability section.
                a = attn.float().flatten()
                entry["amil_attn_entropy"] = float(-(a * (a + 1e-12).log()).sum().item())

                entry["amil_vote_pred"], entry["amil_vote_share"] = majority_vote(
                    patch_logits, species)

            if vote_model is not None:
                logits = vote_model.infer(patches, chunk_size=chunk_size)
                entry["vote_pred"], entry["vote_share"] = majority_vote(logits, species)

            rows.append(entry)

    return pd.DataFrame(rows)


def score(df, species, key):
    cm = confusion_matrix(df["true_class"].tolist(), df[key].tolist(), labels=species)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return float(accuracy_score(df["true_class"].tolist(), df[key].tolist())), cm, per_class


def main():
    parser = argparse.ArgumentParser(description="BarkNet-AMIL evaluation")
    add_config_args(parser)
    parser.add_argument("--mode", choices=["amil", "vote", "both"], default="both")
    parser.add_argument("--bags", choices=["full", "capped", "both"], default="both",
                        help="full = uncapped test bags (primary); capped = capped to the "
                             "training cap (robustness check on the bag-size shift).")
    args = parser.parse_args()

    cfg = load_config(args, stage="test")
    device = torch.device(cfg["project"]["device"])
    species = cfg["data"]["species"]
    model_cfg, test_cfg = cfg["model"], cfg["test"]
    seed = cfg["project"].get("seed", 42)
    chunk_size = int(test_cfg.get("chunk_size", 64))
    train_cap = cfg["data"].get("max_patches_per_bag")

    results_dir = Path(test_cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, results_dir)

    amil_ckpt = test_cfg.get("checkpoint")
    backbone_ckpt = model_cfg.get("backbone_checkpoint")
    do_amil = args.mode in ("amil", "both") and amil_ckpt and Path(amil_ckpt).exists()
    do_vote = args.mode in ("vote", "both") and backbone_ckpt and Path(backbone_ckpt).exists()
    if args.mode == "amil" and not do_amil:
        raise FileNotFoundError(f"test.checkpoint not found: {amil_ckpt}")
    if args.mode == "vote" and not do_vote:
        raise FileNotFoundError(f"model.backbone_checkpoint not found: {backbone_ckpt}")
    if not do_amil and not do_vote:
        raise FileNotFoundError("Neither an AMIL checkpoint nor a Stage-1 backbone was found.")

    amil_model = vote_model = None
    if do_amil:
        amil_model, _ = ConvNeXtAMIL(
            device=device, species=species, model_size=model_cfg["size"], weights="none",
            pretrained_checkpoint=amil_ckpt, save_data=False,
            channels_last=False,   # inference is fp32 and chunked; keep it simple
        ).get_model()
        amil_model.eval()
    if do_vote:
        vote_model, _ = ConvNeXtAMIL(
            device=device, species=species, model_size=model_cfg["size"], weights="none",
            backbone_checkpoint=backbone_ckpt, save_data=False, channels_last=False,
        ).get_patch_model()
        vote_model.eval()

    passes = []
    if args.bags in ("full", "both"):
        passes.append(("full", None))
    if args.bags in ("capped", "both") and train_cap:
        passes.append((f"capped_{int(train_cap)}", int(train_cap)))
    if not passes:
        passes = [("full", None)]

    summary = {
        "n_classes": len(species),
        "model_size": model_cfg["size"],
        "input_size": cfg["augmentation"]["input_size"],
        "patch_root": cfg["data"]["patch_root"],
        "train_bag_cap": train_cap,
        "evaluations": {},
    }
    sheets, frames = {}, {}

    for tag, cap in passes:
        df = evaluate(cfg, species, amil_model, vote_model, chunk_size, device, seed, cap, tag)
        frames[tag] = df
        df.to_csv(results_dir / f"test_predictions_{tag}.csv", index=False)

        block = {
            "n_images": len(df),
            "bag_cap": cap,
            "mean_bag_size": float(df["n_patches"].mean()),
            "max_bag_size": int(df["n_patches"].max()),
        }
        print(f"\n--- {tag} bags (n={len(df)}, mean {block['mean_bag_size']:.0f} patches) ---")

        for name, label in PREDICTORS:
            key = f"{name}_pred"
            if key not in df.columns:
                continue
            acc, cm, per_class = score(df, species, key)
            block[f"{name}_accuracy"] = acc
            sheets[f"CM_{name}_{tag}"[:31]] = pd.DataFrame(
                cm, index=[f"True_{s}" for s in species],
                columns=[f"Pred_{s}" for s in species])
            sheets[f"PC_{name}_{tag}"[:31]] = pd.DataFrame(
                {"class": species, "accuracy": per_class, "support": cm.sum(axis=1)})
            print(f"  {name:<10} {acc:.4f}   ({label})")

        if "amil_attn_entropy" in df.columns:
            block["mean_attn_entropy"] = float(df["amil_attn_entropy"].mean())

        correct = {n: (df[f"{n}_pred"] == df["true_class"]).tolist()
                   for n, _ in PREDICTORS if f"{n}_pred" in df.columns}

        for a, b, why in (
            ("amil", "amil_vote", "MECHANISM: same weights, attention vs voting"),
            ("amil", "vote", "PIPELINE: AMIL vs prior-work Stage-1 + vote"),
        ):
            if a in correct and b in correct:
                delta = block[f"{a}_accuracy"] - block[f"{b}_accuracy"]
                mc = mcnemar(correct[a], correct[b])
                block[f"{a}_vs_{b}"] = {"delta_pp": 100.0 * delta, "mcnemar": mc}
                print(f"  {a} - {b}: {100 * delta:+.2f} pp | p={mc['p_value']:.3g}  [{why}]")

        summary["evaluations"][tag] = block

    excel_path = results_dir / "classification_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        flat = [{"bags": tag, **{k: v for k, v in block.items() if not isinstance(v, dict)}}
                for tag, block in summary["evaluations"].items()]
        pd.DataFrame(flat).to_excel(writer, sheet_name="Summary", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name)
        for tag, df in frames.items():
            df.to_excel(writer, sheet_name=f"Pred_{tag}"[:31], index=False)

    with open(results_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved to {excel_path}")


if __name__ == "__main__":
    main()
