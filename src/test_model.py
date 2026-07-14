"""Evaluate on the held-out test trees: AMIL vs. hard majority voting.

Two image-level predictors are scored on exactly the same bags:

* **amil** - the Stage-2 attention-MIL model (attention-pooled bag prediction; the
  auxiliary per-patch head is not used at inference).
* **vote** - the Stage-1 patch classifier + hard majority vote over the image's patches.
  This is what prior bark work does, and it is the baseline the whole project exists to
  beat (or to fail to beat, honestly).

Running both in one pass means the predictions are paired image-by-image, which is what
McNemar's test needs.

Bags are NOT capped here: inference streams each bag through the extractor in chunks
(``test.chunk_size``), so peak VRAM is bounded by the chunk, not by a 336-patch outlier.

Usage:
    python test_model.py -c configs/config.yaml
    python test_model.py -c configs/config_cluster.yaml --mode both \
        --checkpoint $SCRATCH/runs/smoke/train/best_model.pth \
        --backbone-checkpoint $SCRATCH/runs/smoke/pretrain/best_backbone.pth \
        --output-dir $SCRATCH/runs/smoke/test
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


def mcnemar(amil_correct, vote_correct):
    """Exact McNemar test on the paired image-level decisions."""
    b = int(sum(1 for a, v in zip(amil_correct, vote_correct) if a and not v))  # AMIL only
    c = int(sum(1 for a, v in zip(amil_correct, vote_correct) if v and not a))  # vote only
    n = b + c
    out = {"amil_only_correct": b, "vote_only_correct": c, "discordant": n}
    if n == 0:
        out["p_value"] = 1.0
        return out
    try:
        from scipy.stats import binomtest
        out["p_value"] = float(binomtest(b, n, 0.5).pvalue)
    except Exception:  # scipy missing -- normal approximation with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / n
        out["p_value"] = float(np.exp(-chi2 / 2))
        out["p_value_note"] = "normal approximation (scipy unavailable)"
    return out


def summarise(results, species, key):
    y_true = [r["true_class"] for r in results]
    y_pred = [r[key] for r in results]
    cm = confusion_matrix(y_true, y_pred, labels=species)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return accuracy_score(y_true, y_pred), cm, per_class


def main():
    parser = argparse.ArgumentParser(description="barknet-AMIL evaluation")
    add_config_args(parser)
    parser.add_argument("--mode", choices=["amil", "vote", "both"], default="both",
                        help="amil = Stage-2 model, vote = Stage-1 backbone + majority vote.")
    args = parser.parse_args()

    cfg = load_config(args, stage="test")

    device = torch.device(cfg["project"]["device"])
    species = cfg["data"]["species"]
    model_cfg = cfg["model"]
    test_cfg = cfg["test"]
    seed = cfg["project"].get("seed", 42)
    chunk_size = int(test_cfg.get("chunk_size", 64))

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

    amil_model = None
    if do_amil:
        amil_model, _ = ConvNeXtAMIL(
            device=device, species=species, model_size=model_cfg["size"],
            weights="none",                       # weights come from the checkpoint
            pretrained_checkpoint=amil_ckpt,
            save_data=False,
        ).get_model()
        amil_model.eval()

    vote_model = None
    if do_vote:
        vote_model, _ = ConvNeXtAMIL(
            device=device, species=species, model_size=model_cfg["size"],
            weights="none",
            backbone_checkpoint=backbone_ckpt,    # loads into PatchClassifier (strict)
            save_data=False,
        ).get_patch_model()
        vote_model.eval()

    print(f"Evaluating: amil={do_amil} vote={do_vote} | chunk_size={chunk_size}")
    test_loader = dt.build_bag_loader(cfg, split="test", seed=seed)

    results = []
    with torch.no_grad():
        for patches, true_label, image_id in tqdm(test_loader, desc="Testing images"):
            patches = patches.squeeze(0).to(device)
            true_label = int(true_label.item())
            image_id = image_id[0]

            entry = {
                "image_id": image_id,
                "true_class": species[true_label],
                "n_patches": int(patches.size(0)),
            }

            if do_amil:
                image_logits, _, attn = amil_model.infer(patches, chunk_size=chunk_size)
                probs = torch.softmax(image_logits.float(), dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                entry["amil_pred"] = species[pred]
                entry["amil_conf"] = float(probs[pred])
                entry["amil_attn_max"] = float(attn.max().item())
                for idx, p in enumerate(probs):
                    entry[f"amil_p_{species[idx]}"] = float(p)

            if do_vote:
                patch_logits = vote_model.infer(patches, chunk_size=chunk_size)
                patch_preds = patch_logits.argmax(dim=1).cpu().tolist()
                counts = Counter(patch_preds)
                # Ties broken by the lowest class index (deterministic), as in a plain vote.
                top = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
                entry["vote_pred"] = species[top[0]]
                entry["vote_share"] = top[1] / max(len(patch_preds), 1)

            results.append(entry)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "test_predictions.csv", index=False)

    summary = {
        "n_images": len(results),
        "n_classes": len(species),
        "model_size": model_cfg["size"],
        "input_size": cfg["augmentation"]["input_size"],
        "patch_root": cfg["data"]["patch_root"],
    }
    sheets = {}

    for name, key in (("amil", "amil_pred"), ("vote", "vote_pred")):
        if key not in df.columns:
            continue
        acc, cm, per_class = summarise(results, species, key)
        summary[f"{name}_image_accuracy"] = float(acc)
        sheets[f"CM_{name}"] = pd.DataFrame(
            cm, index=[f"True_{s}" for s in species], columns=[f"Pred_{s}" for s in species]
        )
        sheets[f"PerClass_{name}"] = pd.DataFrame(
            {"class": species, "accuracy": per_class, "support": cm.sum(axis=1)}
        )
        print(f"{name.upper():<5} image-level accuracy: {acc:.4f}")

    if do_amil and do_vote:
        amil_correct = (df["amil_pred"] == df["true_class"]).tolist()
        vote_correct = (df["vote_pred"] == df["true_class"]).tolist()
        summary["mcnemar"] = mcnemar(amil_correct, vote_correct)
        summary["amil_minus_vote"] = summary["amil_image_accuracy"] - summary["vote_image_accuracy"]
        print(f"AMIL - vote: {summary['amil_minus_vote']*100:+.2f} pp | McNemar {summary['mcnemar']}")

    excel_path = results_dir / "classification_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, dict)}]).to_excel(
            writer, sheet_name="Summary", index=False
        )
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31])
        df.to_excel(writer, sheet_name="Predictions", index=False)

    with open(results_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Report saved to {excel_path}")


if __name__ == "__main__":
    main()
