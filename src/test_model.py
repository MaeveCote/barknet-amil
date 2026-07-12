"""Evaluate a trained ConvNeXt + AMIL checkpoint on a held-out patch set.

Predictions are image-level (attention-pooled); the auxiliary per-patch head is not
used at inference. Writes an Excel workbook with per-class results, a confusion matrix,
and per-class accuracy, plus a flat per-image predictions CSV.

Usage:
    python test_model.py -c configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import helper.data_loader as dt
from helper.model_wrapper import ConvNeXtAMIL

def main():
    parser = argparse.ArgumentParser(description="barknet-AMIL evaluation")
    parser.add_argument("-c", "--config", default="configs/config.yaml", type=str)
    args = parser.parse_args()

    with open(Path(args.config).resolve(), "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["project"]["device"])
    species = cfg["data"]["species"]
    model_cfg = cfg["model"]
    test_cfg = cfg["test"]

    results_dir = Path(test_cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    wrapper = ConvNeXtAMIL(
        device=device,
        species=species,
        model_size=model_cfg["size"],
        weights="none",  # weights come from the checkpoint, not ImageNet
        pretrained_checkpoint=test_cfg["checkpoint"],
        save_data=False,
    )
    model, _ = wrapper.get_model()
    model.eval()

    test_loader = dt.build_test_loader(cfg, data_root=test_cfg["data_root"], seed=cfg["project"].get("seed", 42))

    results = []
    per_class = {name: [] for name in species}

    with torch.no_grad():
        for patches, true_label, image_id in tqdm(test_loader, desc="Testing images"):
            patches = patches.squeeze(0).to(device)
            true_label = true_label.item()
            image_id = image_id[0]

            image_logits, _ = model(patches)
            probs = torch.softmax(image_logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

            entry = {
                "image_id": image_id,
                "true_class": species[true_label],
                "predicted_class": species[pred],
                "confidence": float(probs[pred]),
            }
            for idx, p in enumerate(probs):
                entry[f"{species[idx]}_prob"] = float(p)

            results.append(entry)
            per_class[species[true_label]].append(entry)

    # Flat per-image predictions (one row per test image) for easy downstream analysis.
    csv_path = results_dir / "classification_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    excel_path = results_dir / "classification_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name in species:
            if per_class[name]:
                pd.DataFrame(per_class[name]).to_excel(
                    writer, sheet_name=f"{name}"[:31], index=False
                )

        y_true = [r["true_class"] for r in results]
        y_pred = [r["predicted_class"] for r in results]
        cm = confusion_matrix(y_true, y_pred, labels=species)
        pd.DataFrame(
            cm, index=[f"True_{s}" for s in species], columns=[f"Pred_{s}" for s in species]
        ).to_excel(writer, sheet_name="Confusion_Matrix")

        overall = accuracy_score(y_true, y_pred)
        class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        pd.DataFrame(
            {
                "Metric": ["Overall_Accuracy"] + [f"{s}_Accuracy" for s in species],
                "Value": [overall] + list(class_acc),
            }
        ).to_excel(writer, sheet_name="Performance_Metrics", index=False)

    print(f"Overall accuracy: {overall:.4f}")
    print(f"Report saved to {excel_path}")
    print(f"Per-image predictions saved to {csv_path}")

if __name__ == "__main__":
    main()
