#!/usr/bin/env python3
"""Compile BarkNet-AMIL ablation results across folds into a master summary.

Reads the PER-BAG rows that ``test_model.py`` already wrote into each run's
``classification_results.xlsx`` (sheet ``Pred_full`` = uncapped bags, the honest
evaluation).  Nothing has to be re-inferred: ``n_patches`` and ``amil_attn_entropy`` are
recorded per bag, which is everything needed for the attention-concentration statistics.

For every fold it recomputes, from the per-bag rows:
  * image-level accuracy for ``amil``, ``amil_vote``, ``vote``
  * exact McNemar (two-sided binomial on discordant pairs) for amil-vs-amil_vote
    (the MECHANISM comparison) and amil-vs-vote (the PIPELINE comparison)
  * attention concentration, per bag then averaged:
      - effective_patches   = exp(H)          "how many patches are effectively in play"
      - effective_fraction  = exp(H) / N      in [0,1]; 1.0 = perfectly uniform attention
      - normalized_entropy  = H / log(N)      in [0,1]; the other common normalisation
    exp(H) is the perplexity of the attention distribution.  These are computed PER BAG
    and then averaged, which is NOT the same as exp(mean H) / mean N -- bag sizes vary a
    lot (min 2, max 336), so the naive version carries a real Jensen error.

Then it aggregates folds within each arm (patch size or model size) and reports
mean +/- sample std.

Usage:
    python compile_results.py --runs-root /scratch/$USER/runs --out-dir ./compiled
    python compile_results.py --runs-root ./runs --pattern 'abl_p*_nano_f*'
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# abl_p224_nano_f3  ->  arm="p224_nano", axis_value=224, model="nano", fold=3
RUN_RE = re.compile(r"^abl_(?:p(?P<patch>\d+)|msize)_(?P<model>[a-z]+)(?:_(?P<msize_patch>\d+))?_f(?P<fold>\d+)$")

PREDICTORS = ["amil", "amil_vote", "vote"]


def mcnemar(a_correct, b_correct) -> dict:
    """Exact two-sided McNemar. Mirrors test_model.py so numbers reconcile exactly."""
    b = int(sum(1 for x, y in zip(a_correct, b_correct) if x and not y))
    c = int(sum(1 for x, y in zip(a_correct, b_correct) if y and not x))
    n = b + c
    out = {"a_only_correct": b, "b_only_correct": c, "discordant": n}
    if n == 0:
        out["p_value"] = 1.0
        return out
    try:
        from scipy.stats import binomtest
        out["p_value"] = float(binomtest(b, n, 0.5).pvalue)
    except Exception:
        chi2 = (abs(b - c) - 1) ** 2 / n
        out["p_value"] = float(math.exp(-chi2 / 2))
        out["p_value_note"] = "normal approximation (scipy unavailable)"
    return out


def parse_run_name(name: str):
    """Return (arm, axis_name, axis_value, model, fold) or None if it isn't a run dir."""
    m = RUN_RE.match(name)
    if not m:
        return None
    fold = int(m.group("fold"))
    model = m.group("model")
    if m.group("patch"):                      # patch-size ablation: abl_p224_nano_f0
        return (f"patch_{m.group('patch')}", "patch_size", int(m.group("patch")), model, fold)
    # model-size ablation: abl_msize_tiny_224_f0
    patch = int(m.group("msize_patch") or 224)
    return (f"model_{model}", "model_size", model, model, fold, patch)


def attention_stats(df: pd.DataFrame) -> dict:
    """Per-bag attention concentration, averaged over bags.

    Bags with N < 2 are excluded from normalized_entropy (log(1) = 0 -> undefined) but
    kept for effective_patches / effective_fraction, where the values are well defined.
    """
    out = {}
    if "amil_attn_entropy" not in df.columns or "n_patches" not in df.columns:
        return out

    H = df["amil_attn_entropy"].to_numpy(dtype=float)
    N = df["n_patches"].to_numpy(dtype=float)

    eff_patches = np.exp(H)                       # perplexity of the attention weights
    eff_fraction = np.clip(eff_patches / np.maximum(N, 1.0), 0.0, 1.0)

    out["mean_attn_entropy"] = float(H.mean())
    out["mean_bag_size"] = float(N.mean())
    out["effective_patches_mean"] = float(eff_patches.mean())
    out["effective_patches_std"] = float(eff_patches.std(ddof=1)) if len(eff_patches) > 1 else 0.0
    out["effective_fraction_mean"] = float(eff_fraction.mean())
    out["effective_fraction_std"] = float(eff_fraction.std(ddof=1)) if len(eff_fraction) > 1 else 0.0

    ok = N >= 2
    if ok.any():
        norm_H = H[ok] / np.log(N[ok])
        norm_H = np.clip(norm_H, 0.0, 1.0)
        out["normalized_entropy_mean"] = float(norm_H.mean())
        out["normalized_entropy_std"] = float(norm_H.std(ddof=1)) if norm_H.size > 1 else 0.0
        out["n_bags_for_normalized"] = int(ok.sum())
    return out


def fold_metrics(xlsx: Path, sheet: str = "Pred_full") -> dict | None:
    """Recompute every reported number for one fold from its per-bag rows."""
    try:
        df = pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as exc:
        print(f"    ! could not read {sheet} from {xlsx.name}: {exc}")
        return None
    if "true_class" not in df.columns:
        print(f"    ! {xlsx.name}:{sheet} has no true_class column")
        return None

    res = {"n_images": int(len(df))}
    correct = {}
    for p in PREDICTORS:
        col = f"{p}_pred"
        if col not in df.columns:
            continue
        ok = (df[col] == df["true_class"])
        correct[p] = ok.tolist()
        res[f"{p}_accuracy"] = float(ok.mean())

    for a, b in (("amil", "amil_vote"), ("amil", "vote")):
        if a in correct and b in correct:
            mc = mcnemar(correct[a], correct[b])
            res[f"{a}_vs_{b}_delta_pp"] = 100.0 * (res[f"{a}_accuracy"] - res[f"{b}_accuracy"])
            res[f"{a}_vs_{b}_p"] = mc["p_value"]
            res[f"{a}_vs_{b}_discordant"] = mc["discordant"]
            res[f"{a}_vs_{b}_a_only"] = mc["a_only_correct"]
            res[f"{a}_vs_{b}_b_only"] = mc["b_only_correct"]

    res.update(attention_stats(df))
    return res


def stage1_val_acc(run_dir: Path) -> dict:
    """Patch-level (Stage-1) accuracy -- the instance-quality column for H1."""
    p = run_dir / "pretrain" / "pretrain_summary.json"
    if not p.exists():
        return {}
    try:
        js = json.loads(p.read_text())
        return {"stage1_val_acc": js.get("best_val_acc"),
                "stage1_best_epoch": js.get("best_epoch")}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="folder containing abl_* run dirs")
    ap.add_argument("--pattern", default="abl_*_f*", help="glob for run dirs")
    ap.add_argument("--sheet", default="Pred_full",
                    help="per-bag sheet: Pred_full (uncapped, primary) or Pred_capped_128")
    ap.add_argument("--out-dir", default="compiled")
    args = ap.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        xlsx = run_dir / "test" / "classification_results.xlsx"
        if not xlsx.exists():
            print(f"  - {run_dir.name}: no test results yet, skipping")
            continue

        print(f"  + {run_dir.name}")
        met = fold_metrics(xlsx, args.sheet)
        if met is None:
            continue

        arm, axis_name, axis_value, model, fold = parsed[0], parsed[1], parsed[2], parsed[3], parsed[4]
        row = {"run": run_dir.name, "arm": arm, "axis": axis_name,
               "axis_value": axis_value, "model": model, "fold": fold}
        row.update(stage1_val_acc(run_dir))
        row.update(met)
        rows.append(row)

    if not rows:
        print("No completed runs found. Check --runs-root / --pattern.")
        return

    per_fold = pd.DataFrame(rows).sort_values(["arm", "fold"])
    per_fold_path = out / "per_fold_results.csv"
    per_fold.to_csv(per_fold_path, index=False)

    # ---- aggregate across folds within each arm -------------------------------------
    numeric = [c for c in per_fold.columns
               if c not in ("run", "arm", "axis", "axis_value", "model", "fold")
               and pd.api.types.is_numeric_dtype(per_fold[c])]

    agg_rows = []
    for arm, grp in per_fold.groupby("arm", sort=False):
        rec = {"arm": arm, "axis": grp["axis"].iloc[0],
               "axis_value": grp["axis_value"].iloc[0],
               "model": grp["model"].iloc[0], "n_folds": int(len(grp))}
        for c in numeric:
            vals = grp[c].dropna()
            if vals.empty:
                continue
            rec[f"{c}_mean"] = float(vals.mean())
            # sample std; undefined for a single fold -> report 0.0 and rely on n_folds
            rec[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(rec)

    master = pd.DataFrame(agg_rows)
    # sort numerically for patch arms, alphabetically otherwise
    try:
        master = master.sort_values("axis_value", key=lambda s: pd.to_numeric(s, errors="coerce"))
    except Exception:
        master = master.sort_values("axis_value")

    master_path = out / "master_summary.csv"
    master.to_csv(master_path, index=False)

    def _jsonable(o):
        """numpy scalars leak in from pandas and are not JSON serializable."""
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    with open(out / "master_summary.json", "w") as f:
        json.dump({"per_fold": rows, "aggregated": agg_rows}, f, indent=2, default=_jsonable)

    # ---- headline table to stdout ----------------------------------------------------
    print(f"\nWrote {per_fold_path}  ({len(per_fold)} folds)")
    print(f"Wrote {master_path}      ({len(master)} arms)\n")

    cols = [("axis_value", "arm"), ("n_folds", "folds"),
            ("stage1_val_acc_mean", "patch acc"),
            ("amil_accuracy_mean", "AMIL"), ("amil_vote_accuracy_mean", "amil_vote"),
            ("vote_accuracy_mean", "vote"),
            ("amil_vs_amil_vote_delta_pp_mean", "AMIL-vote pp"),
            ("effective_fraction_mean_mean", "eff.frac"),
            ("mean_bag_size_mean", "bag N")]
    hdr = " | ".join(f"{lbl:>12}" for _, lbl in cols)
    print(hdr)
    print("-" * len(hdr))
    for _, r in master.iterrows():
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append(f"{'-':>12}")
            elif isinstance(v, float):
                cells.append(f"{v:>12.4f}")
            else:
                cells.append(f"{str(v):>12}")
        print(" | ".join(cells))

    print("\nPer-arm mean +/- std for every metric is in master_summary.csv")


if __name__ == "__main__":
    main()
