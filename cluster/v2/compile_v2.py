#!/usr/bin/env python3
"""Compile every v2 result into the tables requested for main_v2.tex. CPU only.

Reads only per-image prediction files (nothing is re-inferred):
  tree-level Part B   $V2_ROOT/eval/tree_<arm>_f<k>/test/test_predictions_full.csv
                      (or, after a purge, $V2_ROOT/runs/retrain_abl_<arm>_f<k>/test/...)
  image-level A2/B3   $V2_ROOT/runs/v2img_p512_nano_f<k>/test/test_predictions_full.csv
  image-level A1      $V2_ROOT/runs/v2img_whole_nano_f<k>/wholeimage_preds.csv
plus Stage-1 summaries (validation per-patch accuracy) and, for re-scored checkpoints, the
ORIGINAL test_summary.json -- the original attention-MIL / vote accuracies must be
reproduced exactly, which proves the new evaluators did not change the old ones.

Writes to --out-dir (default $V2_ROOT/tables):
  tables.md                  Tables 1-3 + per-fold deltas + checks, ready to transcribe
  table1_tree_vs_image.csv, table2_aggregators_by_patch.csv (+ _macro_f1), table3_single_stage.csv
  per_fold_metrics.csv       every metric, one row per (arm, fold)
  summary_metrics.csv        mean / std over folds per arm
  fold_deltas.csv            Delta(attention MIL - uniform / soft vote) per fold, with sign
  predictions/<arm>.csv      image_id, tree_id, true, pred_<aggregator>..., fold
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # cluster/ for compile_results
from compile_results import attention_stats, macro_f1  # noqa: E402

FOLDS = range(5)
PATCH_SIZES = [160, 224, 288, 384, 512]

# (column prefix, row label) in display order
AGGREGATORS = [
    ("vote", "Majority vote (Stage-1 classifier)"),
    ("vote_soft", "Soft vote (Stage-1 classifier)"),
    ("amil_vote", "Attention-MIL vote (MIL patch head)"),
    ("amil_soft", "Soft vote (MIL patch head)"),
    ("amil_uniform", "Uniform attention"),
    ("amil", "Attention MIL"),
]
DELTAS = [("amil", "amil_uniform"), ("amil", "amil_soft"), ("amil", "vote_soft")]

# Tree-level whole-image baseline: trained and evaluated LOCALLY for the paper
# (runs_wholeimage/wholeimage_allfolds_summary.csv), so not on the cluster. Fold values
# copied verbatim; mean +/- std = 91.9 +/- 2.7, the reference quoted for v2.
TREE_WHOLE_IMAGE = {"accuracy": [0.936028, 0.929113, 0.871532, 0.938121, 0.919805],
                    "macro_f1": [0.932412, 0.928020, 0.875027, 0.939394, 0.920586]}
# Tree-level references quoted with the v2 plan (paper numbers), for a consistency check.
REFERENCE_TREE = {"whole_accuracy": (91.9, 2.7), "stage1_val_patch_accuracy": (89.6, 3.3),
                  "vote_accuracy": (94.7, 2.1), "amil_accuracy": (94.9, 2.2),
                  "effective_fraction": (99.0, 1.1)}


# --------------------------------------------------------------------------- discovery
def arms(v2: Path, tree_runs: Path):
    """Every arm: name, split, kind, and per fold (pred file, stage-1 summary, original)."""
    out = []

    def mil_arm(name, split, eval_name, src_prefix, retrain_prefix=None, v2run_prefix=None):
        folds = {}
        for k in FOLDS:
            cand = []
            if eval_name:
                cand.append((v2 / "eval" / f"{eval_name}_f{k}" / "test" / "test_predictions_full.csv",
                             tree_runs / f"{src_prefix}_f{k}", "rescored"))
            if retrain_prefix:
                run = v2 / "runs" / f"{retrain_prefix}_f{k}"
                cand.append((run / "test" / "test_predictions_full.csv", run, "retrained"))
            if v2run_prefix:
                run = v2 / "runs" / f"{v2run_prefix}_f{k}"
                cand.append((run / "test" / "test_predictions_full.csv", run, "trained"))
            for pred, src, how in cand:
                if pred.exists():
                    folds[k] = {"pred": pred, "src": src, "how": how}
                    break
        out.append({"arm": name, "split": split, "kind": "mil", "folds": folds})

    for ps in PATCH_SIZES:
        mil_arm(f"tree_p{ps}_nano", "tree", f"tree_p{ps}_nano", f"abl_p{ps}_nano",
                retrain_prefix=f"retrain_abl_p{ps}_nano" if ps == 512 else None)
    mil_arm("tree_1stage_nano_224", "tree", "tree_1stage_nano_224", "abl_1stage_nano_224",
            retrain_prefix="retrain_abl_1stage_nano_224")
    for ms in ("pico", "tiny"):
        mil_arm(f"tree_msize_{ms}_224", "tree", f"tree_msize_{ms}_224", f"abl_msize_{ms}_224")
    mil_arm("image_p512_nano", "image", None, None, v2run_prefix="v2img_p512_nano")

    whole = {}
    for k in FOLDS:
        p = v2 / "runs" / f"v2img_whole_nano_f{k}" / "wholeimage_preds.csv"
        if p.exists():
            whole[k] = {"pred": p, "src": p.parent, "how": "trained"}
    out.append({"arm": "image_whole_nano", "split": "image", "kind": "whole", "folds": whole})
    return out


# --------------------------------------------------------------------------- metrics
def load_preds(arm, k, info, fold_ids):
    df = pd.read_csv(info["pred"])
    if arm["kind"] == "whole":
        df = df.rename(columns={"pred_class": "whole_pred"})
        if "image_id" not in df.columns:
            raise ValueError(f"{info['pred']} has no image_id column (old evaluator?)")
    else:
        if "image_key" in df.columns:
            df["image_id_full"] = df["image_key"]
        else:
            df["image_id_full"] = df["true_class"] + ":" + df["image_id"].astype(str)
        df = df.drop(columns=["image_id"]).rename(columns={"image_id_full": "image_id"})
        df["tree_id"] = df["image_id"].str.split(":").str[1]
    df["fold"] = k
    df["true"] = df["true_class"]
    if fold_ids is not None and arm["split"] == "image":
        bad = df.loc[df["image_id"].map(fold_ids) != k, "image_id"]
        if len(bad):
            raise AssertionError(f"{arm['arm']} fold {k}: {len(bad)} test images are not in "
                                 f"fold {k} of the fold file, e.g. {bad.iloc[0]}")
    return df


def tree_vote(df, col):
    key = df["true"] + ":" + df["tree_id"].astype(str)
    truth = df.groupby(key)["true"].first()
    pred = df.groupby(key)[col].agg(lambda s: sorted(s.value_counts().items(),
                                                     key=lambda kv: (-kv[1], str(kv[0])))[0][0])
    return float((pred == truth).mean()), int(len(truth))


def stage1_val(run_dir: Path):
    p = run_dir / "pretrain" / "pretrain_summary.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("best_val_acc")
    return None if v is None else float(v) / 100.0


def fold_metrics(arm, k, info, df):
    row = {"arm": arm["arm"], "split": arm["split"], "fold": k, "source": info["how"],
           "n_images": int(len(df))}
    present = []
    for name in [a for a, _ in AGGREGATORS] + ["whole"]:
        col = f"{name}_pred"
        if col in df.columns:
            present.append(name)
            ok = df[col].astype(str) == df["true"].astype(str)
            row[f"{name}_accuracy"] = float(ok.mean())
            row[f"{name}_macro_f1"] = float(macro_f1(df["true"].astype(str), df[col].astype(str)))
            acc_t, n_trees = tree_vote(df, col)
            row[f"{name}_tree_vote_accuracy"] = acc_t
            row["n_trees"] = n_trees
    for head in ("vote", "amil"):
        c = f"{head}_patch_correct"
        if c in df.columns:
            row[f"{head}_patch_test_accuracy"] = float(df[c].sum() / max(df["n_patches"].sum(), 1))
    if arm["kind"] == "mil":
        row["stage1_val_patch_accuracy"] = stage1_val(info["src"])
        att = attention_stats(df)
        row["effective_fraction"] = att.get("effective_fraction_mean")
        row["mean_bag_size"] = att.get("mean_bag_size")
    for a, b in DELTAS:
        if f"{a}_accuracy" in row and f"{b}_accuracy" in row:
            row[f"delta_{a}_minus_{b}_pp"] = 100 * (row[f"{a}_accuracy"] - row[f"{b}_accuracy"])
    # reproduction check: the re-scored original predictors must match the paper run
    if info["how"] == "rescored":
        orig = info["src"] / "test" / "test_summary.json"
        if orig.exists():
            ev = json.loads(orig.read_text())["evaluations"]["full"]
            diffs = [abs(row[f"{p}_accuracy"] - ev[f"{p}_accuracy"])
                     for p in ("amil", "amil_vote", "vote")
                     if f"{p}_accuracy" in row and f"{p}_accuracy" in ev]
            row["repro_max_abs_diff_pp"] = 100 * max(diffs) if diffs else None
    return row, present


# --------------------------------------------------------------------------- formatting
def ms(vals, pct=True):
    v = pd.Series([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))],
                  dtype=float)
    if v.empty:
        return None, None, 0
    s = v.std(ddof=1) if len(v) > 1 else float("nan")
    f = 100.0 if pct else 1.0
    return f * v.mean(), f * s, len(v)


def fmt(m, s, n, nfold=5):
    if m is None:
        return "n/a"
    txt = f"{m:.1f} ± {s:.1f}" if not math.isnan(s) else f"{m:.1f}"
    return txt + ("" if n == nfold else f" (n={n})")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--tree-runs", required=True)
    ap.add_argument("--fold-file", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    v2, tree_runs = Path(args.v2_root), Path(args.tree_runs)
    out = Path(args.out_dir) if args.out_dir else v2 / "tables"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    fold_file = Path(args.fold_file) if args.fold_file else v2 / "folds" / "image_folds.csv"
    fold_ids = (pd.read_csv(fold_file).set_index("image_id")["fold"].to_dict()
                if fold_file.exists() else None)

    rows, preds = [], {}
    for arm in arms(v2, tree_runs):
        frames = []
        for k, info in sorted(arm["folds"].items()):
            df = load_preds(arm, k, info, fold_ids)
            r, present = fold_metrics(arm, k, info, df)
            rows.append(r)
            keep = ["image_id", "tree_id", "true"] + [f"{p}_pred" for p in present] + ["fold"]
            frames.append(df[keep])
        if frames:
            allp = pd.concat(frames, ignore_index=True)
            allp.columns = [c[:-5] if c.endswith("_pred") else c for c in allp.columns]
            allp = allp.rename(columns={c: f"pred_{c}" for c in allp.columns
                                        if c not in ("image_id", "tree_id", "true", "fold")})
            allp.to_csv(out / "predictions" / f"{arm['arm']}.csv", index=False)
            preds[arm["arm"]] = len(frames)
        print(f"  {arm['arm']:<24} {len(arm['folds'])}/5 folds "
              f"{sorted(set(i['how'] for i in arm['folds'].values())) or ''}")

    if not rows:
        print("No results found yet.")
        return
    per_fold = pd.DataFrame(rows).sort_values(["arm", "fold"])
    per_fold.to_csv(out / "per_fold_metrics.csv", index=False)

    num = [c for c in per_fold.columns if c not in ("arm", "split", "fold", "source")
           and pd.api.types.is_numeric_dtype(per_fold[c])]
    summ = per_fold.groupby("arm")[num].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "summary_metrics.csv", index=False)

    def cell(arm, metric, pct=True):
        sub = per_fold[per_fold.arm == arm]
        if sub.empty or metric not in sub:
            return None, None, 0
        return ms(sub[metric].tolist(), pct)

    md = ["# BarkNet v2 results", "",
          f"Compiled from `{v2}` (tree-level checkpoints: `{tree_runs}`). Accuracy and "
          "macro-F1 in %, mean ± sample std over folds; `(n=k)` marks an arm with fewer "
          "than 5 folds finished.", ""]

    # ---- Table 1 ------------------------------------------------------------------
    t1 = []
    tw = ms(TREE_WHOLE_IMAGE["accuracy"])
    spec = [
        ("Whole image, accuracy", tw, cell("image_whole_nano", "whole_accuracy")),
        ("Whole image, macro-F1", ms(TREE_WHOLE_IMAGE["macro_f1"]), cell("image_whole_nano", "whole_macro_f1")),
        ("512 px per-patch, Stage-1 validation", cell("tree_p512_nano", "stage1_val_patch_accuracy"),
         cell("image_p512_nano", "stage1_val_patch_accuracy")),
        ("512 px per-patch, Stage-1 test", cell("tree_p512_nano", "vote_patch_test_accuracy"),
         cell("image_p512_nano", "vote_patch_test_accuracy")),
        ("512 px majority vote", cell("tree_p512_nano", "vote_accuracy"), cell("image_p512_nano", "vote_accuracy")),
        ("512 px attention MIL", cell("tree_p512_nano", "amil_accuracy"), cell("image_p512_nano", "amil_accuracy")),
        ("512 px attention MIL, macro-F1", cell("tree_p512_nano", "amil_macro_f1"), cell("image_p512_nano", "amil_macro_f1")),
        ("512 px effective attention fraction", cell("tree_p512_nano", "effective_fraction"),
         cell("image_p512_nano", "effective_fraction")),
    ]
    for label, tr, im in spec:
        d = (im[0] - tr[0]) if (tr[0] is not None and im[0] is not None) else None
        t1.append({"Quantity": label, "Tree-level": fmt(*tr), "Image-level": fmt(*im),
                   "Δ (pp)": "n/a" if d is None else f"{d:+.1f}"})
    t1 = pd.DataFrame(t1)
    t1.to_csv(out / "table1_tree_vs_image.csv", index=False)
    md += ["## Table 1 — tree-level vs image-level split", "", md_table(t1), "",
           "Tree-level whole image = the paper's local run (fold values embedded in "
           "compile_v2.py). Per-patch 'validation' is the paper's per-patch number "
           "(best Stage-1 val accuracy); 'test' is the new per-patch test accuracy.", ""]

    tl = []
    for name, lab in AGGREGATORS + [("whole", "Whole image")]:
        arm = "image_whole_nano" if name == "whole" else "image_p512_nano"
        m = cell(arm, f"{name}_tree_vote_accuracy")
        if m[0] is not None:
            tl.append({"Aggregator (image-level split)": lab, "Tree-level vote within fold": fmt(*m)})
    if tl:
        md += ["### Image-level split, aggregated to trees within each fold — NOT COMPARABLE",
               "", "A tree's images straddle the folds under the image-level split, so these "
               "'trees' are partial trees; do not compare with the paper's tree-level vote.", "",
               md_table(pd.DataFrame(tl)), ""]

    chk = []
    for key, (rm, rs) in REFERENCE_TREE.items():
        got = {"whole_accuracy": tw, "stage1_val_patch_accuracy": cell("tree_p512_nano", "stage1_val_patch_accuracy"),
               "vote_accuracy": cell("tree_p512_nano", "vote_accuracy"),
               "amil_accuracy": cell("tree_p512_nano", "amil_accuracy"),
               "effective_fraction": cell("tree_p512_nano", "effective_fraction")}[key]
        ok = got[0] is not None and abs(got[0] - rm) < 0.051 and abs(got[1] - rs) < 0.051
        chk.append({"Reference (tree, 512 px)": key, "Quoted": f"{rm} ± {rs}",
                    "Recomputed": fmt(*got), "Match": "yes" if ok else "CHECK"})
    md += ["### Check: recomputed tree-level values vs the quoted references", "",
           md_table(pd.DataFrame(chk)), ""]

    # ---- Table 2 ------------------------------------------------------------------
    for metric, tag, title in (("accuracy", "", "accuracy"), ("macro_f1", "_macro_f1", "macro-F1")):
        t2 = []
        for name, lab in AGGREGATORS:
            r = {"Aggregator": lab}
            for ps in PATCH_SIZES:
                r[f"{ps} px"] = fmt(*cell(f"tree_p{ps}_nano", f"{name}_{metric}"))
            t2.append(r)
        if metric == "accuracy":
            t2.append({"Aggregator": "Effective attention fraction",
                       **{f"{ps} px": fmt(*cell(f"tree_p{ps}_nano", "effective_fraction"))
                          for ps in PATCH_SIZES}})
            t2.append({"Aggregator": "Per-patch test accuracy (Stage 1)",
                       **{f"{ps} px": fmt(*cell(f"tree_p{ps}_nano", "vote_patch_test_accuracy"))
                          for ps in PATCH_SIZES}})
        t2 = pd.DataFrame(t2)
        t2.to_csv(out / f"table2_aggregators_by_patch{tag}.csv", index=False)
        md += [f"## Table 2{'b' if tag else ''} — aggregators by patch size, tree-level, {title}",
               "", md_table(t2), ""]

    # ---- Table 3 ------------------------------------------------------------------
    t3 = []
    for name, lab in AGGREGATORS:
        acc, f1 = cell("tree_1stage_nano_224", f"{name}_accuracy"), cell("tree_1stage_nano_224", f"{name}_macro_f1")
        t3.append({"Aggregator": lab, "Accuracy": fmt(*acc), "Macro-F1": fmt(*f1)})
    t3.append({"Aggregator": "Effective attention fraction",
               "Accuracy": fmt(*cell("tree_1stage_nano_224", "effective_fraction")), "Macro-F1": ""})
    t3 = pd.DataFrame(t3)
    t3.to_csv(out / "table3_single_stage.csv", index=False)
    md += ["## Table 3 — single-stage nano, 224 px, tree-level", "", md_table(t3), "",
           "Single-stage training has no Stage-1 classifier, so the two Stage-1 rows are n/a.", ""]

    # ---- B4 (optional) -----------------------------------------------------------
    if any(per_fold.arm.str.startswith("tree_msize_")):
        t4 = []
        for name, lab in AGGREGATORS:
            r = {"Aggregator": lab}
            for arm, col in (("tree_msize_pico_224", "pico"), ("tree_p224_nano", "nano"),
                             ("tree_msize_tiny_224", "tiny")):
                r[f"{col} @ 224"] = fmt(*cell(arm, f"{name}_accuracy"))
            t4.append(r)
        md += ["## (B4) aggregators by backbone, 224 px, tree-level, accuracy", "",
               md_table(pd.DataFrame(t4)), ""]

    # ---- per-fold deltas --------------------------------------------------------
    drows = []
    for _, r in per_fold.iterrows():
        for a, b in DELTAS:
            c = f"delta_{a}_minus_{b}_pp"
            if c in r and pd.notna(r[c]):
                drows.append({"arm": r["arm"], "fold": r["fold"], "comparison": f"{a} - {b}",
                              "delta_pp": round(r[c], 3), "sign": "+" if r[c] > 0 else ("-" if r[c] < 0 else "0")})
    if drows:
        dd = pd.DataFrame(drows)
        dd.to_csv(out / "fold_deltas.csv", index=False)
        piv = []
        for (arm, comp), g in dd.groupby(["arm", "comparison"], sort=False):
            g = g.sort_values("fold")
            piv.append({"Arm": arm, "Δ": comp,
                        "Per fold (pp)": ", ".join(f"{v:+.2f}" for v in g["delta_pp"]),
                        "Signs": "".join(g["sign"]),
                        "Mean ± std": fmt(*ms((g["delta_pp"] / 100).tolist()), nfold=len(g))})
        md += ["## Per-fold Δ (attention MIL − uniform attention / soft vote)", "",
               md_table(pd.DataFrame(piv)), ""]

    # ---- checks ---------------------------------------------------------------------
    if "repro_max_abs_diff_pp" in per_fold:
        rr = per_fold.dropna(subset=["repro_max_abs_diff_pp"])
        worst = rr["repro_max_abs_diff_pp"].max() if len(rr) else float("nan")
        md += ["## Check: re-scored checkpoints reproduce the paper's numbers", "",
               f"{len(rr)} re-scored folds; largest |Δ| between the new and the original "
               f"amil / amil_vote / vote accuracy: {worst:.4f} pp "
               f"({'identical' if worst == 0 else 'NOT identical -- investigate'}).", ""]
    leak = fold_file.parent / "image_folds_leakage.csv"
    if leak.exists():
        lk = pd.read_csv(leak)
        lk["test images whose tree is in train (%)"] = (100 * lk["test_images_tree_in_train"]).round(2)
        md += ["## Check: image-level folds leak trees (as intended)", "",
               md_table(lk[["fold", "n_train", "n_val", "n_test",
                            "test images whose tree is in train (%)"]]), ""]
    status = per_fold.groupby("arm")["fold"].apply(lambda s: ",".join(map(str, sorted(s))))
    md += ["## Folds found per arm", "", md_table(status.reset_index().rename(columns={"fold": "folds"})), ""]

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out / 'tables.md'} and CSVs in {out}")


if __name__ == "__main__":
    main()
