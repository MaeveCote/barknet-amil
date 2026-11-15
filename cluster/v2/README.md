# Leakage experiment and aggregation baselines (`cluster/v2/`)

Experiments added to the paper on top of the main ablations:

- **Image-level split (A).** The paper's pipeline re-run under a deliberately *leaky*
  image-level split (images of one tree on both sides), to measure how much a split that
  ignores trees inflates accuracy. Arms: the whole-image baseline (A1) and the
  recommended 512 px two-stage attention-MIL model (A2).
- **Aggregation baselines (B).** Inference only, on the trained tree-level checkpoints:
  soft vote (over the Stage-1 classifier and over the MIL patch head) and uniform
  attention (α = 1/N), next to attention MIL, attention-MIL vote and majority vote.

The pipeline is the paper's, unchanged: same 20 species, hyperparameters, seed (42),
augmentation, epochs, training bag cap (128) and evaluation code. Nothing is re-tuned and
no checkpoint is chosen on a test fold. The tree-level runs are only ever read. Every
new output goes under `$SCRATCH/results/v2/` (`V2_ROOT`, see `v2_env.sh`).

## Prerequisites

- The tree-level runs of the main ablations under `$SCRATCH/runs/` (`TREE_RUNS`), named
  as in `cluster/job_train.sh` (`abl_p<N>_nano_f<k>`, `abl_1stage_nano_224_f<k>`,
  `abl_msize_<size>_224_f<k>`). Part B re-scores these.
- Patch tars `$SCRATCH/data/barknet_patches_<N>.tar` (`cluster/job_cut_patches.sh`), and
  the raw photos both as a folder `$SCRATCH/data/barknet_raw/` (for the fold file) and as
  `$SCRATCH/data/barknet_raw.tar` (staged by the whole-image jobs).
- Your allocation: `export ACCOUNT_GPU=<you>_gpu ACCOUNT_CPU=<you>_cpu`.
- `/scratch` is purged after about 60 days without access. The preflight reports how
  close the checkpoints are to that; back them up to `/project` if needed.

## 1. Preflight (login node, no GPU, a few minutes)

```bash
bash cluster/v2/00_preflight.sh
```

Read-only on existing runs. It writes `$V2_ROOT/preflight/preflight_<stamp>.txt` with:
- every tree-level checkpoint Part B needs, whether it exists, and how many days since it
  was last touched (and, if the cluster publishes one, which of them are on the purge
  list);
- saved per-patch logit files (none are expected, so soft vote and uniform attention are
  recomputed on GPU from the checkpoints);
- the patch tars and raw photos;
- the **image-level fold file** `$V2_ROOT/folds/image_folds.csv`, built once by
  `src/data_preparation/make_image_folds.py` (`StratifiedKFold(5, shuffle=True,
  seed=42)` over the 23,271 images). It holds the same rows as the shipped
  `folds/image_level_folds.csv`, which you can also pass directly as `FOLD_FILE`. It also writes
  `image_folds_leakage.csv`: per fold, the share of test images whose tree is also in
  training (about 100 %);
- a verdict, `MODE=full` or `MODE=minimal` (checkpoints missing).

## 2. Submit (login node)

```bash
bash cluster/v2/submit_v2.sh           # dry run: prints every sbatch call
GO=1 bash cluster/v2/submit_v2.sh      # submits; job ids -> $V2_ROOT/submitted_jobs_<stamp>.txt
```

| Part | Jobs | What runs | Output |
|---|---|---|---|
| A1 | 5 | whole image @ 224, nano, image-level folds (same recipe as the paper's whole-image baseline) | `runs/v2img_whole_nano_f<k>/` |
| A2 | 5 → 5 | nano 512 px two-stage on image-level folds; Stage 2 waits for its own fold's Stage 1 (`aftercorr`) | `runs/v2img_p512_nano_f<k>/` |
| B3 | — | part of A2's test step (every aggregator is scored there) | same |
| B1, B2, B4 | 1 array (40 tasks, at most 10 running at once) | re-scores tree-level checkpoints: nano 160–512 two-stage, nano 224 single-stage, pico/tiny 224 | `eval/tree_*_f<k>/` |
| compile | 1 CPU | builds the tables once everything above has ended | `tables/` |

`MODE=minimal` (set automatically when B1/B2 checkpoints are missing; override with
`MODE=...`) replaces the B array with retraining tree-level nano 512 two-stage and nano 224
single-stage (paper settings, 5 folds each) into `runs/retrain_*`; their test steps produce
Part B. `INCLUDE_B4=0` skips the pico/tiny re-scoring.

Each job refuses to overwrite a finished result, and `submit_v2.sh` refuses to run twice
(`$V2_ROOT/submitted.lock`; `FORCE=1` to override). To re-run a failed task, resubmit
that array index with the same manifest (finished tasks are skipped), e.g.:

```bash
sbatch --account=$ACCOUNT_GPU --array=0-4 \
  --export=ALL,V2_DIR=$PWD/cluster/v2,MANIFEST=$(ls -t $SCRATCH/results/v2/eval/manifest_*.tsv | head -1) \
  cluster/v2/job_eval.sh
```

Resource notes from the runs used for the paper (H100): A1 takes about 1.5 h per fold,
A2 Stage 1 about 5 h. Evaluations take 6–35 min; 160 px evaluations need 96 GB of host
RAM (the job default).

## 3. Collect

When the compile job has run, everything is in `$V2_ROOT/tables/`:

- `tables.md`: tree- vs image-level split, aggregators × patch size (accuracy and
  macro-F1), single-stage 224, backbones, per-fold Δ(attention MIL − uniform / soft vote)
  with their signs, and checks: re-scored checkpoints must reproduce the original
  accuracies exactly, recomputed tree-level values vs the quoted references, and
  per-fold tree leakage of the image folds.
- CSVs behind every table, `per_fold_metrics.csv`, `summary_metrics.csv`,
  `fold_deltas.csv`, and `predictions/<arm>.csv` (`image_id, tree_id, true, pred_*, fold`).

Re-run the compile by hand at any time for partial results:
`bash cluster/v2/job_compile.sh`.

## Provenance

Every job appends the git commit (and any uncommitted files), config path and checksum,
seed and all knobs to `<run>/provenance.txt`. Each stage also writes its fully-resolved
config (`resolved_config.yaml`).

## Definitions

- **Image id** `<SPECIES>:<tree>:<uid>`: `uid` is the image's index among its tree's raw
  files sorted by name, the same `imageUID` that `cut_patches.py` writes into patch
  names. Whole-image and patch runs therefore share one fold assignment, and every patch
  inherits its image's fold. Jobs check ids against the patch tar's `patch_index.csv`.
- **Aggregators** (all from one forward pass of the same weights): attention MIL
  (`amil`), uniform attention (α = 1/N: mean patch embedding → same head,
  `amil_uniform`; with a linear head this equals averaging the patch logits),
  attention-MIL vote (hard vote over the MIL patch head, `amil_vote`), soft vote over the
  MIL patch head (`amil_soft`), majority and soft vote over the Stage-1 classifier
  (`vote`, `vote_soft`; two-stage runs only).
- **Per-patch accuracy**: per-patch *test* accuracy, pooled over all test patches
  (`vote_patch_test_accuracy`). The Stage-1 *validation* accuracy at the best epoch is also
  kept (`stage1_val_patch_accuracy`); it is selected on itself and so slightly optimistic.
- **Effective attention fraction**: exp(H(α))/N per bag, averaged over bags.
- **A1 settings**: the tree-level whole-image baseline was trained with rotation 15° and
  batch 64, so A1 uses exactly those. The patch models use 30°.
