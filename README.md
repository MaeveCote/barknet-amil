# BarkNet-AMIL

Code for *"Toward an Optimal Configuration for Tree Species Classification from Bark
Images: A Leakage-Safe Ablation Study"* (paper: arXiv link TBD).

Attention-based Multiple Instance Learning (AMIL) over patches cut from bark images,
with a ConvNeXt-V2 backbone, evaluated under a **tree-level, leakage-safe 5-fold
cross-validation** on [BarkNet 1.0](https://github.com/ulaval-damas/tree-bark-classification).
Follows up on Carpentier et al. (2018) and Cui et al. (2023), both of which split at the
image level; a random image-level split lets crops of the *same tree* land in both train
and test, which inflates accuracy. This project splits at the **tree** level instead (see
[Leakage-safe splitting](#leakage-safe-splitting-and-fold-generation) below) and reports
five ablations against that corrected baseline: patch size, aggregation (majority vote vs.
AMIL), training regime (1- vs. 2-stage), backbone capacity, and test-time bag size, plus a
whole-image baseline.

## Repo map

```
configs/     YAML configs (local, cluster, ablation, batch-sweep)
src/         portable pipeline: data prep, training, evaluation, shared helpers
scripts/     local Windows entry points (.bat) — the generic reproduction path
cluster/     SLURM/DRAC job scripts + Windows-local ablation drivers used for the paper
tests/       offline correctness checks (split leakage, determinism, model plumbing)
notebooks/   figure-generating notebook + exported PDFs (patch grid, confusion matrix,
             training curves, bag-size curve)
docs/        hypothesis tracker (H1-H7) used while running the ablations
```

`src/` has no dependency on `cluster/` or `scripts/` — every entry point is a plain
`python src/<script>.py -c <config>.yaml [overrides]` call. `cluster/` and `scripts/` are
two different ways of *invoking* that same core: `scripts/*.bat` for a single local run,
`cluster/*.sh`/`*.ps1` for the SLURM sweeps and ablations the paper actually used.

### `src/`

| File | Role |
|---|---|
| `data_preparation/cut_patches.py` | Raw images → tree-stratified train/test split → cut patches |
| `data_preparation/compute_dataset_stats.py` | Per-channel mean/std for `augmentation.normalize` |
| `pretrain_backbone.py` | **Stage 1**: per-patch backbone pretraining |
| `train_model.py` | **Stage 2**: image-level AMIL fine-tuning (loads the Stage-1 backbone) |
| `test_model.py` | Evaluation: `amil` / `amil_vote` / `vote` predictors, McNemar tests, Excel report |
| `hyperparameter_tuning.py` | Optuna search over Stage-1/Stage-2 hyperparameters (see [Known issues](#known-issues)) |
| `batch_training.py` | Chains tune→pretrain→tune→train→test across a (model size, patch size) sweep |
| `merge_split.py` | Recombines a pre-split patch dataset into one folder (the runtime loader does its own tree-level split from a single root) |
| `compute_dpi.py` | Local, no GPU: estimates effective DPI of images/patches from filenames |
| `vram_probe.py` | Local: measures/projects peak VRAM for a given bag size and model size |
| `helper/data_loader.py` | Tree-level leakage-safe split, k-fold, bag/patch `Dataset`s and loaders |
| `helper/model_wrapper.py` | `ConvNeXtAMIL` — builds/trains/validates both stages |
| `helper/model.py` | `PatchAttentionMIL`, `PatchClassifier` |
| `helper/config_cli.py` | YAML loading, `$VAR` expansion, CLI overrides |
| `helper/optimizer_scheduler.py`, `helper/early_stopping.py`, `helper/timing.py` | Small utilities |

## Setup

```bash
pip install -r requirements.txt
```

Developed and run with **Python 3.11+**, **torch 2.1+** (tested up to 2.12), **timm
1.0.3+** (tested up to 1.0.27). A CUDA GPU is required for anything beyond
`compute_dpi.py`/`merge_split.py`; Stage-2 fine-tuning needs enough VRAM to hold one
whole bag of patches (`data.max_patches_per_bag`) — use `src/vram_probe.py` to check
before committing to a model size on your card.

## Data

1. Download BarkNet 1.0 from
   [ulaval-damas/tree-bark-classification](https://github.com/ulaval-damas/tree-bark-classification).
2. Place the per-species folders under `data/barknet/dataset/<SPECIES>/*.jpg`
   (`data/` is gitignored — nothing here is ever committed).
3. Cut patches (tree-stratified train/test split happens here, at cut time):

   ```bash
   python src/data_preparation/cut_patches.py data/barknet/dataset data/barknet/patches_224 --patch-size 224 --test-ratio 0.15
   ```

   or `scripts/1_prepare_data.bat` (patch size 384 by default — edit the `.bat` for
   other sizes). `--test-ratio 0` cuts everything into `train/` only, which is what the
   runtime k-fold split below expects.

## Reproducing the paper's results

### Local path (generic reproduction, one config/one run at a time)

```
scripts/1_prepare_data.bat            # cut patches (edit patch size inside)
scripts/2-2_pretrain_backbone.bat     # Stage 1: backbone pretraining
scripts/3-2_train_model.bat           # Stage 2: AMIL fine-tuning
scripts/4_testing_model.bat           # evaluation (amil / amil_vote / vote, full + capped bags)
```

Each `.bat` just `cd`s to the repo root and calls the matching `src/*.py -c
configs/config.yaml`. Edit `configs/config.yaml` for patch root, model size, fold, etc.,
or override on the command line, e.g.:

```bash
python src/train_model.py -c configs/config.yaml --fold 2 --set data.split.n_folds=5
```

`scripts/2-1_*.bat` / `3-1_*.bat` (Optuna search) and `scripts/batch_training.bat` (sweep
orchestration) are wired up the same way but currently hit the `hyperparameter_tuning.py`
gap noted below.

### Cluster path (what actually produced the paper's numbers)

Run on the Digital Research Alliance of Canada (DRAC). Every `cluster/*.sh` needs its
`#SBATCH --account=YOUR_ACCOUNT_gpu`/`_cpu` placeholder replaced with your own
allocation (or override at submit time: `sbatch --account=<you> cluster/job_x.sh`), and
`cluster/00_prefetch_weights.sh` run once on a login node first (compute nodes have no
internet). See `cluster/common.sh` for the full environment contract.

| Ablation | Script(s) | Config |
|---|---|---|
| Patch size (224/288/384, 3 sizes) | `submit_ablation.sh` → `job_abl_stage{1,2}.sh` | `config_ablation.yaml` |
| Patch size (full 6-size × 5-fold sweep: 96/160/224/288/384/512) | `submit_cv_ablation.sh` → `job_cv_stage{1,2}_{small,large}.sh` | `config_ablation.yaml` |
| Aggregation (AMIL vs. majority vote) | *(no separate script)* — every `test_model.py` run reports `amil`, `amil_vote`, `vote` together with paired McNemar tests | any |
| Training regime (1- vs. 2-stage) | `job_onestage.sh` (1-stage) vs. `job_abl_stage{1,2}.sh` @ patch 224 (2-stage), compiled together by `run_compile_1stage.sh` | `config_ablation.yaml` |
| Backbone capacity (pico/nano/tiny @ 224) | `prefetch_msize_weights.sh` once, then `submit_msize_pico.sh` / `submit_msize_tiny.sh` (nano reuses the patch-size sweep's 224 run); compiled by `run_compile_msize.sh` | `config_ablation.yaml` |
| Whole-image baseline | `run_wholeimage.ps1` (train) → `run_wholeimage_test.ps1` (test) → `run_eval_wholeimage.ps1` (per-image eval via `eval_wholeimage.py`) | `config.yaml` |
| Test-time bag size | `run_bagsize_inference.ps1` — inference-only re-scoring of already-trained models at bag sizes 4..256 | *(reuses trained checkpoints)* |
| Final/best model | `job_final_model.sh` (single-stage nano @ 512, 5-fold CV), compiled by `compile_final_results.sh` | `config_ablation.yaml` |

The three `.ps1` "whole-image"/"bag-size" drivers and `eval_wholeimage.py` run **locally**
(Windows, no SLURM) against checkpoints already produced on the cluster — they're
inference/analysis passes, not training jobs.

### Figures

`notebooks/visualizations.ipynb` regenerates the patch-grid, augmentation-preview,
training-curve, and confusion-matrix figures from a run's output CSVs and
`classification_results.xlsx`; `notebooks/*.pdf` are its exported outputs.

## Leakage-safe splitting and fold generation

The train/val/test split is **tree-level** (a tree's patches/images never appear in more
than one split) and is computed **at run time** by `helper.data_loader.split_trees()`
from a single patch root — there is no separate fold-assignment file to ship or go stale.
It's deterministic from three config values:

- `project.seed` — RNG seed
- `data.split.n_folds` — set to `5` for the paper's CV; `null` for a plain holdout
- `data.split.fold_index` — which fold is held out as test (`--fold` on the CLI)

Each class is split independently (`random.Random(f"{seed}:{label}")`), so adding or
dropping a species doesn't perturb the other classes' assignments. `split_trees()` calls
`_assert_disjoint()` on every call, which raises immediately if any tree ever ends up in
two splits — this is a hard runtime guarantee, not just a one-time check. To reproduce
fold *N* of the 5-fold CV for any config: `--set data.split.n_folds=5 --fold N`.

## Known issues

- **`src/hyperparameter_tuning.py`** is invoked with `--stage pretrain|board` by
  `batch_training.py` and `scripts/2-1_*.bat`/`3-1_*.bat`, but defines no such flag and
  doesn't branch its search space by stage — it will fail argparse if called that way.
  The tuned hyperparameter values actually used for the paper are already inlined in
  `configs/config.yaml`/`config_cluster.yaml`/`config_ablation.yaml`, so this doesn't
  block reproducing results, only re-running the search from scratch.
- **`tests/test_pipeline.py`** expects a synthetic fixture at `/tmp/fake/train` (5
  species × 6 trees × 3 images, with one deliberately oversized 40-patch bag) that no
  script in this repo generates — it isn't runnable as committed. The checks themselves
  (tree-level disjointness, k-fold partitioning, stochastic bag-cap behaviour, Stage-1→
  Stage-2 checkpoint transfer, chunked-inference equivalence) are still useful reading for
  understanding the split/loader invariants even without running them.
- Several `cluster/*.ps1` / `run_compile*.sh` scripts assume checkpoints and run
  directories from a specific prior run (e.g. `abl_p224_nano_f*`) already exist on
  `$SCRATCH` — they're compilation/analysis passes over completed sweeps, not
  standalone entry points.

## Citation

```
(to be completed — arXiv / venue TBD)
```
