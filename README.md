# BarkNet-AMIL

Code for *"Toward an Optimal Configuration for Tree Species Classification from Bark
Images: A Leakage-Safe Ablation Study"* (paper: arXiv link TBD).

Attention-based Multiple Instance Learning (AMIL) over patches cut from bark images,
with a ConvNeXt-V2 backbone, evaluated under a **tree-level, leakage-safe 5-fold
cross-validation** on [BarkNet 1.0](https://github.com/ulaval-damas/tree-bark-classification)
(23 Canadian tree species; this project trains on the 20-species subset used by prior
work, see `configs/config.yaml`).
Follows up on Carpentier et al. (2018) and Cui et al. (2023), both of which split at the
image level; a random image-level split lets crops of the *same tree* land in both train
and test, which inflates accuracy. This project splits at the **tree** level instead (see
[Leakage-safe splitting](#leakage-safe-splitting-and-fold-generation) below) and reports
five ablations against that corrected baseline: patch size, aggregation (majority vote vs.
AMIL), training regime (1- vs. 2-stage), backbone capacity, and test-time bag size, plus a
whole-image baseline.

## Repo map

```
configs/     the ONE config (config.yaml) + the batch-sweep definition
src/         portable pipeline: data prep, training, evaluation, shared helpers
scripts/     local Windows entry points (.bat) — the generic reproduction path
cluster/     SLURM/DRAC job scripts + Windows-local ablation drivers used for the paper
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
3. Cut patches into a single root (the runtime k-fold split below carves train/val/test
   out of this one folder — no separate pre-cut test set needed):

   ```bash
   python src/data_preparation/cut_patches.py data/barknet/dataset data/barknet/patches_224 --patch-size 224 --test-ratio 0.0
   ```

   or `scripts/1_prepare_data.bat` (same command, patch size 224 to match
   `configs/config.yaml`'s default — edit both together if you change patch size).

## Reproducing the paper's results

### Local path (generic reproduction, one config/one run at a time)

```
scripts/1_prepare_data.bat            # cut patches (edit patch size inside)
scripts/2-2_pretrain_backbone.bat     # Stage 1: backbone pretraining
scripts/3-2_train_model.bat           # Stage 2: AMIL fine-tuning
scripts/4_testing_model.bat           # evaluation (amil / amil_vote / vote, full + capped bags)
```

Each `.bat` just `cd`s to the repo root and calls the matching `src/*.py -c
configs/config.yaml` — the one config file everything reads. Edit it directly for patch
root, model size, fold, etc., or override on the command line, e.g.:

```bash
python src/train_model.py -c configs/config.yaml --fold 2
```

(5-fold CV is already `config.yaml`'s default — see
[Leakage-safe splitting](#leakage-safe-splitting-and-fold-generation).)

`scripts/2-1_*.bat` / `3-1_*.bat` (Optuna search) and `scripts/batch_training.bat` (sweep
orchestration) are wired up the same way but currently hit the `hyperparameter_tuning.py`
gap noted below.

### Cluster path (what actually produced the paper's numbers)

Run on the Digital Research Alliance of Canada (DRAC). Every `cluster/*.sh` needs its
`#SBATCH --account=YOUR_ACCOUNT_gpu`/`_cpu` placeholder replaced with your own
allocation (or override at submit time: `sbatch --account=<you> cluster/job_x.sh`), and
`cluster/00_prefetch_weights.sh` run once on a login node first (compute nodes have no
internet). See `cluster/common.sh` for the full environment contract.

`cluster/job_train.sh` is the single parametrized SLURM array job behind every training
ablation below — set `PATCH_SIZE`/`MODEL_SIZE`/`STAGES`/etc. via `sbatch --export=...`
instead of maintaining a separate script per experiment. Its header comments give the
exact invocation for each row. `cluster/compile_results.py` (wrapped by
`cluster/run_compile.sh`) recomputes every reported metric — accuracy, macro-F1,
McNemar, attention concentration, tree-level vote — from each run's per-bag prediction
rows; it parses any of the run-name conventions below out of the box.

| Ablation | How | Compile |
|---|---|---|
| Patch size (96/160/224/288/384/512) | `job_train.sh` once per size: `sbatch --export=ALL,PATCH_SIZE=<N> cluster/job_train.sh` | `run_compile.sh` (default pattern) |
| Aggregation (AMIL vs. majority vote) | *(no separate run)* — every `test_model.py` pass reports `amil`, `amil_vote`, `vote` together with paired McNemar tests | already in each run's `classification_results.xlsx` |
| Training regime (1- vs. 2-stage) | `job_train.sh` with `STAGES=2` + the single-stage `--set` overrides in its header (no Stage-1 pretraining, uniform LR) vs. the normal 2-stage run at the same patch size | `run_compile.sh` |
| Backbone capacity (pico/nano/tiny @ 224) | `job_train.sh` with `MODEL_SIZE=pico\|nano\|tiny`, `PATCH_SIZE=224` | `run_compile.sh` |
| Whole-image baseline | `run_wholeimage.ps1` (train) → `run_wholeimage_test.ps1` (test) → `run_eval_wholeimage.ps1` (per-image eval via `eval_wholeimage.py`) | printed by `run_eval_wholeimage.ps1` itself |
| Test-time bag size | `run_bagsize_inference.ps1` — inference-only re-scoring of already-trained models at bag sizes 4..256 | printed by the script itself |
| Final/best model | `job_train.sh`, single-stage nano @ 512 | `run_compile.sh` |

The `.ps1` whole-image/bag-size drivers and `eval_wholeimage.py` run **locally**
(Windows, no SLURM) against checkpoints already produced on the cluster — they're
inference/analysis passes, not training jobs.

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
two splits — this is a hard runtime guarantee, not just a one-time check.
`configs/config.yaml` already defaults to `n_folds: 5`; reproduce fold *N* with
`--fold N` (or the `job_train.sh` array, where `SLURM_ARRAY_TASK_ID` becomes the fold).

## Known issues

- **`src/hyperparameter_tuning.py`** is invoked with `--stage pretrain|board` by
  `batch_training.py` and `scripts/2-1_*.bat`/`3-1_*.bat`, but defines no such flag and
  doesn't branch its search space by stage — it will fail argparse if called that way.
  The tuned hyperparameter values actually used for the paper are already inlined in
  `configs/config.yaml`, so this doesn't block reproducing results, only re-running the
  search from scratch.
- `cluster/run_wholeimage_test.ps1`, `run_eval_wholeimage.ps1`, `run_bagsize_inference.ps1`,
  and `run_compile.sh` all assume checkpoints/run directories from a prior training pass
  already exist (locally or on `$SCRATCH`) — they're inference/compilation passes over
  completed runs, not standalone entry points.

## License

[MIT](LICENSE).

## Citation

```
(to be completed — arXiv / venue TBD)
```
