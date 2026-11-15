# BarkNet-AMIL

Code for *"Toward an Optimal Configuration for Tree Species Classification from Bark
Images: A Leakage-Safe Ablation Study"* (paper: [arXiv link](https://arxiv.org/abs/2611.05680)).

Attention-based Multiple Instance Learning (AMIL) over patches cut from bark images,
with a ConvNeXt-V2 backbone, evaluated under a **tree-level, leakage-safe 5-fold
cross-validation** on [BarkNet 1.0](https://github.com/ulaval-damas/tree-bark-classification)
(23 Canadian tree species; this project trains on the 20-species subset used by prior
work, see `configs/config.yaml`).

## Repo map

```
configs/     the one config (config.yaml) + the batch-sweep definition
src/         portable pipeline: data prep, training, evaluation, shared helpers
scripts/     local Windows entry points (.bat) — the generic reproduction path
cluster/     SLURM/DRAC job scripts + Windows-local drivers used for the paper
cluster/v2/  image-level (leakage) experiment and aggregation baselines (own runbook)
folds/       per-image fold assignments used in the paper (tree-level and image-level)
```

`src/` has no dependency on `cluster/` or `scripts/` — every entry point is a plain
`python src/<script>.py -c <config>.yaml [overrides]` call. `cluster/` and `scripts/` are
two different ways of *invoking* that same core: `scripts/*.bat` for a single local run,
`cluster/*.sh`/`*.ps1` for the SLURM sweeps and ablations the paper actually used.

### `src/`

| File | Role |
|---|---|
| `data_preparation/cut_patches.py` | Raw images → cut patches (+ `patch_index.csv`) |
| `data_preparation/compute_dataset_stats.py` | Per-channel mean/std for `augmentation.normalize` |
| `data_preparation/make_image_folds.py` | Image-level fold file for the leakage experiment (CPU only) |
| `pretrain_backbone.py` | **Stage 1**: per-patch backbone pretraining |
| `train_model.py` | **Stage 2**: image-level AMIL fine-tuning (loads the Stage-1 backbone) |
| `test_model.py` | Evaluation: every aggregator below from one forward pass, per-patch test accuracy, McNemar tests, Excel report |
| `hyperparameter_tuning.py` | Optuna search over Stage-1/Stage-2 hyperparameters (see [Known issues](#known-issues)) |
| `batch_training.py` | Chains tune→pretrain→tune→train→test across a (model size, patch size) sweep |
| `merge_split.py` | Recombines a pre-split patch dataset into one folder (the runtime loader does its own tree-level split from a single root) |
| `compute_dpi.py` | Local, no GPU: estimates effective DPI of images/patches from filenames |
| `vram_probe.py` | Local: measures/projects peak VRAM for a given bag size and model size |
| `helper/data_loader.py` | Tree-level leakage-safe split (and the opt-in image-level split), k-fold, bag/patch `Dataset`s and loaders |
| `helper/image_folds.py` | Image ids shared by raw photos and patches; image-level folds and leakage report |
| `helper/model_wrapper.py` | `ConvNeXtAMIL` — builds/trains/validates both stages |
| `helper/model.py` | `PatchAttentionMIL`, `PatchClassifier` |
| `helper/config_cli.py` | YAML loading, `$VAR` expansion, CLI overrides |
| `helper/optimizer_scheduler.py`, `helper/early_stopping.py`, `helper/timing.py` | Small utilities |

### Aggregators reported by `test_model.py`

All computed from the same weights in one forward pass per image:

| Name | Aggregation |
|---|---|
| `amil` | attention MIL: attention-weighted mean of patch embeddings → shared head |
| `amil_uniform` | uniform attention (α = 1/N): plain mean of patch embeddings → same head |
| `amil_vote` / `amil_soft` | hard / soft vote over the Stage-2 model's own per-patch head |
| `vote` / `vote_soft` | hard / soft vote over the Stage-1 patch classifier (two-stage runs only) |

Soft vote = argmax of the mean per-patch softmax. Per-patch accuracy is reported on the
**test** patches (`vote_patch_accuracy`, pooled over all test patches of a fold).

## Setup

```bash
pip install -r requirements.txt
```

Developed and run with **Python 3.11+**, **torch 2.1+** (tested up to 2.14), **timm
1.0.3+** (tested up to 1.0.29). A CUDA GPU is required for anything beyond
`compute_dpi.py`/`merge_split.py`/`make_image_folds.py`; Stage-2 fine-tuning needs enough
VRAM to hold one whole bag of patches (`data.max_patches_per_bag`) — use
`src/vram_probe.py` to check before committing to a model size on your card.

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

   or `scripts/1_prepare_data.bat` (same command, patch size 224). The network input is
   always resized to `augmentation.input_size` (224 in the paper), so other patch sizes
   only need a different `--patch-size` and patch root.

## Reproducing the paper's results

### Local path (generic reproduction, one config/one run at a time)

```
scripts/1_prepare_data.bat            # cut patches (edit patch size inside)
scripts/2-2_pretrain_backbone.bat     # Stage 1: backbone pretraining
scripts/3-2_train_model.bat           # Stage 2: AMIL fine-tuning
scripts/4_testing_model.bat           # evaluation (all aggregators, full + capped bags)
```

Each `.bat` just `cd`s to the repo root and calls the matching `src/*.py -c
configs/config.yaml` — the one config file everything reads. It carries the paper's
hyperparameters; override anything on the command line, e.g.:

```bash
python src/train_model.py -c configs/config.yaml --fold 2
```

(5-fold CV is already `config.yaml`'s default — see
[Leakage-safe splitting](#leakage-safe-splitting-and-fold-assignments).)

### Cluster path (what actually produced the paper's numbers)

Run on the Digital Research Alliance of Canada (DRAC). Every `cluster/*.sh` needs its
`#SBATCH --account=YOUR_ACCOUNT_gpu`/`_cpu` placeholder replaced with your own
allocation (or override at submit time: `sbatch --account=<you> cluster/job_x.sh`), and
`cluster/00_prefetch_weights.sh` run once on a login node first (compute nodes have no
internet). See `cluster/common.sh` for the full environment contract. Every run writes
`provenance.txt` (git commit, config, seed, all knobs) and a `resolved_config.yaml` per
stage next to its outputs.

`cluster/job_train.sh` is the single parametrized SLURM array job behind every training
ablation below — set `PATCH_SIZE`/`MODEL_SIZE`/`STAGES`/`SINGLE_STAGE`/`RUN_PREFIX`/etc.
via `sbatch --export=...` instead of maintaining a separate script per experiment. Its
header comments give the exact invocation for each row, with the settings the paper's
runs used. `cluster/compile_results.py` (wrapped by `cluster/run_compile.sh`) recomputes
every reported metric — accuracy, macro-F1, McNemar, attention concentration, tree-level
vote — from each run's per-bag prediction rows; it recognises the run-name prefixes below
(set with `RUN_PREFIX`), and silently skips any other name.

| Ablation | How | Run prefix → compile |
|---|---|---|
| Patch size (160/224/288/384/512) | `job_train.sh`, `PATCH_SIZE=<N>` | `abl_p<N>_nano` → `run_compile.sh` (default pattern) |
| Aggregation | *(no separate run)* — every `test_model.py` pass reports all aggregators above with paired McNemar tests | already in each run's `classification_results.xlsx` |
| Training regime (1- vs. 2-stage) | `job_train.sh`, `SINGLE_STAGE=1`, `PATCH_SIZE=224` (no Stage-1 pretraining, 40 epochs from ImageNet) vs. the normal 2-stage run at the same patch size | `abl_1stage_nano_224` → `PATTERN='abl_1stage_*_f*'` |
| Backbone capacity (pico/nano/tiny @ 224) | `job_train.sh`, `MODEL_SIZE=pico\|tiny`, `PATCH_SIZE=224`, Stage-1 drop-path 0.15 / 0.35 via `EXTRA_ARGS` (see header) | `abl_msize_<size>_224` → `PATTERN='abl_msize_*_f*'` |
| Whole-image baseline | `run_wholeimage.ps1` (train) → `run_eval_wholeimage.ps1` (one prediction per whole image via `eval_wholeimage.py`) | printed by `run_eval_wholeimage.ps1` itself |
| Test-time bag size | `run_bagsize_inference.ps1` — inference-only re-scoring of already-trained models at bag sizes 4..256 | printed by the script itself |
| Final/best model | `job_train.sh`, `SINGLE_STAGE=1`, `PATCH_SIZE=512` | `final_nano_p512_1stage` → `PATTERN='final*_f*'` |
| Image-level split (leakage) and aggregation baselines | `cluster/v2/` — preflight, one submit script, one compile: see [`cluster/v2/README.md`](cluster/v2/README.md) | `cluster/v2/job_compile.sh` → `tables.md` |

The single-stage runs use the config's Stage-2 optimiser settings unchanged (base LR
1.0e-5, ×15 on the attention/classifier heads); see the `SINGLE_STAGE` block in
`cluster/run_chain.sh`. The whole-image baseline uses rotation 15° (set in
`run_wholeimage.ps1`); the patch models use the config's 30°.

The `.ps1` whole-image/bag-size drivers and `eval_wholeimage.py` run **locally**
(Windows, no SLURM) — the bag-size pass re-scores checkpoints produced on the cluster.

## Leakage-safe splitting and fold assignments

The train/val/test split is **tree-level** (a tree's patches/images never appear in more
than one split) and is computed **at run time** by `helper.data_loader.split_trees()`
from a single patch root. It's deterministic from three config values:

- `project.seed` — RNG seed (42 in the paper)
- `data.split.n_folds` — `5` for the paper's CV; `null` for a plain holdout
- `data.split.fold_index` — which fold is held out as test (`--fold` on the CLI)

Each class is split independently (`random.Random(f"{seed}:{label}")`), so adding or
dropping a species doesn't perturb the other classes' assignments. Within the non-test
trees, `data.val_ratio` (15 %) of each species' trees are held out for validation.
`split_trees()` calls `_assert_disjoint()` on every call, which raises immediately if any
tree ever ends up in two splits. Trees are keyed by species *and* tree id, since raw tree
ids repeat across species.

The resulting per-image assignment is shipped in [`folds/`](folds/README.md)
(`tree_level_folds.csv`; it agrees with `split_trees()` for every tree), together with the
deliberately leaky image-level folds of the leakage experiment
(`image_level_folds.csv`, opt-in with `data.split.unit: image`).

## Known issues

`hyperparameter_tuning.py` was used for the original (local) Optuna search, whose tuned
values are written into `configs/config.yaml`. `batch_training.py` and
`scripts/2-1_*.bat` / `3-1_*.bat` call it with a `--stage pretrain|board` flag that the
script does not define, so those two scripts fail at argument parsing. Run
`python src/hyperparameter_tuning.py -c configs/config.yaml` directly instead. None of
this is needed to reproduce the paper, which uses the values in the config.

## License

[MIT](LICENSE).

## Citation
```
Maeve Côté. "Toward an Optimal Configuration for Tree Species Classification from Bark Images: A Leakage-Safe Ablation Study." arXiv preprint arXiv:2611.05680 (2026).
```
