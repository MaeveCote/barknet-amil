# v2 experiments — runbook

New experiments for v2 of the paper: an **image-level split** (deliberately leaky, to
measure how much it inflates accuracy) and **inference-only aggregation baselines**
(soft vote, uniform attention) on the existing tree-level checkpoints. The pipeline is the
paper's, unchanged: same 20 species, hyperparameters, seed (42), augmentation, epochs,
training bag cap (128) and evaluation code. Nothing is re-tuned, no checkpoint is chosen
on a test fold, and the tree-level runs are only ever read.

Every new output goes under `$SCRATCH/results/v2/` (`V2_ROOT`, see `v2_env.sh`).

## 0. Put this code on the cluster (login node)

The cluster checkout still has uncommitted edits from the paper runs; stash them (they
stay recoverable with `git stash list`), then pull:

```bash
cd ~/BarkNet_ML && git stash push -m "pre-v2 cluster edits" && git pull
```

Optional but recommended (`/project` is not purged): back up the tree-level runs first.

```bash
rsync -a --info=progress2 $SCRATCH/runs/ ~/links/projects/def-oberman/barknet_runs_backup/
```

## 1. Preflight (login node, no GPU, a few minutes)

```bash
bash cluster/v2/00_preflight.sh
```

Read-only on existing runs. It writes `$V2_ROOT/preflight/preflight_<stamp>.txt` with:
- every tree-level checkpoint Part B needs, whether it exists, and how many days since it
  was last touched (scratch purges after about 60 days);
- saved per-patch logit files (none are expected, so soft vote and uniform attention are
  recomputed on GPU from the checkpoints);
- the patch tars and raw photos;
- the **image-level fold file** `$V2_ROOT/folds/image_folds.csv` (`image_id, tree_id,
  species, fold, src_image`), built once with `StratifiedKFold(5, shuffle=True, seed=42)`
  over the 23,271 images, plus `image_folds_leakage.csv`: per fold, the share of test
  images whose tree is also in training (it should be about 100%);
- a verdict, `MODE=full` or `MODE=minimal` (checkpoints purged).

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
| B1, B2, B4 | 1 array (40 tasks, at most 10 running at once) | re-scores existing tree-level checkpoints: nano 160–512 two-stage, nano 224 single-stage, pico/tiny 224 | `eval/tree_*_f<k>/` |
| compile | 1 CPU | builds the tables once everything above has ended | `tables/` |

`MODE=minimal` (set automatically when B1/B2 checkpoints are missing; override with
`MODE=...`) replaces the B array with retraining tree-level nano 512 two-stage and nano 224
single-stage (paper settings, 5 folds each) into `runs/retrain_*`; their test steps produce
Part B. `INCLUDE_B4=0` skips the pico/tiny re-scoring.

Each job refuses to overwrite a finished result, and `submit_v2.sh` refuses to run twice
(`$V2_ROOT/submitted.lock`; `FORCE=1` to override). To re-run just one failed fold, re-run
its job by hand. Finished folds are skipped.

## 3. Collect

When the compile job has run, everything is in `$V2_ROOT/tables/`:

- `tables.md`: Table 1 (tree vs image split), Table 2 (aggregators × patch size,
  accuracy and macro-F1), Table 3 (single-stage 224), B4, per-fold Δ(attention MIL −
  uniform / soft vote) with their signs, and checks: re-scored checkpoints must reproduce
  the paper's accuracies exactly, recomputed tree-level values vs the quoted references,
  and per-fold tree leakage of the image folds.
- CSVs behind every table, `per_fold_metrics.csv`, `summary_metrics.csv`,
  `fold_deltas.csv`, and `predictions/<arm>.csv` (`image_id, tree_id, true, pred_*, fold`).

```bash
scp -r <user>@rorqual.alliancecan.ca:/scratch/<user>/results/v2/tables .
```

You can re-run the compile by hand at any time to see partial results:
`bash cluster/v2/job_compile.sh`.

## Provenance

Every job appends the git commit (and any uncommitted files), config path and checksum,
seed and all knobs to `<run>/provenance.txt`. Each stage also writes its fully-resolved
config (`resolved_config.yaml`). Commit before pulling on the cluster so the hash means
something.

## Definitions

- **Image id** `<SPECIES>:<tree>:<uid>`: `uid` is the image's index among its tree's raw
  files sorted by name, the same `imageUID` that `cut_patches.py` writes into patch
  names. Whole-image and patch runs therefore share one fold assignment, and every patch
  inherits its image's fold. Jobs check ids against the patch tar's `patch_index.csv`.
- **Aggregators** (all from one forward pass of the same weights): attention MIL
  (`amil`), uniform attention (α = 1/N: mean patch embedding → same head,
  `amil_uniform`), attention-MIL vote (hard vote over the MIL patch head, `amil_vote`),
  soft vote over the MIL patch head (`amil_soft`), majority and soft vote over the Stage-1
  classifier (`vote`, `vote_soft`; two-stage runs only).
- **Per-patch accuracy**: the paper's number is the best Stage-1 *validation* patch
  accuracy. Per-patch *test* accuracy is reported as well.
- **Effective attention fraction**: exp(H(α))/N per bag, averaged over bags.
- **A1 settings**: the tree-level whole-image reference was trained with rotation 15°
  and batch 64 (the local config at the time), so A1 uses exactly those. The patch runs
  use 30°.
