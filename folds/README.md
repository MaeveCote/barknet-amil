# Per-image fold assignments

The 5-fold cross-validation assignment of every one of the 23,271 BarkNet 1.0 images
(20 species) used in the paper. One row per image, with the same columns in both files:

| Column | Meaning |
|---|---|
| `image_id` | `<SPECIES>:<tree>:<uid>`, where `uid` is the image's index among its tree's photos sorted by file name (the `imageUID` that `src/data_preparation/cut_patches.py` writes into every patch name) |
| `tree_id` | tree number from the file name. Tree ids are **not** unique across species (tree 882 exists for MEL and for ORA), so a tree is `(species, tree_id)` |
| `species` | BarkNet species code |
| `fold` | the fold (0–4) in which the image is in the **test** set |
| `src_image` | original BarkNet file name |

## `tree_level_folds.csv`: the paper's evaluation

Leakage-safe: each of the 999 `(species, tree)` pairs is in exactly one fold. The code
computes this split at run time (`helper.data_loader.split_trees`, `project.seed=42`,
`data.split.n_folds=5`), so this file is for reference. It was exported from the
per-image test predictions of the experiments and checked to agree with `split_trees`
for every tree. The fold is the same for every patch size and backbone. Within the
non-test trees of a fold, 15 % of the trees of each species are held out for validation
(`data.val_ratio`).

## `image_level_folds.csv`: the leakage experiment

Deliberately **not** leakage-safe, used only to measure how much an image-level split
inflates accuracy. Built by `src/data_preparation/make_image_folds.py` with
`StratifiedKFold(5, shuffle=True, random_state=42)` over images, stratified by species,
ignoring trees. `image_level_folds_leakage.csv` gives, per fold, the share of test images
whose tree also appears in training (99.9–100 %). Use it through
`--set data.split.unit=image --set data.split.fold_file=folds/image_level_folds.csv`;
see `cluster/v2/README.md`.
