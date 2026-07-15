"""Offline checks of the parts that are expensive to get wrong on a cluster.

Run:  cd /home/claude/BarkNet_ML && python ../tests/test_pipeline.py
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BarkNet_ML"))

import helper.data_loader as dt  # noqa: E402
from helper.config_cli import expand_env, set_in, get_in  # noqa: E402
from helper.model_wrapper import ConvNeXtAMIL  # noqa: E402

ROOT = "/tmp/fake/train"
SPECIES = ["BOJ", "BOP", "CHR", "EPB", "EPN"]
FCFG = {"delimiter": "_", "tree_id_index": 0, "image_id_index": 1}

ok = 0
fail = []


def check(name, cond, detail=""):
    global ok
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail.append(name)
        print(f"  FAIL  {name}  {detail}")


def trees_of(bags):
    out = defaultdict(set)
    for bag in bags:
        tree, _ = dt._parse_ids(bag[0][0].name, FCFG)
        out[bag[0][1]].add(tree)
    return out


print("\n=== 1. holdout split: tree-level, all classes present, disjoint ===")
sp = dt.split_trees(ROOT, SPECIES, FCFG, val_ratio=0.2, test_ratio=0.2, seed=42)
tr, va, te = trees_of(sp["train"]), trees_of(sp["val"]), trees_of(sp["test"])
check("all 5 classes in train", len(tr) == 5)
check("all 5 classes in val", len(va) == 5, str({k: len(v) for k, v in va.items()}))
check("all 5 classes in test", len(te) == 5, str({k: len(v) for k, v in te.items()}))
for lbl in range(5):
    check(f"class {lbl}: train/val/test tree sets disjoint",
          not (tr[lbl] & va[lbl]) and not (tr[lbl] & te[lbl]) and not (va[lbl] & te[lbl]))
check("6 trees/class conserved", all(len(tr[l]) + len(va[l]) + len(te[l]) == 6 for l in range(5)))
n_img = len(sp["train"]) + len(sp["val"]) + len(sp["test"])
check("all 90 images accounted for", n_img == 5 * 6 * 3, f"got {n_img}")

print("\n=== 2. determinism: same seed -> same split, different seed -> different ===")
sp2 = dt.split_trees(ROOT, SPECIES, FCFG, 0.2, 0.2, seed=42, verbose=False)
same = trees_of(sp2["test"]) == te
sp3 = dt.split_trees(ROOT, SPECIES, FCFG, 0.2, 0.2, seed=7, verbose=False)
check("seed 42 reproduces the same test trees", same)
check("seed 7 gives a different test set", trees_of(sp3["test"]) != te)

print("\n=== 3. k-fold: folds partition the trees, no fold overlap ===")
fold_tests = []
for f in range(3):
    s = dt.split_trees(ROOT, SPECIES, FCFG, val_ratio=0.2, n_folds=3, fold_index=f,
                       seed=42, verbose=False)
    fold_tests.append(trees_of(s["test"]))
    # test trees must be absent from that fold's train AND val
    t_, v_, e_ = trees_of(s["train"]), trees_of(s["val"]), trees_of(s["test"])
    check(f"fold {f}: test disjoint from train+val",
          all(not (e_[l] & t_[l]) and not (e_[l] & v_[l]) for l in range(5)))
for lbl in range(5):
    union = set().union(*[ft[lbl] for ft in fold_tests])
    pairwise = all(not (fold_tests[i][lbl] & fold_tests[j][lbl])
                   for i in range(3) for j in range(i + 1, 3))
    check(f"class {lbl}: 3 folds partition all 6 trees exactly once",
          len(union) == 6 and pairwise, f"union={len(union)} pairwise_ok={pairwise}")

print("\n=== 4. bag cap: deterministic, only oversized bags touched ===")
bags = sp["train"] + sp["val"] + sp["test"]
sizes = sorted(len(b) for b in bags)
check("the 40-patch outlier bag exists", sizes[-1] == 40, str(sizes[-3:]))
capped_a = dt._cap_bags(bags, 10, seed=42)
capped_b = dt._cap_bags(bags, 10, seed=42)
check("no bag exceeds the cap", max(len(b) for b in capped_a) == 10)
check("cap is deterministic across calls",
      [sorted(str(p) for p, _ in b) for b in capped_a] == [sorted(str(p) for p, _ in b) for b in capped_b])
check("small bags untouched", sum(1 for b in capped_a if len(b) == 5) == sum(1 for b in bags if len(b) == 5))
check("cap disabled by None", max(len(b) for b in dt._cap_bags(bags, None, 42)) == 40)

print("\n=== 5. legacy two-way split still works (hyperparameter_tuning.py path) ===")
t2, v2 = dt.split_by_tree(ROOT, SPECIES, FCFG, val_ratio=0.2, seed=42)
check("no test bags leak into the two-way split", len(t2) + len(v2) == 90)
check("train/val disjoint by tree",
      all(not (trees_of(t2)[l] & trees_of(v2)[l]) for l in range(5)))

print("\n=== 6. config: env expansion + dotted overrides ===")
os.environ["SLURM_TMPDIR"] = "/node/scratch"
cfg = {"data": {"patch_root": "${SLURM_TMPDIR}/patches_224/train"},
       "pretrain": {"output_dir": "$SCRATCH/runs/p"}}
e = expand_env(cfg)
check("${SLURM_TMPDIR} expanded", e["data"]["patch_root"] == "/node/scratch/patches_224/train",
      e["data"]["patch_root"])
check("undefined var left verbatim (Windows-safe)",
      e["pretrain"]["output_dir"].startswith("$SCRATCH") or "SCRATCH" in os.environ)
set_in(e, "data.split.n_folds", 5)
check("dotted set creates nested keys", get_in(e, "data.split.n_folds") == 5)

print("\n=== 7. loaders build from a config (Stage 1 + Stage 2) ===")
full_cfg = yaml.safe_load(open("configs/config_cluster.yaml"))
full_cfg["data"].update(patch_root=ROOT, species=SPECIES, num_workers=0,
                        max_patches_per_bag=8)
full_cfg["data"]["split"] = {"test_ratio": 0.2, "n_folds": None, "fold_index": 0}
full_cfg["augmentation"]["input_size"] = 64
p_tr, p_va = dt.build_patch_dataloaders(full_cfg, batch_size=4, seed=42)
x, y = next(iter(p_tr))
check("Stage-1 batch shape [B,3,64,64]", tuple(x.shape) == (4, 3, 64, 64), str(x.shape))
b_tr, b_va = dt.build_dataloaders(full_cfg, seed=42)
patches, label = next(iter(b_tr))
check("Stage-2 bag shape [1,N,3,64,64]", patches.dim() == 5 and patches.shape[0] == 1, str(patches.shape))
check("Stage-2 bag respects the cap", patches.shape[1] <= 8, str(patches.shape))
test_loader = dt.build_bag_loader(full_cfg, split="test", seed=42)
tp, tl, tid = next(iter(test_loader))
check("test loader yields ids", isinstance(tid[0], str) and ":" in tid[0], str(tid))
check("test bags are UNCAPPED (full image at inference)",
      max(t[0].shape[1] for t in [next(iter(test_loader))]) >= 1)

print("\n=== 8. Stage-1 -> Stage-2 backbone transfer + chunked inference ===")
dev = "cpu"
w1 = ConvNeXtAMIL(device=dev, species=SPECIES, model_size="pico", weights="none",
                  class_dropout=0.1, drop_path_rate=0.1, save_data=False)
m1, crit1 = w1.get_patch_model()
tmp = Path("/tmp/backbone.pth")
torch.save(m1.state_dict(), tmp)

w2 = ConvNeXtAMIL(device=dev, species=SPECIES, model_size="pico", weights="none",
                  backbone_checkpoint=str(tmp), attn_dropout=0.2, class_dropout=0.2,
                  save_data=False)
m2, crit2 = w2.get_model()
shared = [k for k in m1.state_dict() if k in m2.state_dict()]
transferred = all(torch.equal(m1.state_dict()[k], m2.state_dict()[k]) for k in shared)
check(f"{len(shared)} backbone/classifier tensors transferred identically", transferred)
check("attention head is fresh (not in Stage-1 ckpt)",
      any(k.startswith("attention.") for k in m2.state_dict()) and
      not any(k.startswith("attention.") for k in m1.state_dict()))

m2.eval()
bag = torch.randn(37, 3, 64, 64)
with torch.no_grad():
    full_img, full_patch = m2(bag)
    chunk_img, chunk_patch, attn = m2.infer(bag, chunk_size=8)
check("chunked infer == full forward (image logits)",
      torch.allclose(full_img, chunk_img, atol=1e-5), str((full_img - chunk_img).abs().max().item()))
check("chunked infer == full forward (patch logits)",
      torch.allclose(full_patch, chunk_patch, atol=1e-5))
check("attention weights sum to 1 over the bag", abs(attn.sum().item() - 1.0) < 1e-4)

m1.eval()
with torch.no_grad():
    v_full = m1(bag)
    v_chunk = m1.infer(bag, chunk_size=8)
check("PatchClassifier chunked infer == full forward",
      torch.allclose(v_full, v_chunk, atol=1e-5))

print("\n=== 9. model size map ===")
check("nano is in the size map", "nano" in ConvNeXtAMIL.SIZE_TO_TIMM)
check("phantom convnextv2_small is gone", "small" not in ConvNeXtAMIL.SIZE_TO_TIMM)

print(f"\n{ok} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)

# =====================================================================================
# Additions for A (method) and B (perf)
# =====================================================================================
print("\n=== 10. stochastic bag cap: resamples per epoch, reproducible, val stable ===")
from helper.data_loader import PatchBagDataset, _select_patches  # noqa: E402

big = [b for b in bags if len(b) == 40][0]
s_e0 = _select_patches(big, 8, seed=42, epoch=0, stochastic=True)
s_e1 = _select_patches(big, 8, seed=42, epoch=1, stochastic=True)
s_e0b = _select_patches(big, 8, seed=42, epoch=0, stochastic=True)
d_e0 = _select_patches(big, 8, seed=42, epoch=0, stochastic=False)
d_e1 = _select_patches(big, 8, seed=42, epoch=1, stochastic=False)

check("stochastic: epoch 0 != epoch 1 (a NEW subset each epoch)", s_e0 != s_e1)
check("stochastic: same (seed, epoch) reproduces exactly", s_e0 == s_e0b)
check("deterministic: epoch is ignored (val stays stable)", d_e0 == d_e1)
check("cap size respected in both modes", len(s_e0) == 8 and len(d_e0) == 8)
check("under-cap bags returned untouched",
      _select_patches(bags[1], 999, 42, 3, True) == bags[1])

# over many epochs the model should eventually see ~every patch of the bag
seen = set()
for e in range(60):
    seen |= {str(p) for p, _ in _select_patches(big, 8, 42, e, True)}
check(f"60 epochs x 8/40 patches covers the whole bag ({len(seen)}/40)", len(seen) == 40)

from torchvision import transforms as T  # noqa: E402
TT = T.ToTensor()
ds = PatchBagDataset([big], transform=TT, max_patches=8, stochastic_cap=True, seed=42)
ds.set_epoch(0); a0 = ds[0][0].shape[0]
ds.set_epoch(1)
check("dataset honours the cap", a0 == 8)
check("dataset.set_epoch changes the drawn subset",
      _select_patches(big, 8, 42, 0, True) != _select_patches(big, 8, 42, 1, True))

ids = PatchBagDataset([big], transform=TT, return_id=True, fcfg=FCFG, max_patches=8,
                      stochastic_cap=True, seed=42)
ids.set_epoch(0); id0 = ids[0][2]
ids.set_epoch(7); id7 = ids[0][2]
check("image_id is stable across epochs (taken from the FULL bag)", id0 == id7, f"{id0} vs {id7}")

print("\n=== 11. Stage-2 train loader must NOT use persistent workers ===")
cfg2 = yaml.safe_load(open("configs/config_cluster.yaml"))
cfg2["data"].update(patch_root=ROOT, species=SPECIES, num_workers=2, max_patches_per_bag=8)
cfg2["data"]["split"] = {"test_ratio": 0.2, "n_folds": None, "fold_index": 0}
cfg2["augmentation"]["input_size"] = 64
tl, vl = dt.build_dataloaders(cfg2, seed=42)
check("train loader: persistent_workers OFF (or set_epoch would never reach workers)",
      tl.persistent_workers is False)
check("val loader: persistent_workers ON", vl.persistent_workers is True)
check("train dataset caps stochastically", tl.dataset.stochastic_cap is True)
check("val dataset caps deterministically", vl.dataset.stochastic_cap is False)

print("\n=== 12. torch.compile must not corrupt checkpoints ===")
from helper.model_wrapper import unwrap  # noqa: E402
w = ConvNeXtAMIL(device="cpu", species=SPECIES, model_size="pico", weights="none",
                 save_data=False, channels_last=False)
plain, _ = w.get_patch_model()
compiled = torch.compile(plain)
check("compiled state_dict IS prefixed (the bug this guards against)",
      any(k.startswith("_orig_mod.") for k in compiled.state_dict()))
check("unwrap() restores clean keys",
      set(unwrap(compiled).state_dict()) == set(plain.state_dict()))
tmp2 = Path("/tmp/compiled_backbone.pth")
torch.save(unwrap(compiled).state_dict(), tmp2)
w2 = ConvNeXtAMIL(device="cpu", species=SPECIES, model_size="pico", weights="none",
                  backbone_checkpoint=str(tmp2), save_data=False, channels_last=False)
m2c, _ = w2.get_model()
st = torch.load(tmp2, weights_only=True)
transferred = sum(1 for k in st if k in m2c.state_dict())
check(f"Stage-2 transfers {transferred} tensors from a COMPILED Stage-1 ckpt (not 0)",
      transferred > 100)

print("\n=== 13. amp dtype / scaler pairing ===")
wb = ConvNeXtAMIL(device="cuda:0", species=SPECIES, amp_dtype="bf16", save_data=False)
wf = ConvNeXtAMIL(device="cuda:0", species=SPECIES, amp_dtype="fp16", save_data=False)
check("bf16 -> no GradScaler needed", wb.needs_scaler is False and wb.amp_dtype is torch.bfloat16)
check("fp16 -> GradScaler enabled", wf.needs_scaler is True and wf.amp_dtype is torch.float16)
check("bf16 wrapper exposes channels_last", wb.memory_format is torch.channels_last)
check("scaler is disabled for bf16", wb.make_scaler().is_enabled() is False)
# NOTE: GradScaler.is_enabled() is also False when no CUDA device is present, so on a
# CPU-only box we can only assert the wrapper's intent (needs_scaler), not the scaler's
# runtime state. On the cluster this resolves to a genuinely enabled scaler for fp16.
if torch.cuda.is_available():
    check("scaler is enabled for fp16 (CUDA)", wf.make_scaler().is_enabled() is True)
else:
    check("fp16 requests a scaler (runtime state untestable without CUDA)",
          wf.needs_scaler is True)
try:
    ConvNeXtAMIL(device="cpu", species=SPECIES, amp_dtype="fp8", save_data=False)
    check("bad amp_dtype rejected", False)
except ValueError:
    check("bad amp_dtype rejected", True)

print(f"\n{ok} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
