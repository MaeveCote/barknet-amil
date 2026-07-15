"""ConvNeXt-V2 backbone wrapped for the two-stage bark pipeline.

Stage 1 (``get_patch_model`` + ``*_patch_epoch``): the backbone and a shared classifier
are trained as a plain per-patch classifier (one label per patch), like the original
bark papers. The best checkpoint is saved as ``best_backbone.pth``.

Stage 2 (``get_model`` + ``*_epoch``): the AMIL model is built, its backbone and
classifier are initialised from the Stage-1 checkpoint, and the attention head plus
classifier are fine-tuned end-to-end with the auxiliary instance loss.

Only ConvNeXt + AMIL is supported, so there is no architecture abstraction layer.
"""
from pathlib import Path

import timm
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from helper.model import PatchAttentionMIL, PatchClassifier


def unwrap(model):
    """Return the real module behind a ``torch.compile`` wrapper.

    torch.compile returns an OptimizedModule whose ``state_dict()`` keys are all prefixed
    with ``_orig_mod.``. Saving that checkpoint would be quietly catastrophic here: Stage 2
    transfers the backbone with ``strict=False``, so every key would be "unexpected", ZERO
    tensors would transfer, and the run would train from ImageNet init while printing a
    cheerful success message. Always save ``unwrap(model).state_dict()``.
    """
    return getattr(model, "_orig_mod", model)


class ConvNeXtAMIL:
    # Verified against timm 1.0.24/1.0.28: the ConvNeXt-V2 family is
    # atto/femto/pico/nano/tiny/base/large/huge. `convnextv2_small` exists as an
    # architecture but has NO pretrained tag -- `convnextv2_small.fcmae_ft_in1k` does
    # not exist. The size ladder for this project is pico -> nano -> tiny -> base.
    SIZE_TO_TIMM = {
        "atto": "convnextv2_atto.fcmae_ft_in1k",
        "femto": "convnextv2_femto.fcmae_ft_in1k",
        "pico": "convnextv2_pico.fcmae_ft_in1k",
        "nano": "convnextv2_nano.fcmae_ft_in1k",
        "tiny": "convnextv2_tiny.fcmae_ft_in1k",
        "base": "convnextv2_base.fcmae_ft_in1k",
        "large": "convnextv2_large.fcmae_ft_in1k",
        "huge": "convnextv2_huge.fcmae_ft_in1k",
    }

    def __init__(
            self,
            device,
            species,
            model_size="pico",
            weights="default",
            pretrained_checkpoint=None,
            backbone_checkpoint=None,
            attn_dropout=0.1,
            class_dropout=0.4,
            drop_path_rate=0.2,
            label_smoothing=0.1,
            instance_loss_weight=0.5,
            save_data=True,
            pretrained_file=None,
            amp_dtype="bf16",
            channels_last=True,
            compile_model=False,
    ):
        self.device = device
        self.species = species
        self.model_size = str(model_size).lower()
        self.weights = str(weights).lower()
        self.pretrained_checkpoint = pretrained_checkpoint   # full AMIL resume (strict)
        self.backbone_checkpoint = backbone_checkpoint        # Stage-1 backbone transfer
        self.attn_dropout = attn_dropout if attn_dropout is not None else 0.1
        self.class_dropout = class_dropout if class_dropout is not None else 0.4
        self.drop_path_rate = drop_path_rate if drop_path_rate is not None else 0.2
        self.label_smoothing = label_smoothing
        self.instance_loss_weight = instance_loss_weight
        self.save_data = save_data
        # Optional path to an ImageNet weight file on disk (.safetensors / .bin), for
        # compute nodes with no internet. If unset, timm resolves through the HF cache
        # (pre-warm it on the login node and export HF_HUB_OFFLINE=1).
        self.pretrained_file = pretrained_file

        # device_type for autocast must be "cuda"/"cpu", never the full "cuda:0" string.
        self.amp_device = "cuda" if "cuda" in str(self.device) else "cpu"

        # --- performance -------------------------------------------------------------
        # bf16 on an H100: same speed as fp16, wider exponent range, and NO GradScaler
        # needed (fp16's scaler exists to stop small gradients underflowing; bf16 has
        # fp32's exponent so they don't). fp16 stays available for older cards.
        self.amp_dtype_name = str(amp_dtype).lower()
        if self.amp_dtype_name in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        elif self.amp_dtype_name in ("fp16", "float16", "half"):
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"amp_dtype must be bf16 or fp16, got '{amp_dtype}'.")
        # bf16 on CPU is fine; fp16 on CPU is not well supported -- fall back.
        if self.amp_device == "cpu" and self.amp_dtype is torch.float16:
            self.amp_dtype = torch.bfloat16

        # GradScaler is only meaningful for fp16. Constructing it disabled for bf16 keeps
        # the call sites identical (scaler.scale/step/update become pass-throughs).
        self.needs_scaler = self.amp_dtype is torch.float16

        # ConvNets hit the tensor cores through NHWC. PyTorch's default NCHW forces a
        # transpose on every conv; channels_last removes it.
        self.channels_last = bool(channels_last) and self.amp_device == "cuda"
        self.memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        self.compile_model = bool(compile_model)

    def make_scaler(self):
        """The scaler the training scripts should use, matched to amp_dtype."""
        return torch.amp.GradScaler(device=self.amp_device, enabled=self.needs_scaler)

    def _finalise(self, model):
        """Common post-construction steps: device, memory format, optional compile."""
        model.to(self.device)
        if self.channels_last:
            model.to(memory_format=torch.channels_last)
        if self.compile_model:
            # Stage 2 feeds ragged bag shapes [N, C, H, W] with N varying per image, which
            # makes torch.compile recompile constantly. Only Stage 1 (fixed batch shape)
            # should set this.
            print("torch.compile enabled (first batch will be slow while it warms up).")
            model = torch.compile(model)
        return model

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_extractor(self, use_imagenet):
        if self.model_size not in self.SIZE_TO_TIMM:
            valid = ", ".join(self.SIZE_TO_TIMM)
            raise ValueError(f"Invalid size '{self.model_size}'. Choose from: {valid}.")

        tag = self.SIZE_TO_TIMM[self.model_size]
        kwargs = dict(pretrained=use_imagenet, num_classes=0, drop_path_rate=self.drop_path_rate)

        if use_imagenet and self.pretrained_file:
            weight_path = Path(self.pretrained_file)
            if not weight_path.exists():
                raise FileNotFoundError(
                    f"model.pretrained_file does not exist: {weight_path}. It must point at "
                    f"the ImageNet weight file for '{tag}' (.safetensors or .bin)."
                )
            # timm's documented way to load ImageNet weights from disk instead of the hub.
            kwargs["pretrained_cfg_overlay"] = dict(file=str(weight_path))
            print(f"Loading ImageNet weights for {tag} from local file {weight_path}")

        try:
            return timm.create_model(tag, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- surface the offline case explicitly
            if use_imagenet:
                raise RuntimeError(
                    f"Could not materialise pretrained weights for '{tag}'. On a compute node "
                    f"there is no internet: pre-cache the weights on the login node "
                    f"(HF_HOME=$HOME/.cache/huggingface) and export HF_HUB_OFFLINE=1 in the job, "
                    f"or set model.pretrained_file to the weight file on disk. Original error: {exc}"
                ) from exc
            raise

    def _criterion(self):
        if self.label_smoothing:
            return CrossEntropyLoss(label_smoothing=self.label_smoothing)
        return CrossEntropyLoss()

    def get_patch_model(self):
        """Stage-1 model: extractor + shared classifier (no attention)."""
        resuming = bool(self.backbone_checkpoint) and Path(self.backbone_checkpoint).exists()
        use_imagenet = (self.weights == "default") and not resuming

        extractor = self._build_extractor(use_imagenet)
        model = PatchClassifier(
            extractor=extractor,
            in_features=extractor.num_features,
            num_classes=len(self.species),
            dropout_p_classifier=self.class_dropout,
        )

        if resuming:
            state = torch.load(
                self.backbone_checkpoint, map_location=self.device, weights_only=True
            )
            model.load_state_dict(state)
            print(f"Resumed Stage-1 backbone from {self.backbone_checkpoint}")

        model = self._finalise(model)
        return model, self._criterion()

    def get_model(self):
        """Stage-2 AMIL model, optionally initialised from a Stage-1 backbone."""
        use_imagenet = (
                self.weights == "default"
                and self.backbone_checkpoint is None
                and self.pretrained_checkpoint is None
        )
        extractor = self._build_extractor(use_imagenet)
        in_features = extractor.num_features

        model = PatchAttentionMIL(
            extractor=extractor,
            in_features=in_features,
            num_classes=len(self.species),
            dropout_p_attention=self.attn_dropout,
            dropout_p_classifier=self.class_dropout,
        )

        # Transfer the Stage-1 backbone + classifier (attention stays freshly init).
        if self.backbone_checkpoint:
            ckpt_path = Path(self.backbone_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"backbone_checkpoint not found: {ckpt_path}. Run pretrain.py first "
                    f"or set model.backbone_checkpoint to null."
                )
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            transferred = [k for k in state if k not in unexpected]
            print(
                f"Loaded Stage-1 backbone from {ckpt_path} "
                f"({len(transferred)} tensors transferred; attention head fresh)."
            )

        # Resume a full AMIL model (overrides everything above).
        if self.pretrained_checkpoint:
            state = torch.load(
                self.pretrained_checkpoint, map_location=self.device, weights_only=True
            )
            model.load_state_dict(state)
            print(f"Resumed full AMIL model from {self.pretrained_checkpoint}")

        model = self._finalise(model)
        return model, self._criterion()

    def get_last_conv_layer(self, model):
        """Grad-CAM target: the last block of the extractor's final stage."""
        return [model.extractor.stages[-1].blocks[-1]]

    # ------------------------------------------------------------------ #
    # Stage 1 — patch-level loops (standard mini-batches)
    # ------------------------------------------------------------------ #
    def train_patch_epoch(self, model, criterion, optimizer, scaler, loader, epoch, total=None):
        """``total`` lets the caller pass an explicit batch count when ``loader`` is
        something like an itertools.islice wrapper that has no __len__ of its own
        (e.g. the hyperparameter search capping batches-per-epoch) -- without it,
        tqdm can't show a percentage or ETA and falls back to a bare iteration count."""
        model.train()
        running_loss = 0.0
        correct = 0
        total_seen = 0

        for images, labels in tqdm(loader, desc=f"[Stage 1] Epoch {epoch} Training", total=total):
            images = images.to(self.device, memory_format=self.memory_format)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            with torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total_seen += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / max(total_seen, 1)
        epoch_acc = 100.0 * correct / max(total_seen, 1)
        return epoch_loss, epoch_acc

    def validate_patch_epoch(self, model, criterion, loader, total=None):
        model.eval()
        running_loss = 0.0
        correct = 0
        total_seen = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="[Stage 1] Validating", leave=False, total=total):
                images = images.to(self.device, memory_format=self.memory_format)
                labels = labels.to(self.device)

                with torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                total_seen += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / max(total_seen, 1), 100.0 * correct / max(total_seen, 1)

    # ------------------------------------------------------------------ #
    # Stage 2 — image-level AMIL loops (one bag at a time, accumulated)
    # ------------------------------------------------------------------ #
    def _combined_loss(self, image_logits, patch_logits, label, criterion):
        """L = L_bag + lambda * L_instance (label broadcast to every patch)."""
        loss_bag = criterion(image_logits, label)
        patch_labels = label.expand(patch_logits.size(0))
        loss_instance = criterion(patch_logits, patch_labels)
        return loss_bag + self.instance_loss_weight * loss_instance

    def train_epoch(
            self, model, criterion, optimizer, scaler, train_loader, epoch,
            result_dir=None, accumulation_steps=16,
    ):
        """One epoch of image-level AMIL training with gradient accumulation."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for i, (patches, label) in enumerate(
                tqdm(train_loader, desc=f"[Stage 2] Epoch {epoch} Training")
        ):
            # Drop the dummy batch dim so the model sees a single bag [N, C, H, W].
            patches = patches.squeeze(0).to(self.device, memory_format=self.memory_format)
            label = label.to(self.device)

            with torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                image_logits, patch_logits = model(patches)
                loss = self._combined_loss(image_logits, patch_logits, label, criterion)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            _, predicted = image_logits.max(1)
            total += 1
            correct += predicted.eq(label).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / max(total, 1)

        if self.save_data and result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            torch.save(unwrap(model).state_dict(), Path(result_dir) / f"state_{epoch}.pth")

        return epoch_loss, epoch_acc

    def validate_epoch(self, model, criterion, val_loader):
        """Validation uses the image-level prediction only (no auxiliary term)."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for patches, label in tqdm(val_loader, desc="[Stage 2] Validating", leave=False):
                patches = patches.squeeze(0).to(self.device, memory_format=self.memory_format)
                label = label.to(self.device)

                with torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                    image_logits, _ = model(patches)
                    loss = criterion(image_logits, label)

                running_loss += loss.item()
                _, predicted = image_logits.max(1)
                total += 1
                correct += predicted.eq(label).sum().item()

        val_loss = running_loss / max(len(val_loader), 1)
        val_acc = 100.0 * correct / max(total, 1)
        return val_loss, val_acc
