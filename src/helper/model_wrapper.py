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

class ConvNeXtAMIL:
    SIZE_TO_TIMM = {
        "atto": "convnextv2_atto.fcmae_ft_in1k",
        "femto": "convnextv2_femto.fcmae_ft_in1k",
        "pico": "convnextv2_pico.fcmae_ft_in1k",
        "tiny": "convnextv2_tiny.fcmae_ft_in1k",
        "small": "convnextv2_small.fcmae_ft_in1k",
        "base": "convnextv2_base.fcmae_ft_in1k",
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

        # device_type for autocast must be "cuda"/"cpu", never the full "cuda:0" string.
        self.amp_device = "cuda" if "cuda" in str(self.device) else "cpu"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_extractor(self, use_imagenet):
        if self.model_size not in self.SIZE_TO_TIMM:
            valid = ", ".join(self.SIZE_TO_TIMM)
            raise ValueError(f"Invalid size '{self.model_size}'. Choose from: {valid}.")
        # num_classes=0 strips timm's head and returns pooled embeddings per patch.
        return timm.create_model(
            self.SIZE_TO_TIMM[self.model_size],
            pretrained=use_imagenet,
            num_classes=0,
            drop_path_rate=self.drop_path_rate,
        )

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

        model.to(self.device)
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

        model.to(self.device)
        return model, self._criterion()

    def get_last_conv_layer(self, model):
        """Grad-CAM target: the last block of the extractor's final stage."""
        return [model.extractor.stages[-1].blocks[-1]]

    # ------------------------------------------------------------------ #
    # Stage 1 — patch-level loops (standard mini-batches)
    # ------------------------------------------------------------------ #
    def train_patch_epoch(self, model, criterion, optimizer, scaler, loader, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"[Stage 1] Epoch {epoch} Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            with torch.autocast(device_type=self.amp_device, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        return epoch_loss, epoch_acc

    def validate_patch_epoch(self, model, criterion, loader):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="[Stage 1] Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.autocast(device_type=self.amp_device, dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / max(total, 1), 100.0 * correct / max(total, 1)

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
            patches = patches.squeeze(0).to(self.device)
            label = label.to(self.device)

            with torch.autocast(device_type=self.amp_device, dtype=torch.float16):
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
            torch.save(model.state_dict(), Path(result_dir) / f"state_{epoch}.pth")

        return epoch_loss, epoch_acc

    def validate_epoch(self, model, criterion, val_loader):
        """Validation uses the image-level prediction only (no auxiliary term)."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for patches, label in tqdm(val_loader, desc="[Stage 2] Validating", leave=False):
                patches = patches.squeeze(0).to(self.device)
                label = label.to(self.device)

                with torch.autocast(device_type=self.amp_device, dtype=torch.float16):
                    image_logits, _ = model(patches)
                    loss = criterion(image_logits, label)

                running_loss += loss.item()
                _, predicted = image_logits.max(1)
                total += 1
                correct += predicted.eq(label).sum().item()

        val_loss = running_loss / max(len(val_loader), 1)
        val_acc = 100.0 * correct / max(total, 1)
        return val_loss, val_acc
