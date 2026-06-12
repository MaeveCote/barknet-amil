"""Attention-based Multiple Instance Learning over the patches of a single image.

A *bag* is one bark image, represented as ``N`` patches of shape ``[N, C, H, W]``.
A shared ConvNeXt extractor embeds every patch; an attention MLP scores each patch;
the bag embedding is the attention-weighted sum of the patch *embeddings* (pooling
happens in embedding space, not in class space). A shared linear head then produces
the image-level prediction.

The same shared head is also applied independently to every patch embedding. Those
per-patch logits feed the auxiliary instance loss ``L = L_bag + lambda * L_instance``,
which restores a dense per-patch gradient signal that vanilla attention pooling
otherwise suppresses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAttentionMIL(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        in_features: int,
        num_classes: int,
        attn_hidden: int = 128,
        dropout_p_attention: float = 0.1,
        dropout_p_classifier: float = 0.4,
    ):
        super().__init__()
        self.extractor = extractor

        # Attention head: scores each patch embedding with a scalar weight.
        self.attention = nn.Sequential(
            nn.Linear(in_features, attn_hidden),
            nn.Tanh(),
            nn.Dropout(p=dropout_p_attention),
            nn.Linear(attn_hidden, 1),
        )

        # Single classifier head, shared by the pooled bag and the per-patch path.
        self.dropout = nn.Dropout(p=dropout_p_classifier)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        """Args:
            x: ``[N, C, H, W]`` patches of a single image.

        Returns:
            image_logits: ``[1, num_classes]`` attention-pooled bag prediction.
            patch_logits: ``[N, num_classes]`` per-patch predictions (auxiliary loss).
        """
        features = self.extractor(x)  # [N, in_features]

        # Dense per-patch supervision signal.
        patch_logits = self.classifier(self.dropout(features))  # [N, num_classes]

        # Attention pooling in embedding space.
        attn = F.softmax(self.attention(features), dim=0)        # [N, 1]
        pooled = torch.sum(attn * features, dim=0, keepdim=True)  # [1, in_features]
        image_logits = self.classifier(self.dropout(pooled))     # [1, num_classes]

        return image_logits, patch_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-patch softmax attention weights ``[N, 1]`` (for heatmaps)."""
        features = self.extractor(x)
        return F.softmax(self.attention(features), dim=0)


class PatchClassifier(nn.Module):
    """Stage-1 patch classifier: shared ConvNeXt extractor + linear head.

    Trains the backbone and the shared classifier as a plain per-patch classifier
    (one label per patch), exactly like the original bark papers. Its ``extractor``
    and ``classifier`` submodules share names with :class:`PatchAttentionMIL`, so the
    learned weights transfer straight into the AMIL model for Stage-2 fine-tuning.
    """

    def __init__(self, extractor, in_features, num_classes, dropout_p_classifier=0.4):
        super().__init__()
        self.extractor = extractor
        self.dropout = nn.Dropout(p=dropout_p_classifier)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x is a standard mini-batch of patches [B, C, H, W].
        features = self.extractor(x)
        return self.classifier(self.dropout(features))