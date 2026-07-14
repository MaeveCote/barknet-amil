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
    def _features_chunked(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Embed a bag in chunks so peak activation memory is bounded by ``chunk_size``
        instead of by the bag size. Only valid without gradients."""
        if chunk_size is None or chunk_size <= 0 or x.size(0) <= chunk_size:
            return self.extractor(x)
        return torch.cat([self.extractor(x[i:i + chunk_size])
                          for i in range(0, x.size(0), chunk_size)], dim=0)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, chunk_size: int = 64):
        """Memory-safe inference over a bag of ANY size.

        Training caps bags (``data.max_patches_per_bag``) because backprop must hold every
        patch's activations. At test time there are no gradients, so we can stream the bag
        through the extractor in chunks and still attention-pool over all N patches --
        i.e. the model sees the *whole* image at test time and an uncapped 336-patch
        outlier cannot OOM the job after training already succeeded.

        Returns ``(image_logits [1, C], patch_logits [N, C], attn [N, 1])``.
        """
        features = self._features_chunked(x, chunk_size)          # [N, F]
        patch_logits = self.classifier(self.dropout(features))     # [N, C]
        attn = F.softmax(self.attention(features), dim=0)          # [N, 1]
        pooled = torch.sum(attn * features, dim=0, keepdim=True)   # [1, F]
        image_logits = self.classifier(self.dropout(pooled))       # [1, C]
        return image_logits, patch_logits, attn

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """Return the per-patch softmax attention weights ``[N, 1]`` (for heatmaps)."""
        features = self._features_chunked(x, chunk_size)
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

    @torch.no_grad()
    def infer(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        """Per-patch logits ``[N, C]`` for a whole bag, streamed in chunks.

        This is what the majority-voting baseline runs on: Stage-1 classifies every patch
        of an image independently and the image label is the hard vote over its patches --
        exactly the aggregation rule that AMIL is being tested against.
        """
        if chunk_size is None or chunk_size <= 0 or x.size(0) <= chunk_size:
            return self.classifier(self.dropout(self.extractor(x)))
        return torch.cat(
            [self.classifier(self.dropout(self.extractor(x[i:i + chunk_size])))
             for i in range(0, x.size(0), chunk_size)],
            dim=0,
        )