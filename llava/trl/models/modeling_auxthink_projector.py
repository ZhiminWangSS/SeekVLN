"""AuxThink Base projector compatibility without modifying vendored NaVILA code."""

import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from llava.model.multimodal_projector.base_projector import MultimodalProjector


class AuxThinkDownSample3x3FixBlock(nn.Module):
    """Spatially group a square feature map into fixed 3x3 neighborhoods."""

    def forward(self, features):
        batch, tokens, channels = features.shape
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError(f"AuxThink projector requires square visual tokens, got {tokens}")
        features = features.reshape(batch, side, side, channels)
        pad = (3 - side % 3) % 3
        if pad:
            features = F.pad(features, (0, 0, 0, pad, 0, pad))
        side += pad
        features = features.reshape(batch, side, side // 3, channels * 3)
        features = features.permute(0, 2, 1, 3).contiguous()
        return features.reshape(batch, (side // 3) ** 2, channels * 9)


class AuxThinkMultimodalProjector(MultimodalProjector):
    """Load AuxThink's released 3x3 projector through NaVILA's builder API."""

    def __init__(self, projector_config, model_config):
        if projector_config.mm_projector_type != "mlp_downsample_3x3_fix":
            super().__init__(projector_config, model_config)
            return
        PreTrainedModel.__init__(self, projector_config)
        self.layers = nn.Sequential(
            AuxThinkDownSample3x3FixBlock(),
            nn.LayerNorm(model_config.mm_hidden_size * 9),
            nn.Linear(model_config.mm_hidden_size * 9, 3456),
            nn.GELU(),
            nn.LayerNorm(3456),
            nn.Linear(3456, model_config.hidden_size),
            nn.GELU(),
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
        )


def load_auxthink_base(load_pretrained_model, model_path, model_name, **kwargs):
    """Temporarily register the released projector while constructing AuxThink Base."""

    from llava.model.multimodal_projector import builder

    original = builder.MultimodalProjector
    builder.MultimodalProjector = AuxThinkMultimodalProjector
    try:
        # Forward loader controls such as ``device_map`` and ``device``.  The
        # VERL rollout server uses an explicit per-replica map; dropping these
        # arguments here would silently fall back to Transformers' ``auto``
        # placement and can leave meta tensors in the multimodal composite.
        return load_pretrained_model(model_path, model_name, **kwargs)
    finally:
        builder.MultimodalProjector = original
