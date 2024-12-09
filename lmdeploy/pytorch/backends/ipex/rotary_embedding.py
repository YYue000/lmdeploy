from dataclasses import asdict

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch.transformers.models.reference.fusion.mha_fusion import RotaryEmbedding

from ..rotary_embedding import (Llama3Parameters, LongRoPEScalingParameters,
                                RopeType, RotaryEmbeddingBuilder,
                                RotaryEmbeddingImpl, YarnParameters)

class IPEXRotaryEmbeddingImpl(RotaryEmbeddingImpl, RotaryEmbedding):
    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 max_position_embeddings: int = 2048,
                 backbone: "str" = None,
                 kwargs=None):
        
        RotaryEmbedding.__init__(max_position_embeddings, dim, backbone, base, kwargs)
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """forward."""
        return RotaryEmbedding.forward()
    
class DefaultRotaryEmbeddingBuilder(RotaryEmbeddingBuilder):
    """rotary embedding builder."""

    @staticmethod
    def build(
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0,
        yarn_params: YarnParameters = None,
        longrope_params: LongRoPEScalingParameters = None,
        llama3_params: Llama3Parameters = None,
        emb_type: RopeType = RopeType.Default,
    ):
        """build."""
        if emb_type in (RopeType.Default, RopeType.LinearScaling):
            return IPEXRotaryEmbeddingImpl(dim, base, scaling_factor)
        elif emb_type == RopeType.DynamicNTKScaling:
            return LlamaDynamicNTKScalingRotaryEmbedding(
                dim, base, scaling_factor, max_position_embeddings)
        elif emb_type == RopeType.Llama3:
            kwargs = asdict(llama3_params)
            kwargs["factor"] = scaling_factor
            kwargs["rope_type"] = "llama3"
            return IPEXRotaryEmbeddingImpl(
                dim, base, max_position_embeddings, , kwargs)
        elif emb_type == RopeType.Yarn:
            return YarnRotaryEmbeddingImpl(dim,
                                           base,
                                           scaling_factor,
                                           max_position_embeddings,
                                           yarn_params=yarn_params)
        elif emb_type == RopeType.LongRoPEScaling:
            return LongRoPEScalingRotaryEmbeddingImpl(
                dim,
                base,
                max_position_embeddings=max_position_embeddings,
                longrope_params=longrope_params,
            )
        else:
            raise NotImplementedError(
                f'Unsupported embedding type: {emb_type}')
