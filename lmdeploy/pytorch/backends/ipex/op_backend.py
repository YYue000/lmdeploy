from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')

class IPEXOpsBackend(DefaultOpsBackend):
    """ipex layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'IPEX'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        if layer_type == OpType.Attention:
            from .attention import IPEXAttentionBuilder
            return IPEXAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import IPEXApplyRotaryEmbBuilder
            return IPEXApplyRotaryEmbBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import IPEXRMSNormBuilder
            return IPEXRMSNormBuilder
        elif layer_type == OpType.SiluAndMul:
            from .activation import IPEXSiluAndMulBuilder
            return IPEXSiluAndMulBuilder
        elif layer_type == OpType.GeluAndMul:
            from .activation import IPEXGeluAndMulBuilder
            return IPEXGeluAndMulBuilder
        elif layer_type == OpType.FusedMoE:
            from .moe import IPEXFusedMoEBuilder
            return IPEXFusedMoEBuilder
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        """get attention metadata class."""
        from .attention import IPEXAttentionMetadata
        return IPEXAttentionMetadata
    
    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get k block shape."""
        return (
            num_heads,
            block_size,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get v block shape."""
        return (
            num_heads,
            block_size,
            head_size,
        )
        
    @classmethod
    def update_step_context(cls, step_context):
        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            is_decoding=step_context.is_decoding,
            block_offsets=step_context.block_offsets.cpu(),
            q_seqlens=step_context.q_seqlens.cpu(),
            kv_seqlens=step_context.kv_seqlens.cpu()
        )
        step_context.attn_metadata = attn_metadata
        return step_context
    
    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from .graph_runner import IPEXCPUGraphRunner
        return IPEXCPUGraphRunner(model, model_config, cache_config,
                               backend_config, device)
        