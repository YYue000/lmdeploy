from dataclasses import dataclass

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.llm.modules import PagedAttention
# from intel_extension_for_pytorch.llm.functional import varlen_attention # wrap of torch.nn.functional.scaled_dot_product_attention

from ..attention import AttentionBuilder, AttentionImpl
from lmdeploy.pytorch.kernels.ipex.alibi import get_alibi_slope

@dataclass
class IPEXAttentionMetadata:
    is_decoding: bool
    block_offsets: torch.Tensor # [num_seqs, max_num_blocks_per_seq]
    q_seqlens: torch.Tensor # [num_seqs], from model_inputs
    kv_seqlens: torch.Tensor
    
class IPEXAttentionImpl(AttentionImpl[IPEXAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            **kwargs,
        )
        assert sliding_window is None, "sliding window attention is not supported"
        
        self.alibi_head_offset = self.num_heads * rank
        self.alibi_num_heads = self.num_heads * world_size
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: IPEXAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        inplace: bool = True,
        # k_scale: float = 1.0,
        # v_scale: float = 1.0
    ) -> torch.Tensor:
        """forward."""
        # k_cache: [num_blocks, num_kv_heads, block_size, head_size]
        num_blocks, num_kv_heads, block_size, head_size = k_cache.shape
        
        # fill kv cache
        if key is not None and value is not None:
            num_seq = attn_metadata.block_offsets.shape[0]
            # block_offsets should in [num_seq, max_blocks]
            # key should in [num_tokens, head_num, head_size]
            if attn_metadata.is_decoding:
                # q_seqlens are all ones
                idx = attn_metadata.kv_seqlens-1
                slot_mapping = attn_metadata.block_offsets[torch.arange(num_seq), idx//block_size].flatten().cpu()
                slot_mapping = slot_mapping*block_size + idx%block_size
            else:
                raise NotImplementedError
            
            PagedAttention.reshape_and_cache(key, value, k_cache, v_cache, slot_mapping)
            
        # TODO: consider chunked prefill
        
        o_shape = query.shape[:-1] + (self.v_head_size, )
        attn_output = query.new_empty(o_shape)
        if self.alibi:
            alibi_slopes = get_alibi_slope(self.alibi_num_heads, self.alibi_head_offset, self.num_heads)
        else:
            alibi_slopes = None
        if attn_metadata.is_decoding:
            num_q_heads = self.num_heads
            num_queries_per_kv = num_q_heads//num_kv_heads
            head_mapping = torch.repeat_interleave(
                torch.arange(num_kv_heads, dtype=torch.int32, device="cpu"),
                num_queries_per_kv,
            )
            
            PagedAttention.single_query_cached_kv_attention(
                            attn_output,
                            query,
                            k_cache,
                            v_cache,
                            head_mapping,
                            self.scale,
                            attn_metadata.block_offsets.int(),
                            attn_metadata.kv_seqlens.int(),
                            block_size,
                            attn_metadata.kv_seqlens.max().int().item(),
                            alibi_slopes
                            )
        else:
            raise NotImplementedError
            
        return attn_output
    
class IPEXAttentionBuilder(AttentionBuilder[IPEXAttentionMetadata]):
    """IPEX attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ) -> IPEXAttentionImpl:
        """build."""
        return IPEXAttentionImpl(num_heads,
                                   head_size,
                                   scale=scale,
                                   num_kv_heads=num_kv_heads,
                                   v_head_size=v_head_size,
                                   alibi=alibi,
                                   sliding_window=sliding_window,
                                   **kwargs)
