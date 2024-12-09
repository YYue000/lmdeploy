from dataclasses import dataclass

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch.llm.modules.PagedAttention import single_query_cached_kv_attention, reshape_and_cache
# from intel_extension_for_pytorch.llm.functional import varlen_attention # wrap of torch.nn.functional.scaled_dot_product_attention

from ..attention import AttentionBuilder, AttentionImpl, AttentionMetadata
from lmdeploy.pytorch.kernels.ipex.alibi import get_alibi_slope

@dataclass
class IPEXAttentionMetadata(AttentionMetadata):
    is_decoding: bool
    block_offsets: torch.Tensor # [num_seqs, max_num_blocks_per_seq]
    q_seqlens: torch.Tensor # [num_seqs], from model_inputs
    kv_seqlens: torch.Tensor
    attention_mask: torch.Tensor
    
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
        k_scale: float = 1.0,
        v_scale: float = 1.0
    ) -> torch.Tensor:
        """forward."""
        # assert False, "debug info, check shape"
        # k_cache: [num_blocks, num_kv_heads, block_size, head_size]
        num_blocks, num_kv_heads, block_size, head_size = k_cache.shape
        assert num_kv_heads==self.num_kv_heads and head_size==self.head_size
        
        history_seqlens = attn_metadata.kv_seqlens - attn_metadata.q_seqlens
        
        # fill kv cache
        if key is not None and value is not None:
            num_tokens = attn_metadata.block_offsets.numel()
            # key should in [num_tokens, head_num, head_size]
            if attn_metadata.is_decoding:
                # q_seqlens are all ones
                slot_mapping = attn_metadata.block_offsets[:, attn_metadata.kv_seqlens-1].flatten()
            else:
                # kv_seqlens == q_seqlens
                # block_offsets should in [1, num_tokens]
                slot_mapping = attn_metadata.block_offsets.flatten()
            
            idx = torch.arange(num_tokens)
            slot_mapping = slot_mapping[idx//block_size] + idx%block_size
            reshape_and_cache(key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale)
        # TODO: consider chunked prefill
        
        if self.alibi:
            alibi_slopes = get_alibi_slope(self.num_heads)
        else:
            alibi_slopes = None
        if attn_metadata.is_decoding:
            num_q_heads = self.num_heads
            num_queries_per_kv = num_q_heads//num_kv_heads
            head_mapping = torch.repeat_interleave(
                torch.arange(num_kv_heads, dtype=torch.int32, device="cpu"),
                num_queries_per_kv,
            )
            
            max_context_len = history_seqlens.max().item()
            
            o_shape = query.shape[:-1] + (self.v_head_size, )
            attn_output = query.new_empty(o_shape)
            
            single_query_cached_kv_attention(
                            attn_output,
                            query,
                            k_cache,
                            v_cache,
                            head_mapping,
                            self.scale,
                            attn_metadata.block_offsets,
                            history_seqlens,
                            block_size,
                            max_context_len,
                            alibi_slopes,
                            k_scale,
                            v_scale
                            )
        else:
            attn_output, _ = torch.ops.torch_ipex.flash_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attention_mask=attn_metadata.attention_mask,
                scale=self.scale
            )

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
