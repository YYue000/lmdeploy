import torch
import intel_extension_for_pytorch

from ..moe import FusedMoEBuilder, FusedMoEImpl

class IPEXFusedMoEImpl(FusedMoEImpl):
    """triton fused moe implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        raise NotImplementedError
    
    
class IPEXFusedMoEBuilder(FusedMoEBuilder):
    """triton fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        return IPEXFusedMoEImpl(top_k=top_k,
                                  num_experts=num_experts,
                                  renormalize=renormalize)
    
    