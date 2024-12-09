import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import _IPEXRopeCPU
from torch import Tensor

from ..apply_rotary_emb import ApplyRotaryEmbBuilder, ApplyRotaryEmbImpl

class IPEXApplyRotaryEmbImpl(ApplyRotaryEmbImpl):
    """Apply rotary embedding implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True):
        if inplace:
            return _IPEXRopeCPU.rotary_embedding(query, key, sin, cos, sin.shape[-1], rotary_half=True, position_ids=None)
        raise NotImplementedError
    
class IPEXApplyRotaryEmbBuilder(ApplyRotaryEmbBuilder):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    def build():
        """build implementation."""
        return IPEXApplyRotaryEmbImpl()