import torch
import intel_extension_for_pytorch
# from intel_extension_for_pytorch.llm.functional import add_rms_norm, add_layer_norm
from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import add_rms_norm_cpu, add_layer_norm_cpu

from ..norm import LayerNormBuilder, LayerNormImpl, RMSNormBuilder, RMSNormImpl

class IPEXRMSNormImpl(RMSNormImpl):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        # return add_rms_norm(residual, x, weight, None, self.eps)
        return add_rms_norm_cpu(residual, x, weight, None, self.eps)
    
    
class IPEXRMSNormBuilder(RMSNormBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        return IPEXRMSNormImpl(hidden_size, eps)
    

class IPEXLayerNormImpl(LayerNormImpl):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = normalized_shape
        self.eps = eps
        
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor = None,
                bias: torch.Tensor = None,
                residual: torch.Tensor = None):
        # return add_layer_norm(residual, x, weight, bias, self.eps)
        return add_layer_norm_cpu(residual, x, weight, bias, self.eps)
    
    
class IPEXLayerNormBuilder(LayerNormBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(normalized_shape: int, eps: float = 1e-6):
        """build."""
        return IPEXLayerNormImpl(normalized_shape, eps)
    

