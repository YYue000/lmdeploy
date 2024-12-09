from ..activation import (GeluAndMulBuilder, GeluAndMulImpl, SiluAndMulBuilder,
                          SiluAndMulImpl)
from lmdeploy.pytorch.kernels.ipex.activation import silu_mul, gelu_mul

import torch

class IPEXSiluAndMulImpl(SiluAndMulImpl):
    def __init__(self, inplace: bool):
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor):
        gate, up = x.chunk(2, -1)
        return silu_mul(gate, up, inplace=self.inplace)
    
class IPEXGeluAndMulImpl(GeluAndMulImpl):
    def __init__(self, inplace: bool):
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor):
        gate, up = x.chunk(2, -1)
        return gelu_mul(gate, up, inplace=self.inplace)

class IPEXSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return IPEXSiluAndMulImpl(inplace)
    
class IPEXGeluAndMulBuilder(GeluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(approximate: str = 'none'):
        """build."""
        return IPEXGeluAndMulImpl(approximate)