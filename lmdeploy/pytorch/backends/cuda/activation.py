# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

from ..activation import SiluAndMulBuilder, SiluAndMulImpl


class TritonSiluAndMulImpl(SiluAndMulImpl):
    """silu + multiple residual fused implementation."""

    def __init__(self, inplace: bool):
        self.inplace = inplace

    def _forward_naive(self, x):
        """forward naive."""
        gate, up = x.chunk(2, -1)
        return nn.functional.silu(gate, self.inplace) * up

    def forward(self, x):
        """forward."""

        if x.size(-1) % 2048 != 0:
            return self._forward_naive(x)

        out = None
        x_shape = None
        if x.dim() != 2:
            x_shape = x.shape
            x = x.flatten(0, -2)
        if self.inplace:
            out = x.chunk(2, -1)[0]

        out = silu_and_mul(x, out)

        if x_shape is not None:
            out = out.unflatten(0, x_shape[:-1])
        return out


class TritonSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return TritonSiluAndMulImpl(inplace)