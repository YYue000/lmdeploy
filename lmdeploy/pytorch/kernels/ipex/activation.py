import torch

@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def silu_mul(x: torch.Tensor, inplace: bool=True):
    gate, up = x.chunk(2, -1)
    return torch.nn.functional.silu(gate, inplace=inplace) * up


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def gelu_mul(x: torch.Tensor, approximate: str = "none"):
    gate, up = x.chunk(2, -1)
    return torch.nn.functional.gelu(gate, approximate=approximate) * up