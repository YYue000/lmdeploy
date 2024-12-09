import torch

@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def silu_mul(x, y, inplace=False):
    return torch.nn.functional.silu(x, inplace=inplace) * y


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def gelu_mul(x, y, approximate="none"):
    return torch.nn.functional.gelu(x, approximate=approximate) * y