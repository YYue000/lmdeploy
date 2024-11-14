# Copyright (c) OpenMMLab. All rights reserved.
from ...config import CacheConfig
from .base_block_manager import BaseBlockManager
from .default_block_manager import DefaultBlockManager
from .window_block_manager import WindowBlockManager
from .cpu_block_manager import CPUBlockManager

def build_block_manager(cache_config: CacheConfig) -> BaseBlockManager:
    """build block manager.

    Args:
        cache_config (CacheConfig):  cache_config.
    """
    if cache_config.host == "cpu":
        return CPUBlockManager()
    
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_gpu_blocks = cache_config.num_gpu_blocks
    window_size = cache_config.window_size

    if window_size < 0:
        return DefaultBlockManager(num_gpu_blocks, num_cpu_blocks)
    else:
        return WindowBlockManager(num_gpu_blocks,
                                  num_cpu_blocks,
                                  window_size=window_size)
