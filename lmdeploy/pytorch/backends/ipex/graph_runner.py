# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from importlib import import_module

import torch
import torch.distributed

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner

logger = get_logger('lmdeploy')


class IPEXCPUGraphRunner(GraphRunner):
    """ipex cpu graph runner."""
    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config,
                         device)
        
        """
        self.model = torch.compile(self.model,
                                       fullgraph=True,
                                       dynamic=True,
                                       backend='inductor')
        self.model = self._convert_op()
        """
    
    def _convert_op(self):
        return self.model
        raise NotImplementedError