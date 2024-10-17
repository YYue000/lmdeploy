from ..default import DefaultOpsBackend
from ..base import OpType

class CPUOpsBackend(DefaultOpsBackend):
    @staticmethod
    def get_name() -> str:
        return 'cpu'
    
    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get builder of given layer type."""
        if layer_type == OpType.Attention:
            from .attention import 
        else:
            return DefaultOpsBackend.get_layer_impl_builder(cls, layer_type)