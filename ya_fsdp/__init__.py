try:
    from ._api import FullStateDictConfig, MixedPrecisionPolicy, StateDictType
    from .fully_shard import fully_shard
except Exception as e:
    import logging

    logger = logging.getLogger("ya_fsdp")
    logger.warning(f"Failed to import YaFSDP2: {e}")
    FullStateDictConfig = MixedPrecisionPolicy = StateDictType = fully_shard = None
from .ya_fsdp import YaFSDP

__all__ = [
    "YaFSDP",
    "fully_shard",
    "FullStateDictConfig",
    "MixedPrecisionPolicy",
    "StateDictType",
]
