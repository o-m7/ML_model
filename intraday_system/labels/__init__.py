"""Label generation for classification tasks."""

from .triple_barrier import TripleBarrierLabeler
from .horizons import get_horizon_config

__all__ = ["TripleBarrierLabeler", "get_horizon_config"]

