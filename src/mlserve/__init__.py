"""Serving a scikit-learn model over HTTP, with the parts production needs."""

from .registry import ModelRegistry, ModelBundle
from .schemas import BatchRequest, PredictRequest, PredictResponse

__all__ = ["BatchRequest", "ModelBundle", "ModelRegistry", "PredictRequest", "PredictResponse"]
__version__ = "1.0.0"
