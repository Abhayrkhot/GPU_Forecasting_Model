"""gpufc: forecast LLM serving performance on models and GPUs you have not profiled."""

from gpufc.cost import TERMS, ForecastStepModel, TermCalculator
from gpufc.specs import GPUS, GPUSpec, ModelSpec, get_gpu, get_model

__version__ = "0.1.0"

__all__ = [
    "GPUS",
    "TERMS",
    "ForecastStepModel",
    "GPUSpec",
    "ModelSpec",
    "TermCalculator",
    "__version__",
    "get_gpu",
    "get_model",
]
