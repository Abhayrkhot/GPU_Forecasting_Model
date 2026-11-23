"""How many KV-cache tokens vLLM will have: the other half of a capacity forecast.

vLLM sizes its KV cache after a profiling forward pass: whatever is left of
``gpu_memory_utilization`` of device memory after the weights, the peak activation
memory of a ``max_num_batched_tokens`` forward pass, the sampler's logits for
``max_num_seqs`` sequences, and non-torch allocations (CUDA context, graphs). The weights
are known exactly from the model's shapes; the rest is modelled as

``overhead = a0 + a1 * act_bytes + a2 * logit_bytes``

with ``act_bytes = max_num_batched_tokens * (hidden + 2 * mlp_width) * dtype`` (the widest
per-layer activation, the fused gate/up output) and ``logit_bytes = max_num_seqs * vocab
* 4``. The weights ``a`` are fitted (non-negative) on the KV sizes vLLM logged for other
models, and the fit is checked on models it never saw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from batchsim.calibrate import nnls

from gpufc.specs import GIB, GPUSpec, ModelSpec

FEATURES = ("const_gib", "activation", "logits")


def mlp_width(model: ModelSpec) -> int:
    if model.is_moe:
        return model.experts_per_token * model.moe_intermediate_size
    return model.intermediate_size


def features(model: ModelSpec, max_num_batched_tokens: int, max_num_seqs: int) -> list[float]:
    act = max_num_batched_tokens * (model.hidden_size + 2 * mlp_width(model)) * model.dtype_bytes
    logits = max_num_seqs * model.vocab_size * 4
    return [1.0, act / GIB, logits / GIB]


@dataclass
class MemoryModel:
    coef: dict[str, float] = field(
        default_factory=lambda: {"const_gib": 1.5, "activation": 2.0, "logits": 2.0}
    )
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    report: dict[str, Any] = field(default_factory=dict)

    def overhead_gib(self, model: ModelSpec, bt: int, ns: int) -> float:
        x = features(model, bt, ns)
        return float(sum(self.coef[k] * v for k, v in zip(FEATURES, x)))

    def kv_tokens(self, model: ModelSpec, gpu: GPUSpec, bt: int, ns: int) -> int:
        """Predicted ``GPU KV cache size`` in tokens (0 if the model does not fit)."""
        budget = gpu.memory_bytes * self.gpu_memory_utilization
        free = budget - model.weight_bytes - self.overhead_gib(model, bt, ns) * GIB
        if free <= 0:
            return 0
        blocks = int(free // (self.block_size * model.kv_bytes_per_token))
        return blocks * self.block_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": self.coef,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "block_size": self.block_size,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryModel:
        return cls(
            {k: float(v) for k, v in d["coef"].items()},
            float(d.get("gpu_memory_utilization", 0.9)),
            int(d.get("block_size", 16)),
            d.get("report", {}),
        )


@dataclass(frozen=True)
class MemorySample:
    model: ModelSpec
    gpu: GPUSpec
    max_num_batched_tokens: int
    max_num_seqs: int
    kv_tokens: int


def observed_overhead_gib(s: MemorySample, util: float = 0.9) -> float:
    """What the logged KV size implies the non-weight, non-KV memory was."""
    kv = s.kv_tokens * s.model.kv_bytes_per_token
    return float((s.gpu.memory_bytes * util - s.model.weight_bytes - kv) / GIB)


def fit_memory(samples: Sequence[MemorySample], util: float = 0.9) -> MemoryModel:
    x = np.array([features(s.model, s.max_num_batched_tokens, s.max_num_seqs) for s in samples])
    y = np.array([observed_overhead_gib(s, util) for s in samples])
    coef = nnls(x, y)
    pred = x @ coef
    return MemoryModel(
        dict(zip(FEATURES, (float(c) for c in coef))),
        util,
        report={"samples": len(y), "train_mae_gib": float(np.abs(pred - y).mean())},
    )
