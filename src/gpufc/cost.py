"""Physically structured step-time model that transfers across models and GPUs.

A vLLM engine step runs one forward pass over a batch: ``T`` scheduled tokens from ``S``
sequences, of which ``D`` decode one token each and ``P`` are prefill chunks. Its time is
a sum of GPU work, each part bound by tensor-core throughput or by memory bandwidth, and
host work (scheduling, input preparation, kernel launches) that no GPU spec predicts.
:class:`TermCalculator` turns a batch shape, a :class:`~gpufc.specs.ModelSpec` and a
:class:`~gpufc.specs.GPUSpec` into these terms.

GPU terms, in seconds at datasheet peak (their weights are inverse efficiencies):

``gemm_max`` / ``gemm_min``
    For every weight GEMM ``[M, k] @ [k, n]``, the larger and the smaller of its compute
    time ``2Mkn/F`` and its memory time ``(kn + M(k+n))b/BW``. ``M`` is the token count
    padded to the CUDA graph size, except for the vocabulary projection (``M`` = sequences,
    one logit row each, plus a write and a read of fp32 logits). ``a*max + c*min`` is a
    smooth roofline: weight-read bound at small ``M``, compute bound at large ``M``.
``gemm_sustained``
    ``c * min(1, c / 50 ms)`` with ``c`` the step's tensor-core time at peak (dense and
    expert GEMM FLOPs over peak FLOP/s): zero for decode steps, quadratic for short
    prefills, linear once a step's GEMMs run longer than 50 ms. Its weight is the extra
    time per second of sustained tensor-core load, which a power-limited GPU spends at
    reduced clocks (on the H200, long prefills sustain about 70 % of peak against 85 % for
    short ones).
``moe_max`` / ``moe_min``
    The same for routed experts, whose weight traffic grows with the number of distinct
    experts the batch touches (:meth:`~gpufc.specs.ModelSpec.expected_experts`).
``attn_prefill``
    Causal attention FLOPs of the prefill chunks (``4 * layers * q_dim`` per query-key
    pair).
``attn_decode``
    KV-cache bytes the decode queries read.
``elementwise``
    Norms, rotary embeddings, residuals and activations: ``10 * layers * tokens * hidden``
    bytes.

Host terms, in fixed units (their weights are times):

``const`` 1 ms per step; ``seqs`` 10 us per sequence; ``host_tokens`` 1 us per token;
``prefill_seqs`` 100 us per prefill chunk; ``mixed_decode_seqs`` 10 us per decode that
shares its step with a prefill chunk (it then runs outside the decode CUDA graph, through
the variable-length attention path); ``piecewise_layers`` / ``eager_layers`` 10 us per
layer for steps run with piecewise CUDA graphs (mixed batches) or none (more tokens than
the largest graph).

A :class:`ForecastStepModel` predicts ``sum_k w_k * term_k`` with non-negative weights.
Because the GPU terms already carry the model's shapes and the GPU's peaks, and the host
terms carry neither, one weight vector fitted on other models (or other GPUs) transfers
to a model or GPU it has never seen.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from batchsim.cost import VLLM_CAPTURE_SIZES, BatchStats, StepTimeModel
from numpy.typing import NDArray

from gpufc.specs import GPUSpec, ModelSpec

HOST_TERMS: tuple[str, ...] = (
    "const",
    "seqs",
    "host_tokens",
    "prefill_seqs",
    "mixed_decode_seqs",
    "piecewise_layers",
    "eager_layers",
)
HW_TERMS: tuple[str, ...] = (
    "gemm_max",
    "gemm_min",
    "gemm_sustained",
    "moe_max",
    "moe_min",
    "attn_prefill",
    "attn_decode",
    "elementwise",
)
TERMS: tuple[str, ...] = HOST_TERMS + HW_TERMS
ELEMENTWISE_PASSES = 10  # read+write of a hidden-sized activation, per layer, per token
# Tensor-core work in one step beyond about this long (at peak) runs into power-limited
# clocks; see ``gemm_sustained``. Chosen from {20, 50, 100} ms (held-out step error within
# 0.3 points of each other).
SUSTAINED_TAU_S = 0.05


def pad_tokens(n: int, sizes: Sequence[int]) -> int:
    """Round ``n`` up to the next CUDA graph capture size; beyond the largest, no padding."""
    if not sizes or n > sizes[-1]:
        return n
    return int(sizes[bisect.bisect_left(sizes, n)])


@dataclass(frozen=True)
class Peaks:
    flops: float
    bandwidth: float

    @classmethod
    def of(cls, model: ModelSpec, gpu: GPUSpec, measured: bool = False) -> Peaks:
        bf16 = model.dtype_bytes == 2 and gpu.supports_bf16
        return cls(gpu.peak_flops(model.dtype_bytes, bf16, measured), gpu.bandwidth(measured))


class TermCalculator:
    """Computes :data:`TERMS` for one (model, GPU) pair, caching per token/sequence count."""

    def __init__(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        capture_sizes: Sequence[int] = VLLM_CAPTURE_SIZES,
        measured_peaks: bool = False,
    ) -> None:
        self.model = model
        self.gpu = gpu
        self.capture_sizes = tuple(capture_sizes)
        self.max_capture = self.capture_sizes[-1] if self.capture_sizes else 0
        self.peaks = Peaks.of(model, gpu, measured_peaks)
        self._linears = [(lin.k, lin.n, lin.count) for lin in model.dense_linears()]
        self._tok_cache: dict[int, tuple[float, float, float, float, float, float]] = {}
        self._seq_cache: dict[int, tuple[float, float]] = {}
        m, p = model, self.peaks
        b = m.dtype_bytes
        self._attn_prefill_scale = 4.0 * m.num_layers * m.q_dim / p.flops
        self._attn_decode_scale = m.kv_bytes_per_token / p.bandwidth
        self._elem = ELEMENTWISE_PASSES * m.num_layers * m.hidden_size * b / p.bandwidth

    def _token_terms(self, t_pad: int) -> tuple[float, float, float, float, float, float]:
        """(gemm_max, gemm_min, gemm_sustained, moe_max, moe_min, elementwise)."""
        hit = self._tok_cache.get(t_pad)
        if hit is not None:
            return hit
        m, f, bw = self.model, self.peaks.flops, self.peaks.bandwidth
        b = m.dtype_bytes
        g_max = g_min = tensor = 0.0
        for k, n, count in self._linears:
            cmp = 2.0 * t_pad * k * n / f
            mem = (k * n + t_pad * (k + n)) * b / bw
            g_max += count * max(cmp, mem)
            g_min += count * min(cmp, mem)
            tensor += count * cmp
        e_max = e_min = 0.0
        if m.is_moe and t_pad > 0:
            topk, inter, h = m.experts_per_token, m.moe_intermediate_size, m.hidden_size
            cmp = 2.0 * t_pad * topk * 3 * h * inter / f
            touched = m.expected_experts(t_pad)
            mem = (touched * m.expert_params + t_pad * topk * (2 * h + 3 * inter)) * b / bw
            e_max = m.num_moe_layers * max(cmp, mem)
            e_min = m.num_moe_layers * min(cmp, mem)
            tensor += m.num_moe_layers * cmp
        sustained = tensor * min(1.0, tensor / SUSTAINED_TAU_S)
        out = (g_max, g_min, sustained, e_max, e_min, self._elem * t_pad)
        self._tok_cache[t_pad] = out
        return out

    def _lm_head(self, seqs: int) -> tuple[float, float]:
        """(max, min) of the vocabulary projection's compute and memory time."""
        hit = self._seq_cache.get(seqs)
        if hit is not None:
            return hit
        m, f, bw = self.model, self.peaks.flops, self.peaks.bandwidth
        h, v, b = m.hidden_size, m.vocab_size, m.dtype_bytes
        cmp = 2.0 * seqs * h * v / f
        mem = ((h * v + seqs * h) * b + 2 * seqs * v * 4) / bw  # weights, rows, fp32 logits
        out = (max(cmp, mem), min(cmp, mem))
        self._seq_cache[seqs] = out
        return out

    def terms(self, s: BatchStats) -> tuple[float, ...]:
        """Values of :data:`TERMS` (in that order) for one step."""
        t = s.num_tokens
        decode_only = s.num_prefill == 0
        t_pad = pad_tokens(t, self.capture_sizes)
        g_max, g_min, sustained, e_max, e_min, elem = self._token_terms(t_pad)
        lm_max, lm_min = self._lm_head(s.num_seqs)
        layers = self.model.num_layers * 1e-5
        graphed = t <= self.max_capture
        return (
            1e-3,
            s.num_seqs * 1e-5,
            t * 1e-6,
            s.num_prefill * 1e-4,
            0.0 if decode_only else s.num_decode * 1e-5,
            layers if graphed and not decode_only else 0.0,
            layers if not graphed else 0.0,
            g_max + lm_max,
            g_min + lm_min,
            sustained,
            e_max,
            e_min,
            self._attn_prefill_scale * s.prefill_attn,
            self._attn_decode_scale * s.decode_ctx,
            elem,
        )

    def matrix(self, stats: Sequence[BatchStats]) -> NDArray[np.float64]:
        return np.array([self.terms(s) for s in stats], dtype=np.float64).reshape(-1, len(TERMS))

    def roofline_time(self, s: BatchStats) -> float:
        """Pure datasheet roofline (every GPU term at peak, no host overhead)."""
        v = dict(zip(TERMS, self.terms(s)))
        return math.fsum(
            v[k]
            for k in (
                "gemm_max",
                "moe_max",
                "attn_prefill",
                "attn_decode",
                "elementwise",
            )
        )


@dataclass
class ForecastStepModel(StepTimeModel):
    """``step_time = sum_k coef[k] * term_k`` for one (model, GPU) pair.

    ``coef`` is fitted on *other* models and/or GPUs by :mod:`gpufc.fit`; this object binds
    it to the target's shapes and peaks. ``log_interval`` holds conformal quantiles of
    ``log(measured / predicted)`` from held-out data, giving a multiplicative prediction
    interval (see :meth:`interval`).
    """

    kind: ClassVar[str] = "gpufc"
    model: ModelSpec
    gpu: GPUSpec
    coef: dict[str, float]
    capture_sizes: tuple[int, ...] = VLLM_CAPTURE_SIZES
    measured_peaks: bool = False
    scale: float = 1.0
    log_interval: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [t for t in TERMS if t not in self.coef]
        if missing:
            raise ValueError(f"missing coefficients: {missing}")
        self._calc = TermCalculator(self.model, self.gpu, self.capture_sizes, self.measured_peaks)
        self._w = np.array([self.coef[t] for t in TERMS], dtype=np.float64)

    def step_time(self, s: BatchStats) -> float:
        x = self._calc.terms(s)
        return self.scale * math.fsum(w * v for w, v in zip(self._w, x))

    def predict(self, stats: Sequence[BatchStats]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self.scale * (self._calc.matrix(stats) @ self._w)
        return out

    def scaled(self, factor: float) -> ForecastStepModel:
        """The same model with every step time multiplied by ``factor`` (interval bands)."""
        return ForecastStepModel(
            self.model,
            self.gpu,
            dict(self.coef),
            self.capture_sizes,
            self.measured_peaks,
            self.scale * factor,
            dict(self.log_interval),
            dict(self.meta),
        )

    def interval(self, level: str = "90") -> tuple[float, float]:
        """Multiplicative (low, high) factors covering ``level`` % of held-out steps."""
        lo = self.log_interval.get(f"lo{level}")
        hi = self.log_interval.get(f"hi{level}")
        if lo is None or hi is None:
            raise KeyError(f"no {level}% interval stored; have {sorted(self.log_interval)}")
        return math.exp(lo), math.exp(hi)

    def breakdown(self, s: BatchStats) -> dict[str, float]:
        """Seconds each term contributes to one step (explains a forecast)."""
        x = self._calc.terms(s)
        return {t: self.scale * w * v for t, w, v in zip(TERMS, self._w, x) if w * v > 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "gpu": self.gpu.to_dict(),
            "coef": self.coef,
            "capture_sizes": list(self.capture_sizes),
            "measured_peaks": self.measured_peaks,
            "scale": self.scale,
            "log_interval": self.log_interval,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForecastStepModel:
        return cls(
            model=ModelSpec(**d["model"]),
            gpu=GPUSpec.from_dict(d["gpu"]),
            coef={k: float(v) for k, v in d["coef"].items()},
            capture_sizes=tuple(d.get("capture_sizes", VLLM_CAPTURE_SIZES)),
            measured_peaks=bool(d.get("measured_peaks", False)),
            scale=float(d.get("scale", 1.0)),
            log_interval={k: float(v) for k, v in d.get("log_interval", {}).items()},
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> ForecastStepModel:
        d = json.loads(Path(path).read_text())
        d.pop("kind", None)
        return cls.from_dict(d)
