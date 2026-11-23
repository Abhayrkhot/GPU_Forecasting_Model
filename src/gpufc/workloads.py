"""Workloads replayed on the GPUs: a profiling workload and trace-driven validation runs.

The *profile* workload is synthetic. It is sized from the model's and GPU's roofline so
that every (model, GPU) pair sees the same mix of batch shapes in a few minutes of
traffic: isolated prefills of many lengths, a decode ladder (``B`` identical requests at
once), Poisson mixtures with heavy-tailed lengths, and one saturated prefill burst.

*Validation* workloads are windows of the Azure LLM inference traces (conversation and
code), time-scaled to request rates around the pair's saturation point. Forecasts are
scored on these; nothing is ever fitted on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from batchsim.cost import BatchStats, StepTimeModel
from batchsim.metrics import SLO
from batchsim.request import RequestSpec
from batchsim.schedulers import SchedulerConfig
from batchsim.sweep import capacity
from batchsim.workload import LogNormal, Workload, synthetic

from gpufc.cost import ForecastStepModel
from gpufc.specs import GPUSpec, ModelSpec

MAX_MODEL_LEN = 8192
# A deliberately rough prior (about 60 % of datasheet peaks, 1.5 ms per step): used only
# to size workloads and as the "untrained" baseline, never fitted.
PRIOR_COEF: dict[str, float] = {
    "const": 1.5,
    "seqs": 0.5,
    "host_tokens": 0.05,
    "prefill_seqs": 0.5,
    "mixed_decode_seqs": 0.5,
    "piecewise_layers": 1.0,
    "eager_layers": 2.0,
    "gemm_max": 1.4,
    "gemm_min": 0.3,
    "gemm_sustained": 0.0,
    "moe_max": 2.0,
    "moe_min": 0.3,
    "attn_prefill": 2.0,
    "attn_decode": 1.4,
    "elementwise": 1.5,
}


def prior_model(model: ModelSpec, gpu: GPUSpec) -> ForecastStepModel:
    return ForecastStepModel(model, gpu, dict(PRIOR_COEF), meta={"prior": True})


@dataclass(frozen=True)
class ServeLimits:
    """Engine limits that shape batches (vLLM's ``--max-num-*`` and the KV pool)."""

    kv_tokens: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int = MAX_MODEL_LEN

    def scheduler_config(self, block_size: int = 16) -> SchedulerConfig:
        return SchedulerConfig(
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            num_blocks=self.kv_tokens // block_size,
            block_size=block_size,
            max_model_len=self.max_model_len,
        )


def vllm_default_limits(gpu: GPUSpec) -> tuple[int, int]:
    """vLLM 0.34 OpenAI-server defaults: (max_num_batched_tokens, max_num_seqs).

    GPUs with at least 70 GiB that are not A100s get 8192 / 1024, all others 2048 / 256.
    """
    big = gpu.memory_bytes >= 70 * (1 << 30) and not gpu.name.startswith("a100")
    return (8192, 1024) if big else (2048, 256)


def _prefill_time(cost: StepTimeModel, tokens: int) -> float:
    s = BatchStats()
    s.add(tokens, 0)
    return cost.step_time(s)


def _decode_time(cost: StepTimeModel, batch: int, ctx: int) -> float:
    s = BatchStats()
    for _ in range(batch):
        s.add(1, ctx)
    return cost.step_time(s)


def profile_workload(
    model: ModelSpec, gpu: GPUSpec, limits: ServeLimits, seed: int = 1234
) -> Workload:
    """Synthetic workload covering the batch shapes a serving engine sees.

    Phase lengths come from the roofline prior so a 0.5B model on an H200 and a 32B model
    on an A100 both get dense coverage within a few minutes.
    """
    cost = prior_model(model, gpu)
    rng = np.random.default_rng(seed)
    max_prompt = limits.max_model_len - 256
    budget = limits.max_num_batched_tokens
    specs: list[RequestSpec] = []
    t = 0.0

    # 1. Isolated prefills, log-uniform lengths (chunked beyond the token budget).
    lens = np.exp(rng.uniform(math.log(16), math.log(max_prompt), size=32)).astype(int)
    for p in lens:
        specs.append(RequestSpec(t, int(p), 4))
        chunks = math.ceil(p / budget)
        t += max(0.25, 3.0 * chunks * _prefill_time(cost, min(int(p), budget)))
    t += 2.0

    # 2. Decode ladder: B identical requests arrive together, then decode 128 tokens.
    out = 128
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        if b > limits.max_num_seqs:
            break
        for p in (256, 1024, 4096):
            if b * (p + out) > 0.6 * limits.kv_tokens or (b >= 128 and p == 4096):
                continue
            for _ in range(b):
                specs.append(RequestSpec(t, p, out))
            busy = math.ceil(b * p / budget) * _prefill_time(cost, min(b * p, budget))
            busy += out * _decode_time(cost, b, p + out // 2)
            t += 1.3 * busy + 0.5

    # 3. Poisson mixtures at three loads (fractions of the prior's saturation rate).
    p_dist = LogNormal(1000, 1.0, 16, max_prompt)
    o_dist = LogNormal(200, 1.0, 4, 1024)
    mean_p, mean_o = 1400.0, 280.0
    tokens_per_s = budget / _prefill_time(cost, budget)
    batch = min(limits.max_num_seqs, max(1, limits.kv_tokens // int(mean_p + mean_o)))
    decode_rate = batch / _decode_time(cost, batch, int(mean_p + mean_o / 2))
    sat = 1.0 / (mean_p / tokens_per_s + mean_o / decode_rate)
    for frac in (0.3, 0.6, 0.9):
        rate = frac * sat
        n = int(min(600, max(40, rate * 30)))
        wl = synthetic(n, rate, p_dist, o_dist, seed=int(rng.integers(1 << 30)))
        specs.extend(RequestSpec(t + s.arrival, s.prompt_len, s.output_len) for s in wl.specs)
        t += wl.specs[-1].arrival + 3.0 + mean_o * _decode_time(cost, batch, 1500)

    # 4. Saturated burst: many prompts at once fill the token budget every step.
    n_burst = int(min(300, max(30, 20 * limits.max_num_batched_tokens / 1500)))
    burst_lens = LogNormal(1500, 0.8, 16, max_prompt).sample(rng, n_burst)
    for p in burst_lens:
        specs.append(RequestSpec(t, int(p), 8))
    specs.sort(key=lambda s: s.arrival)
    return Workload(tuple(specs), "profile").clip(limits.max_model_len)


def smoke_workload() -> Workload:
    specs = [RequestSpec(0.3 * i, 64 + 200 * i, 8 + i) for i in range(8)]
    specs += [RequestSpec(3.0, 128 + 40 * i, 16) for i in range(16)]
    return Workload(tuple(specs), "smoke")


def saturation_rate(
    base: Workload,
    cost: StepTimeModel,
    limits: ServeLimits,
    ttft_slo: float = 2.0,
    attainment: float = 0.9,
    hi: float = 500.0,
) -> float:
    """Highest rate at which ``attainment`` of requests start within ``ttft_slo`` seconds.

    Used to place validation runs below, near and at the knee of the latency curve.
    """
    res = capacity(
        base,
        cost,
        SLO(ttft=ttft_slo),
        attainment=attainment,
        scheduler=limits.scheduler_config(),
        lo=0.05,
        hi=hi,
        rel_tol=0.03,
    )
    return float(res["rate"])


def validation_workload(
    base: Workload,
    kind: str,
    rate: float,
    seconds: float = 90.0,
    max_seconds: float = 240.0,
    max_requests: int = 1200,
) -> Workload:
    """About ``seconds`` of the trace at ``rate`` req/s: at least 60 requests unless that
    takes longer than ``max_seconds`` (bursty traces saturate at low mean rates)."""
    n = int(min(max_requests, max(rate * seconds, min(60.0, rate * max_seconds), 20.0), len(base)))
    wl = base.head(n).scale_to_rate(rate)
    return Workload(wl.specs, f"{kind}_r{rate:.3g}", {"trace": kind, "rate": rate})
