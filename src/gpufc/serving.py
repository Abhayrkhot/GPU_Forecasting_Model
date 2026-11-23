"""From step times to serving metrics: TTFT/TPOT forecasts, capacity, and GPU planning.

A forecast for a (model, GPU) pair needs three things the user has not measured:

* step times: a :class:`~gpufc.cost.ForecastStepModel` (fitted on other pairs);
* the KV-cache size: :class:`~gpufc.memory.MemoryModel`;
* the latency outside the engine (HTTP, API server, detokenization): a constant fitted on
  other pairs' replays (:data:`Bundle.frontend_latency`).

They are fed to batchsim's vLLM-exact scheduler simulation, which replays a workload and
returns per-request TTFT, TPOT and end-to-end latency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from batchsim.engine import EngineConfig, simulate
from batchsim.metrics import SLO
from batchsim.sweep import capacity
from batchsim.workload import Workload

from gpufc.cost import ForecastStepModel
from gpufc.fit import Coef
from gpufc.memory import MemoryModel
from gpufc.specs import GPUS, GPUSpec, ModelSpec
from gpufc.workloads import MAX_MODEL_LEN, ServeLimits, vllm_default_limits


@dataclass
class Bundle:
    """Everything a forecast needs, fitted once on the measured pairs (``gpufc fit``).

    ``per_gpu`` holds weights fitted on all models of one profiled GPU (used when the
    target GPU is one of them); ``pooled`` and ``pooled_microbench`` hold weights fitted
    on every profiled GPU (used for any other GPU).
    """

    per_gpu: dict[str, Coef]
    pooled: Coef
    pooled_microbench: Coef | None
    memory: MemoryModel
    frontend_latency: float = 0.0
    idle_launch_latency: float = 0.0
    microbench: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def step_model(
        self, model: ModelSpec, gpu: GPUSpec, tier: str | None = None
    ) -> ForecastStepModel:
        """Step model for any (model, GPU); ``tier`` defaults to the best available."""
        if tier is None:
            tier = "seen_gpu" if gpu.name in self.per_gpu else "unseen_gpu"
            if tier == "unseen_gpu" and gpu.measured_flops and self.pooled_microbench:
                tier = "microbench"
        if tier == "seen_gpu":
            coef = self.per_gpu[gpu.name]
        elif tier == "microbench":
            if self.pooled_microbench is None or not gpu.measured_flops:
                raise ValueError("microbench tier needs measured peaks for the GPU")
            coef = self.pooled_microbench
        else:
            coef = self.pooled
        return coef.bind(model, gpu, tier=tier)

    def limits(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        max_model_len: int = MAX_MODEL_LEN,
    ) -> ServeLimits:
        bt, ns = vllm_default_limits(gpu)
        bt = max_num_batched_tokens or bt
        ns = max_num_seqs or ns
        return ServeLimits(self.memory.kv_tokens(model, gpu, bt, ns), ns, bt, max_model_len)

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            async_scheduling=True,
            frontend_latency=self.frontend_latency,
            idle_launch_latency=self.idle_launch_latency,
            record_steps=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_gpu": {k: v.to_dict() for k, v in self.per_gpu.items()},
            "pooled": self.pooled.to_dict(),
            "pooled_microbench": self.pooled_microbench.to_dict()
            if self.pooled_microbench
            else None,
            "memory": self.memory.to_dict(),
            "frontend_latency": self.frontend_latency,
            "idle_launch_latency": self.idle_launch_latency,
            "microbench": self.microbench,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Bundle:
        pm = d.get("pooled_microbench")
        return cls(
            {k: Coef.from_dict(v) for k, v in d["per_gpu"].items()},
            Coef.from_dict(d["pooled"]),
            Coef.from_dict(pm) if pm else None,
            MemoryModel.from_dict(d["memory"]),
            float(d.get("frontend_latency", 0.0)),
            float(d.get("idle_launch_latency", 0.0)),
            d.get("microbench", {}),
            d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Bundle:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def gpu(self, gpu: GPUSpec) -> GPUSpec:
        """``gpu`` with stored microbenchmark peaks, if the bundle has them."""
        mb = self.microbench.get(gpu.name)
        if mb and not gpu.measured_flops:
            return gpu.with_measurements(mb["gemm_flops"], mb["copy_bandwidth"])
        return gpu


def forecast(
    workload: Workload,
    step_model: ForecastStepModel,
    limits: ServeLimits,
    engine: EngineConfig,
) -> dict[str, float]:
    """Simulate ``workload`` and summarise TTFT/TPOT/E2E percentiles (seconds)."""
    res = simulate(workload, step_model, "vllm", limits.scheduler_config(), engine)
    span = workload.duration
    return res.summary(warmup=0.1 * span, cooldown=0.1 * span)


def max_rate(
    base: Workload,
    step_model: ForecastStepModel,
    limits: ServeLimits,
    engine: EngineConfig,
    slo: SLO,
    attainment: float = 0.9,
    hi: float = 1000.0,
) -> float:
    """Highest request rate at which ``attainment`` of requests meet ``slo``."""
    res = capacity(
        base,
        step_model,
        slo,
        attainment=attainment,
        scheduler=limits.scheduler_config(),
        engine=engine,
        lo=0.02,
        hi=hi,
        rel_tol=0.02,
    )
    return float(res["rate"])


@dataclass
class PlanRow:
    gpu: str
    fits: bool
    kv_tokens: int
    tier: str
    rate_per_gpu: float
    rate_low: float
    rate_high: float
    gpus_needed: int
    gpus_needed_worst: int
    cost_per_hour: float | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def plan(
    bundle: Bundle,
    model: ModelSpec,
    base: Workload,
    target_rate: float,
    slo: SLO,
    gpus: list[str] | None = None,
    prices: dict[str, float] | None = None,
    attainment: float = 0.9,
    interval: str = "90",
) -> list[PlanRow]:
    """For each GPU type: SLO capacity of one replica and the replicas ``target_rate`` needs.

    Replicas are independent single-GPU engines behind a load balancer that spreads
    requests evenly, so ``n`` replicas serve ``n`` times one replica's capacity. The low and
    high capacities come from the step model's conformal interval (all step times scaled
    to its upper and lower quantile).
    """
    rows = []
    for name in gpus or sorted(GPUS):
        gpu = bundle.gpu(GPUS[name])
        lim = bundle.limits(model, gpu)
        need_kv = int(np.max(base.prompt_lens)) + int(np.max(base.output_lens))
        if lim.kv_tokens < min(lim.max_model_len, need_kv):
            rows.append(PlanRow(name, False, lim.kv_tokens, "", 0, 0, 0, 0, 0, None))
            continue
        sm = bundle.step_model(model, gpu)
        eng = bundle.engine_config()
        cap = max_rate(base, sm, lim, eng, slo, attainment)
        try:
            lo_f, hi_f = sm.interval(interval)
        except KeyError:
            lo_f = hi_f = 1.0
        # Slower steps (upper residual quantile) give the pessimistic capacity.
        cap_low = max_rate(base, sm.scaled(hi_f), lim, eng, slo, attainment)
        cap_high = max_rate(base, sm.scaled(lo_f), lim, eng, slo, attainment)
        n = math.ceil(target_rate / cap) if cap > 0 else 0
        n_worst = math.ceil(target_rate / cap_low) if cap_low > 0 else 0
        price = (prices or {}).get(name)
        rows.append(
            PlanRow(
                name,
                True,
                lim.kv_tokens,
                str(sm.meta.get("tier", "")),
                cap,
                cap_low,
                cap_high,
                n,
                n_worst,
                price * n if price is not None and n else None,
            )
        )
    return rows
