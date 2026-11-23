"""Synthetic measured data with a known ground truth.

Step times are generated from a fixed coefficient vector over the real term calculator
(with log-normal noise), KV-cache sizes from a fixed memory model, and client replays by
simulating the vLLM scheduler with the true step model plus a fixed frontend latency.
A forecaster that works must recover all of it on pairs it never saw.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from batchsim.engine import EngineConfig, simulate
from batchsim.workload import LogNormal, synthetic

from gpufc.cost import TERMS, ForecastStepModel
from gpufc.data import RunInfo, StepTable
from gpufc.memory import MemoryModel
from gpufc.specs import get_gpu, get_model
from gpufc.workloads import ServeLimits, profile_workload, vllm_default_limits

TRUE_COEF = {
    "const": 1.2,
    "seqs": 0.8,
    "host_tokens": 0.03,
    "prefill_seqs": 0.5,
    "mixed_decode_seqs": 0.7,
    "piecewise_layers": 3.0,
    "eager_layers": 6.0,
    "gemm_max": 1.3,
    "gemm_min": 0.4,
    "gemm_sustained": 0.15,
    "moe_max": 1.8,
    "moe_min": 0.2,
    "attn_prefill": 2.2,
    "attn_decode": 1.4,
    "elementwise": 1.5,
}
TRUE_MEMORY = MemoryModel({"const_gib": 2.0, "activation": 1.5, "logits": 1.0})
TRUE_FRONTEND = 0.004
MODELS = ("qwen2.5-0.5b", "qwen3-0.6b", "qwen2.5-1.5b", "qwen3-1.7b")
GPUS = ("h200", "a100-80g", "a6000")
NOISE = 0.03


def true_model(model: str, gpu: str) -> ForecastStepModel:
    return ForecastStepModel(get_model(model), get_gpu(gpu), dict(TRUE_COEF))


def make_pair(root: Path, model: str, gpu: str, seed: int, replays: bool = True) -> None:
    m, g = get_model(model), get_gpu(gpu)
    rng = np.random.default_rng(seed)
    bt, ns = vllm_default_limits(g)
    kv = TRUE_MEMORY.kv_tokens(m, g, bt, ns)
    limits = ServeLimits(kv, ns, bt, 4096)
    truth = true_model(model, gpu)
    eng = EngineConfig(True, TRUE_FRONTEND, 0.0, record_steps=True)

    runs: dict[str, list[tuple[float, int, int]]] = {}
    labels, stats, ys = [], [], []
    wl_prof = profile_workload(m, g, limits, seed=seed).head(150)
    workloads = {"profile": wl_prof}
    if replays:
        for i, rate in enumerate((0.5, 2.0)):
            workloads[f"conv_r{rate:g}"] = synthetic(
                120,
                rate,
                LogNormal(600, 0.8, 16, 3000),
                LogNormal(80, 0.8, 4, 300),
                seed=seed + i,
            )
    for name, wl in workloads.items():
        res = simulate(wl, truth, "vllm", limits.scheduler_config(), eng)
        # Step records: rebuild batch shapes from the simulator's step log is not possible,
        # so draw step shapes directly: decode batches and prefill chunks of the workload.
        for _ in range(300 if name == "profile" else 120):
            from batchsim.cost import BatchStats

            s = BatchStats()
            for _ in range(int(rng.integers(1, 64))):
                s.add(1, int(rng.integers(16, 3000)))
            if rng.random() < 0.4:
                s.add(int(rng.integers(16, bt)), int(rng.integers(0, 2000)))
            y = truth.step_time(s) * float(np.exp(rng.normal(0, NOISE)))
            labels.append(name)
            stats.append(s)
            ys.append(y)
        runs[name] = [
            (float(a), int(f), int(e))
            for a, f, e in zip(res.arrival, res.prompt_len, res.output_len)
        ]
        recs = []
        t0 = 1.7e9
        for i in range(len(res.arrival)):
            first = float(res.ttft[i])  # includes the frontend latency
            last = float(res.e2e[i])
            recs.append(
                {
                    "i": i,
                    "send": t0 + float(res.arrival[i]),
                    "prompt_len": int(res.prompt_len[i]),
                    "output_len": int(res.output_len[i]),
                    "ntok": int(res.output_len[i]),
                    "first": first,
                    "last": last,
                }
            )
        p = root / "replays" / g.name / m.name / f"{name}.jsonl.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    table = StepTable(m, g, np.array(labels), stats, np.array(ys))
    table.to_csv_gz(root / "steps" / g.name / f"{m.name}.csv.gz")
    info = RunInfo(
        m.name,
        g.name,
        {
            "model": m.name,
            "gpu_name": g.name,
            "max_num_batched_tokens": bt,
            "max_num_seqs": ns,
            "max_model_len": 4096,
        },
        {"kv_tokens": kv},
        {"gemm_flops": g.peak_flops() * 0.8, "copy_bandwidth": g.mem_bandwidth * 0.9},
        [],
        {k: (0.0, 1.0) for k in workloads},
    )
    p = root / "runs" / g.name / f"{m.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info.to_dict()))


@pytest.fixture(scope="session")
def synthetic_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("results")
    seed = 0
    for g in GPUS:
        for m in MODELS:
            make_pair(root, m, g, seed)
            seed += 1
    return root


@pytest.fixture
def all_terms() -> tuple[str, ...]:
    return TERMS
