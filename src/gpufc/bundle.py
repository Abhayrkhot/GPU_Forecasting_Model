"""Fit the shipped forecasting bundle on every measured (model, GPU) pair."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpufc.data import Dataset
from gpufc.evaluate import frontend_table, memory_samples
from gpufc.fit import fit_gpu, fit_tier
from gpufc.memory import fit_memory
from gpufc.serving import Bundle


def fit_bundle(results: Path, intervals: bool = True) -> Bundle:
    ds = Dataset.load(results)
    tables = list(ds)
    if not tables:
        raise SystemExit(f"no measured pairs under {results}")
    per_gpu = {}
    for g in ds.gpus:
        on_gpu = [t for t in tables if t.gpu.name == g]
        if len(on_gpu) >= 2:
            per_gpu[g] = fit_gpu(on_gpu, tables, intervals=intervals)
    pooled = fit_tier(tables, intervals=intervals)
    pooled_mb = fit_tier(tables, measured_peaks=True, intervals=intervals)
    memory = fit_memory(memory_samples(ds))
    fe = frontend_table(ds)
    microbench: dict[str, dict[str, float]] = {}
    for g in ds.gpus:
        mbs = [ds.infos[k].microbench for k in ds.pairs(gpu=g) if ds.infos[k].microbench]
        if mbs:
            microbench[g] = {
                "gemm_flops": float(np.median([m["gemm_flops"] for m in mbs])),
                "copy_bandwidth": float(np.median([m["copy_bandwidth"] for m in mbs])),
            }
    return Bundle(
        per_gpu=per_gpu,
        pooled=pooled,
        pooled_microbench=pooled_mb,
        memory=memory,
        frontend_latency=float(np.median([v["frontend_latency"] for v in fe.values()]))
        if fe
        else 0.0,
        idle_launch_latency=float(np.median([v["idle_launch_latency"] for v in fe.values()]))
        if fe
        else 0.0,
        microbench=microbench,
        meta={"pairs": [list(k) for k in sorted(ds.tables)], "gpus": ds.gpus, "models": ds.models},
    )
