"""Held-out evaluation: forecast pairs the forecaster never saw, against baselines.

Protocols
---------
``lomo`` (leave one model out)
    For each profiled GPU and each model on it, fit on that GPU's *other* models and
    forecast the held-out model: "a new model on hardware we have".
``logo`` (leave one GPU out, strict)
    For each GPU, fit on the other GPUs' pairs *excluding the target model too*, and
    forecast every model on the held-out GPU from its datasheet (``unseen_gpu``) or from
    a one-minute microbenchmark (``microbench``): "new hardware, new model".

Step-level scores use the steps recorded while the server replayed the Azure traces (real
traffic shapes), which no method ever trains on. End-to-end scores replay the same traces
through batchsim with each method's step model and compare TTFT/TPOT/E2E percentiles with
what the client measured.

Baselines
---------
``roofline``     datasheet roofline with fixed efficiencies (:data:`~gpufc.workloads.PRIOR_COEF`)
``gbm``          gradient-boosted trees on raw batch, model and GPU features (black box),
                 trained on exactly the pairs the gpufc fit uses
``oracle``       batchsim's linear model calibrated on the target pair's own profile run:
                 the accuracy you get by profiling the target, an upper bound
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from batchsim.calibrate import fit_linear
from batchsim.cost import BatchStats, StepTimeModel
from batchsim.engine import EngineConfig, simulate
from batchsim.metrics import SLO
from batchsim.validation import Measured, compare, load_replay
from numpy.typing import NDArray

from gpufc.cost import ForecastStepModel
from gpufc.data import Dataset, RunInfo, StepTable
from gpufc.fit import Coef, fit_for_target, predict
from gpufc.memory import MemoryModel, MemorySample, fit_memory
from gpufc.workloads import PRIOR_COEF, ServeLimits

FloatArray = NDArray[np.float64]
ENGINE_EXPLAINED = 0.5
E2E_METRICS = (
    "ttft_p50",
    "ttft_p90",
    "ttft_p99",
    "tpot_p50",
    "tpot_p90",
    "tpot_p99",
    "e2e_p50",
    "e2e_p90",
    "e2e_p99",
)


def step_scores(pred: FloatArray, y: FloatArray) -> dict[str, float]:
    rel = (pred - y) / y
    ape = np.abs(rel)
    return {
        "steps": float(len(y)),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.9)),
        "time_weighted_ape": float(np.abs(pred - y).sum() / y.sum()),
        "total_time_err": float((pred.sum() - y.sum()) / y.sum()),
        "bias": float(np.median(rel)),
    }


def coverage(pred: FloatArray, y: FloatArray, log_interval: dict[str, float]) -> dict[str, float]:
    out = {}
    r = np.log(y / pred)
    for level in ("80", "90", "95"):
        lo, hi = log_interval.get(f"lo{level}"), log_interval.get(f"hi{level}")
        if lo is not None and hi is not None:
            out[f"cov{level}"] = float(((r >= lo) & (r <= hi)).mean())
    return out


# ---- black-box baseline -----------------------------------------------------------------


def raw_features(table: StepTable) -> FloatArray:
    m, g = table.model, table.gpu
    static = [
        m.num_layers,
        m.hidden_size,
        m.num_heads,
        m.num_kv_heads,
        m.head_dim,
        m.intermediate_size,
        m.vocab_size,
        m.num_params / 1e9,
        m.active_params / 1e9,
        m.kv_bytes_per_token,
        m.num_experts,
        m.experts_per_token,
        m.moe_intermediate_size,
        g.peak_flops() / 1e12,
        g.mem_bandwidth / 1e12,
        g.memory_bytes / 2**30,
        g.sm_count,
        g.l2_bytes / 2**20,
    ]
    rows = [
        [
            s.num_tokens,
            s.num_seqs,
            s.num_decode,
            s.decode_ctx,
            s.num_prefill,
            s.prefill_tokens,
            s.prefill_attn,
            *static,
        ]
        for s in table.stats
    ]
    return np.array(rows, dtype=np.float64)


def fit_gbm(tables: Sequence[StepTable], max_steps: int = 6000, seed: int = 0) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    xs, ys = [], []
    for t in tables:
        t = t.subsample(max_steps, seed)
        xs.append(raw_features(t))
        ys.append(np.log(t.y))
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
    gbm.fit(np.vstack(xs), np.concatenate(ys))
    return gbm


def predict_gbm(gbm: Any, table: StepTable) -> FloatArray:
    out: FloatArray = np.exp(gbm.predict(raw_features(table)))
    return out


# ---- folds ------------------------------------------------------------------------------


@dataclass
class Fold:
    """What one held-out target is forecast with."""

    tier: str
    model: str
    gpu: str
    coef: Coef
    train: list[StepTable]


def folds(ds: Dataset, tier: str, intervals: bool = True) -> list[Fold]:
    tables = list(ds)
    out = []
    for (m, g), _ in sorted(ds.tables.items()):
        fitted = fit_for_target(tables, m, g, tier, intervals=intervals)
        if fitted is None:
            continue
        coef, train = fitted
        out.append(Fold(tier, m, g, coef, train))
    return out


def evaluate_steps(
    ds: Dataset,
    fold_list: Sequence[Fold],
    with_gbm: bool = True,
    max_eval_steps: int = 20000,
    keep_sample: int = 0,
) -> list[dict[str, Any]]:
    """Per held-out pair: step-time scores of gpufc and the baselines on validation steps.

    ``keep_sample`` > 0 stores that many (measured, predicted) step times per pair for
    parity plots.
    """
    rows = []
    gbm_cache: dict[tuple[tuple[str, str], ...], Any] = {}
    for f in fold_list:
        target = ds.tables[(f.model, f.gpu)]
        test = target.validation().subsample(max_eval_steps)
        if len(test) == 0:
            continue
        pred = predict(f.coef, test)
        row: dict[str, Any] = {
            "tier": f.tier,
            "model": f.model,
            "gpu": f.gpu,
            "train_pairs": len(f.train),
        }
        row["gpufc"] = {**step_scores(pred, test.y), **coverage(pred, test.y, f.coef.log_interval)}
        prior = Coef(dict(PRIOR_COEF), [])
        row["roofline"] = step_scores(predict(prior, test), test.y)
        if with_gbm:
            key = tuple(sorted(t.key for t in f.train))
            if key not in gbm_cache:
                gbm_cache[key] = fit_gbm(f.train)
            row["gbm"] = step_scores(predict_gbm(gbm_cache[key], test), test.y)
        prof = target.profile()
        oracle, _ = fit_linear(prof.stats, prof.y)
        row["oracle"] = step_scores(np.array([oracle.step_time(s) for s in test.stats]), test.y)
        if keep_sample:
            idx = np.random.default_rng(0).choice(len(test), min(keep_sample, len(test)), False)
            row["sample"] = {
                "y": [round(float(v), 7) for v in test.y[idx]],
                "pred": [round(float(v), 7) for v in pred[idx]],
                "tokens": [int(test.stats[i].num_tokens) for i in idx],
            }
        rows.append(row)
    return rows


# ---- end to end -------------------------------------------------------------------------


def estimate_frontend(
    meas: Measured,
    cost: StepTimeModel,
    limits: ServeLimits,
    burst_window: float = 0.005,
    max_queue: float = 0.005,
) -> dict[str, float]:
    """Latency outside the engine, from one replay (method of batchsim.validation).

    Simulate the replay with zero frontend latency; for requests that arrived alone while
    the simulated engine was busy and were admitted at once, the median of measured minus
    simulated TTFT is the frontend latency. Requests that found the engine idle also pay
    the engine's wake-up, the ``idle_launch_latency``.
    """
    order = np.argsort(meas.send, kind="stable")
    wl = meas.workload("frontend")
    pred = simulate(wl, cost, "vllm", limits.scheduler_config(), EngineConfig(True))
    gap = meas.first_token[order] - pred.ttft
    start, end = pred.steps["start"], pred.steps["start"] + pred.steps["dur"]
    k = np.searchsorted(start, pred.arrival, side="left") - 1
    busy = (k >= 0) & (pred.arrival < end[np.clip(k, 0, None)])
    gaps = np.diff(pred.arrival)
    lone = np.ones(len(gap), dtype=bool)
    lone[1:] &= gaps > burst_window
    lone[:-1] &= gaps > burst_window
    ok = pred.queue_delay <= max_queue
    b, i = busy & lone & ok, ~busy & lone & ok
    frontend = float(np.median(gap[b])) if b.any() else 0.0
    idle = float(np.median(gap[i]) - frontend) if i.any() else 0.0
    return {"frontend_latency": frontend, "idle_launch_latency": max(0.0, idle)}


def run_limits(info: RunInfo, kv_tokens: int | None = None) -> ServeLimits:
    return ServeLimits(
        kv_tokens if kv_tokens is not None else info.kv_tokens,
        info.max_num_seqs,
        info.max_num_batched_tokens,
        info.max_model_len,
    )


def memory_samples(ds: Dataset) -> list[MemorySample]:
    return [
        MemorySample(
            t.model,
            t.gpu,
            ds.infos[k].max_num_batched_tokens,
            ds.infos[k].max_num_seqs,
            ds.infos[k].kv_tokens,
        )
        for k, t in ds.tables.items()
        if "kv_tokens" in ds.infos[k].server
    ]


def memory_lomo(ds: Dataset) -> list[dict[str, Any]]:
    """KV-cache size forecasts for each pair from a memory model fitted on other models."""
    samples = memory_samples(ds)
    rows = []
    for s in samples:
        train = [t for t in samples if t.model.name != s.model.name]
        if len(train) < 3:
            continue
        mm = fit_memory(train)
        pred = mm.kv_tokens(s.model, s.gpu, s.max_num_batched_tokens, s.max_num_seqs)
        rows.append(
            {
                "model": s.model.name,
                "gpu": s.gpu.name,
                "measured": s.kv_tokens,
                "predicted": pred,
                "rel_err": (pred - s.kv_tokens) / s.kv_tokens,
            }
        )
    return rows


def frontend_table(ds: Dataset) -> dict[tuple[str, str], dict[str, float]]:
    """Per pair, frontend latency estimated on its own profile replay with its oracle."""
    out = {}
    for k, t in ds.tables.items():
        paths = ds.replay_paths(*k)
        if "profile" not in paths:
            continue
        prof = t.profile()
        oracle, _ = fit_linear(prof.stats, prof.y)
        out[k] = estimate_frontend(load_replay(paths["profile"]), oracle, run_limits(ds.infos[k]))
    return out


def pooled_frontend(
    table: dict[tuple[str, str], dict[str, float]], exclude: Callable[[tuple[str, str]], bool]
) -> EngineConfig:
    vals = [v for k, v in table.items() if not exclude(k)]
    if not vals:
        return EngineConfig(True)
    return EngineConfig(
        async_scheduling=True,
        frontend_latency=float(np.median([v["frontend_latency"] for v in vals])),
        idle_launch_latency=float(np.median([v["idle_launch_latency"] for v in vals])),
        record_steps=False,
    )


def replay_compare(
    meas: Measured, cost: StepTimeModel, limits: ServeLimits, engine: EngineConfig
) -> dict[str, dict[str, float]]:
    wl = meas.workload("replay")
    pred = simulate(wl, cost, "vllm", limits.scheduler_config(), engine)
    span = wl.duration
    return compare(meas.as_result(), pred, 0.1 * span, 0.1 * span)


def slo_meets(summary: dict[str, float], slo: SLO) -> bool:
    return bool(
        summary.get("ttft_p90", math.inf) <= slo.ttft
        and summary.get("tpot_p90", math.inf) <= slo.tpot
    )


def evaluate_e2e(
    ds: Dataset,
    fold_list: Sequence[Fold],
    frontend: dict[tuple[str, str], dict[str, float]],
    intervals: bool = True,
) -> list[dict[str, Any]]:
    """Replay every validation trace of each held-out pair through each method's model."""
    samples = memory_samples(ds)
    rows = []
    for f in fold_list:
        key = (f.model, f.gpu)
        info, table = ds.infos[key], ds.tables[key]
        paths = {k: v for k, v in ds.replay_paths(*key).items() if k not in ("profile", "smoke")}
        if not paths:
            continue
        held_gpu = f.gpu if f.tier in ("unseen_gpu", "microbench", "known_model") else None
        held_model = None if f.tier == "known_model" else f.model

        def unseen(
            k: tuple[str, str], m: str | None = held_model, g: str | None = held_gpu
        ) -> bool:
            return k[0] == m or k[1] == g

        mem_train = [s for s in samples if not unseen((s.model.name, s.gpu.name))]
        fe = pooled_frontend(frontend, unseen)
        mm = fit_memory(mem_train) if len(mem_train) >= 3 else MemoryModel()
        kv_pred = mm.kv_tokens(
            table.model, table.gpu, info.max_num_batched_tokens, info.max_num_seqs
        )
        sm: ForecastStepModel = f.coef.bind(table.model, table.gpu, tier=f.tier)
        prof = table.profile()
        oracle, _ = fit_linear(prof.stats, prof.y)
        own_fe = frontend.get(key, {})
        oracle_engine = EngineConfig(
            True,
            own_fe.get("frontend_latency", 0.0),
            own_fe.get("idle_launch_latency", 0.0),
            record_steps=False,
        )
        prior = Coef(dict(PRIOR_COEF), []).bind(table.model, table.gpu)
        for run, path in paths.items():
            meas = load_replay(path)
            if len(meas.send) < 20:
                continue
            row: dict[str, Any] = {
                "tier": f.tier,
                "model": f.model,
                "gpu": f.gpu,
                "run": run,
                "rate": len(meas.send) / max(1e-9, float(np.ptp(meas.send))),
                "kv_measured": info.kv_tokens,
                "kv_predicted": kv_pred,
            }
            lim_pred = run_limits(info, kv_pred)
            row["gpufc"] = replay_compare(meas, sm, lim_pred, fe)
            if intervals and sm.log_interval:
                lo, hi = sm.interval("90")
                row["gpufc_fast"] = replay_compare(meas, sm.scaled(lo), lim_pred, fe)
                row["gpufc_slow"] = replay_compare(meas, sm.scaled(hi), lim_pred, fe)
            row["roofline"] = replay_compare(meas, prior, lim_pred, fe)
            row["oracle"] = replay_compare(meas, oracle, run_limits(info), oracle_engine)
            # A replay is "engine-explained" when the oracle (exact step times and KV size
            # for this pair) gets its median TTFT within 50 %. Otherwise the latency was
            # made outside the engine (an API server or client saturated by a small model
            # at high request rates), which no engine model can forecast.
            # Replays too short to have a latency summary are kept out of both groups.
            o = row["oracle"].get("ttft_p50", {}).get("rel_err", math.nan)
            row["engine_explained"] = bool(abs(o) <= ENGINE_EXPLAINED) if math.isfinite(o) else None
            rows.append(row)
    return rows


def summarize_e2e(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    """Median absolute relative error per metric across replays."""
    out = {}
    for m in E2E_METRICS:
        errs = [
            abs(r[method][m]["rel_err"])
            for r in rows
            if method in r and m in r[method] and math.isfinite(r[method][m]["rel_err"])
        ]
        if errs:
            out[m] = float(np.median(errs))
    return out


def stats_from_shapes(shapes: Sequence[tuple[int, int]]) -> BatchStats:
    s = BatchStats()
    for n, c in shapes:
        s.add(n, c)
    return s


def transfer_attribution(ds: Dataset, max_eval_steps: int = 5000) -> dict[str, Any]:
    """Per GPU: which weights make a fit on the other GPUs miss it.

    Fits every GPU's own pairs and all other GPUs' pairs (microbenchmark peaks, so the
    weights are comparable), takes the median signed error of the other-GPU fit on this
    GPU's real-traffic steps, then repeats with one weight at a time replaced by the
    GPU's own value. A term whose swap removes most of the bias is what the other GPUs
    could not teach.
    """
    from gpufc.cost import TERMS
    from gpufc.fit import DEFAULT_RIDGE, fit_tier

    tables = list(ds)
    out: dict[str, Any] = {}
    for gpu in ds.gpus:
        mine = [t for t in tables if t.gpu.name == gpu]
        rest = [t for t in tables if t.gpu.name != gpu]
        if len(mine) < 2 or len(rest) < 2:
            continue
        kw: dict[str, Any] = {"intervals": False, "ridges": (DEFAULT_RIDGE,)}
        own = fit_tier(mine, measured_peaks=True, **kw)
        other = fit_tier(rest, measured_peaks=True, **kw)
        tests = [t.validation().subsample(max_eval_steps) for t in mine]
        tests = [t for t in tests if len(t)]

        def bias(coef: Coef, tests: list[StepTable] = tests) -> float:
            return float(np.median([np.median(predict(coef, t) / t.y - 1) for t in tests]))

        swap = {}
        for k in TERMS:
            values = {**other.values, k: own.values[k]}
            swap[k] = bias(Coef(values, other.trained_on, True))
        out[gpu] = {"bias": bias(other), "own": own.values, "other": other.values, "swap": swap}
    return out


def implied_gemm_efficiency(ds: Dataset, min_tokens: int = 2048) -> list[dict[str, Any]]:
    """Per dense pair: the GEMM efficiency its long prefill steps imply.

    Fits the per-GPU model (as the bundle does) on all dense pairs of each GPU, then,
    for steps with at least ``min_tokens`` tokens (compute-bound GEMMs), solves each step
    for the ``gemm_max``
    weight that would make it exact with the host, attention and elementwise terms at
    their fitted values. The median of ``1 / weight`` (``efficiency``) is the share of
    datasheet tensor throughput the pair's prefills sustained. ``residual_efficiency``
    solves the same way with the sustained-load term also at its fitted value: what the
    model's single per-GPU efficiency has left to explain.
    """
    from gpufc.cost import TERMS, TermCalculator
    from gpufc.fit import fit_gpu

    out: list[dict[str, Any]] = []
    gi, si = TERMS.index("gemm_max"), TERMS.index("gemm_sustained")
    dense = [t for t in ds if not t.model.is_moe]
    for gpu in ds.gpus:
        tabs = [t for t in dense if t.gpu.name == gpu]
        if len(tabs) < 2:
            continue
        coef = fit_gpu(tabs, dense, intervals=False)
        w = np.array([coef.values[k] for k in TERMS])
        for t in tabs:
            big = [i for i, s in enumerate(t.stats) if s.num_tokens >= min_tokens]
            if len(big) < 10:
                continue
            x = TermCalculator(t.model, t.gpu).matrix([t.stats[i] for i in big])
            y = t.y[big]
            rest = x @ w - x[:, gi] * w[gi] - x[:, si] * w[si]
            raw = (y - rest) / x[:, gi]
            resid = (y - rest - x[:, si] * w[si]) / x[:, gi]
            out.append(
                {
                    "gpu": gpu,
                    "model": t.model.name,
                    "steps": len(big),
                    "step_ms": float(np.median(y) * 1e3),
                    "efficiency": float(1.0 / np.median(raw)),
                    "residual_efficiency": float(1.0 / np.median(resid)),
                    "sustained_weight": float(w[si]),
                }
            )
    return out
