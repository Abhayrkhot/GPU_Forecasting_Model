"""Fitting the transferable step model and its prediction intervals.

The weights of :class:`~gpufc.cost.ForecastStepModel` are fitted by non-negative least
squares on *relative* error over steps from many (model, GPU) pairs. Each pair gets equal
total weight, however many steps it logged, so a fast small model does not drown out a
slow large one.

Three forecasting tiers differ only in which pairs are used and which peaks the terms use:

``seen_gpu``     the target GPU was profiled with other models; fit on that GPU's pairs,
                 shrunk towards the fit on every GPU's pairs of those other models.
``unseen_gpu``   the target GPU was never run; fit on all other GPUs, datasheet peaks.
``microbench``   as ``unseen_gpu``, but every GPU's peaks come from a one-minute GEMM and
                 copy microbenchmark instead of the datasheet.

Fits shrink the weights towards a prior (ridge on relative deviations): the physical
prior for pooled fits, the pooled fit for per-GPU fits, so a GPU profiled with two models
leans on what every GPU showed and one profiled with a dozen follows its own data. The
ridge strength is chosen per fit by leave-one-pair-out cross-validation over the training
pairs: each pair in turn is held out, the rest are fitted, and the held-out pair's
real-traffic steps are predicted. The same held-out residuals ``log(measured/predicted)``
then give group-conformal prediction intervals: they answer "how wrong is this
forecaster on a pair it has not seen", the question a user forecasting a new model or GPU
has. No target data is touched at any point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from batchsim.calibrate import nnls
from numpy.typing import NDArray

from gpufc.cost import HOST_TERMS, TERMS, ForecastStepModel, TermCalculator
from gpufc.data import StepTable
from gpufc.specs import GPUSpec, ModelSpec

FloatArray = NDArray[np.float64]
INTERVAL_LEVELS = {"80": (0.10, 0.90), "90": (0.05, 0.95), "95": (0.025, 0.975)}
DEFAULT_MAX_STEPS = 6000
SMALL_MODEL_PARAMS = 8.5e9  # Qwen3-8B and smaller
DEFAULT_RIDGE = 1e-3
RIDGE_GRID = (1e-4, 1e-3, 1e-2)
# Per-GPU fits shrink towards the pooled fit, a prior worth trusting much further: the
# grid reaches ridges at which a GPU with few profiled models simply takes the pooled
# weights.
POOLED_RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
MEASURED_PEAK_TIERS = ("microbench", "known_model", "unseen_gpu_known_model")
# Host overheads do not depend on the GPU spec and are well determined across pairs, so
# their prior is ten times looser than the hardware efficiencies'.
HOST_PRIOR_SCALE = 10.0
CV_EVAL_STEPS = 3000
# Weights of the terms that are a time at peak throughput are inverse efficiencies;
# bounding them below by 0.8 means no kernel beats its peak by more than 25 %.
ROOFLINE_TERMS = (
    "gemm_max",
    "moe_max",
    "attn_prefill",
    "attn_decode",
    "elementwise",
)
MIN_INV_EFFICIENCY = 0.8
# Prior scale for terms whose prior is zero: the sustained-load penalty is GPU-specific
# (zero on GPUs that do not throttle), so it is only loosely tied to its prior of 0.
PRIOR_SCALE = {"gemm_sustained": 1.0}


@dataclass
class Design:
    """Term matrix and targets for a set of steps from several pairs."""

    x: FloatArray
    y: FloatArray
    group: NDArray[np.int64]
    keys: list[tuple[str, str]]

    def __len__(self) -> int:
        return len(self.y)


def design(
    tables: Sequence[StepTable],
    measured_peaks: bool = False,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
) -> Design:
    xs, ys, gs, keys = [], [], [], []
    for gi, t in enumerate(tables):
        t = t.subsample(max_steps, seed)
        calc = TermCalculator(t.model, t.gpu, measured_peaks=measured_peaks)
        xs.append(calc.matrix(t.stats))
        ys.append(t.y)
        gs.append(np.full(len(t.y), gi, dtype=np.int64))
        keys.append(t.key)
    if not xs:
        raise ValueError("no training steps")
    return Design(np.vstack(xs), np.concatenate(ys), np.concatenate(gs), keys)


def combine(parts: Sequence[Design]) -> Design:
    """Stack per-pair designs (each with one group) into one multi-pair design."""
    if not parts:
        raise ValueError("no training steps")
    groups = [np.full(len(p.y), i, dtype=np.int64) for i, p in enumerate(parts)]
    return Design(
        np.vstack([p.x for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate(groups),
        [k for p in parts for k in p.keys],
    )


@dataclass
class Coef:
    """Fitted weights plus the bookkeeping of what they were fitted on."""

    values: dict[str, float]
    trained_on: list[tuple[str, str]]
    measured_peaks: bool = False
    log_interval: dict[str, float] = field(default_factory=dict)
    report: dict[str, float] = field(default_factory=dict)

    def bind(self, model: ModelSpec, gpu: GPUSpec, **meta: Any) -> ForecastStepModel:
        return ForecastStepModel(
            model,
            gpu,
            dict(self.values),
            measured_peaks=self.measured_peaks,
            log_interval=dict(self.log_interval),
            meta={"trained_on": [list(k) for k in self.trained_on], **meta},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "trained_on": [list(k) for k in self.trained_on],
            "measured_peaks": self.measured_peaks,
            "log_interval": self.log_interval,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Coef:
        return cls(
            {k: float(v) for k, v in d["values"].items()},
            [(str(a), str(b)) for a, b in d["trained_on"]],
            bool(d.get("measured_peaks", False)),
            {k: float(v) for k, v in d.get("log_interval", {}).items()},
            {k: float(v) for k, v in d.get("report", {}).items()},
        )


def _row_weights(d: Design) -> FloatArray:
    """Relative error, and equal total weight per pair: w_i = 1 / (y_i * sqrt(n_group))."""
    counts = np.bincount(d.group)
    w: FloatArray = 1.0 / (d.y * np.sqrt(counts[d.group]))
    return w


def fit_coef(
    d: Design,
    prior: dict[str, float] | None = None,
    trim_quantile: float = 0.995,
    measured_peaks: bool = False,
    ridge: float = DEFAULT_RIDGE,
) -> Coef:
    """NNLS on relative error with per-pair weights, shrunk towards a physical prior.

    Minimises ``sum_i w_i^2 (x_i c - y_i)^2 + ridge * P * sum_k ((c_k - p_k) / s_k)^2``
    over ``c >= lb`` (0, or :data:`MIN_INV_EFFICIENCY` for roofline terms), where
    ``w_i = 1 / (y_i sqrt(n_pair))`` (relative error, equal weight per pair), ``P`` is the
    number of pairs, ``p`` the prior and ``s_k = max(p_k, 0.1)`` (times
    :data:`HOST_PRIOR_SCALE` for host terms).
    Weights the data pins down move freely; weights it cannot separate (terms that are
    collinear in the training set, or zero in it, such as MoE terms when only dense models
    were profiled) stay near physically plausible values instead of taking whatever
    extreme value happens to fit, which is what makes them transfer to other GPUs. The
    worst ``1 - trim_quantile`` of steps (one-off stalls) are dropped and the fit redone.
    """
    from gpufc.workloads import PRIOR_COEF

    prior = prior or PRIOR_COEF
    p = np.array([prior[t] for t in TERMS], dtype=np.float64)
    host = np.array([t in HOST_TERMS for t in TERMS])
    scale = np.maximum(p, 0.1) * np.where(host, HOST_PRIOR_SCALE, 1.0)
    for i, t in enumerate(TERMS):
        if t in PRIOR_SCALE:
            scale[i] = PRIOR_SCALE[t]
    lam = np.sqrt(ridge * len(d.keys))
    reg_x = np.diag(lam / scale)
    reg_y = lam * p / scale
    w = _row_weights(d)

    # c = lb + v with v >= 0: no roofline term may run faster than 1/MIN_INV_EFFICIENCY
    # times its peak (nothing beats the datasheet by much; with few training pairs an
    # unconstrained weight can wander there to absorb another term's error).
    lb = np.array([MIN_INV_EFFICIENCY if t in ROOFLINE_TERMS else 0.0 for t in TERMS])

    def solve(mask: NDArray[np.bool_]) -> FloatArray:
        a = np.vstack([d.x[mask] * w[mask, None], reg_x])
        b = np.concatenate([(d.y[mask] - d.x[mask] @ lb) * w[mask], reg_y - reg_x @ lb])
        out: FloatArray = lb + nnls(a, b)
        return out

    keep: NDArray[np.bool_] = np.ones(len(d.y), dtype=bool)
    coef = solve(keep)
    rel = np.abs(d.x @ coef - d.y) / d.y
    if 0 < trim_quantile < 1 and len(rel) > 50:
        keep = rel <= np.quantile(rel, trim_quantile)
        coef = solve(keep)
        rel = np.abs(d.x @ coef - d.y) / d.y
    values = {t: float(c) for t, c in zip(TERMS, coef)}
    report = {
        "steps": float(len(d.y)),
        "pairs": float(len(d.keys)),
        "ridge": float(ridge),
        "train_mape": float(rel.mean()),
        "train_median_ape": float(np.median(rel)),
    }
    return Coef(values, list(d.keys), measured_peaks, report=report)


def predict(coef: Coef, table: StepTable) -> FloatArray:
    calc = TermCalculator(table.model, table.gpu, measured_peaks=coef.measured_peaks)
    w = np.array([coef.values[t] for t in TERMS])
    out: FloatArray = calc.matrix(table.stats) @ w
    return out


def interval_from_residuals(res: FloatArray) -> dict[str, float]:
    if len(res) == 0:
        return {}
    out = {}
    for level, (lo, hi) in INTERVAL_LEVELS.items():
        out[f"lo{level}"] = float(np.quantile(res, lo))
        out[f"hi{level}"] = float(np.quantile(res, hi))
    out["median"] = float(np.median(res))
    return out


def _holdout_rows(t: StepTable, seed: int = 0) -> StepTable:
    """Steps of a held-out training pair to score on: its real-traffic replays if any."""
    val = t.validation()
    return (val if len(val) else t).subsample(CV_EVAL_STEPS, seed)


def fit_tier(
    tables: Sequence[StepTable],
    measured_peaks: bool = False,
    intervals: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
    ridges: Sequence[float] = RIDGE_GRID,
    prior: dict[str, float] | None = None,
) -> Coef:
    """Fit on ``tables``, choosing the ridge and the intervals by leave-one-pair-out.

    ``prior`` defaults to the physical prior. Cross-validation needs three pairs (two
    with an informative ``prior``, when a fit on one pair is still meaningful); with fewer
    the default ridge is used and no interval is attached.
    """
    parts = [design([t], measured_peaks, max_steps) for t in tables]

    def fit_on(idx: Sequence[int], ridge: float) -> Coef:
        return fit_coef(
            combine([parts[i] for i in idx]),
            prior=prior,
            measured_peaks=measured_peaks,
            ridge=ridge,
        )

    everything = list(range(len(parts)))
    min_pairs = 3 if prior is None else 2
    if len(parts) < min_pairs or not (intervals or len(ridges) > 1):
        return fit_on(everything, DEFAULT_RIDGE)
    held = [design([_holdout_rows(t)], measured_peaks, CV_EVAL_STEPS) for t in tables]
    scores: dict[float, float] = {}
    residuals: dict[float, FloatArray] = {}
    for ridge in ridges:
        errs, res = [], []
        for i in everything:
            coef = fit_on([j for j in everything if j != i], ridge)
            w = np.array([coef.values[t] for t in TERMS])
            pred = held[i].x @ w
            y = held[i].y
            errs.append(float(np.abs(pred - y).sum() / y.sum()))
            res.append(np.log(y / pred))
        scores[ridge] = float(np.median(errs))
        residuals[ridge] = np.concatenate(res)
    best = min(scores, key=lambda r: scores[r])
    coef = fit_on(everything, best)
    if intervals:
        coef.log_interval = interval_from_residuals(residuals[best])
    coef.report.update({f"cv_tw_err_ridge_{r:g}": v for r, v in scores.items()})
    return coef


def select_training(
    tables: Sequence[StepTable],
    target_model: str,
    target_gpu: str,
    tier: str,
) -> list[StepTable]:
    """Training pairs for forecasting ``(target_model, target_gpu)`` in ``tier``.

    ``seen_gpu``: same GPU, other models. ``unseen_gpu`` / ``microbench``: other GPUs,
    other models (the strict setting: neither the model nor the GPU was ever measured).
    ``known_model`` (alias ``unseen_gpu_known_model``) relaxes that to other GPUs, any
    model: moving a model you already serve to a GPU you have not profiled.
    ``size_extrapolation``: same GPU, only models up to :data:`SMALL_MODEL_PARAMS`, for a
    target larger than that (empty otherwise): forecast big models from small ones.
    """
    if tier == "seen_gpu":
        return [t for t in tables if t.gpu.name == target_gpu and t.model.name != target_model]
    if tier == "size_extrapolation":
        big = [t for t in tables if t.model.name == target_model]
        if not big or big[0].model.num_params <= SMALL_MODEL_PARAMS:
            return []
        return [
            t
            for t in tables
            if t.gpu.name == target_gpu and t.model.num_params <= SMALL_MODEL_PARAMS
        ]
    if tier in ("unseen_gpu", "microbench"):
        return [t for t in tables if t.gpu.name != target_gpu and t.model.name != target_model]
    if tier in ("unseen_gpu_known_model", "known_model"):
        return [t for t in tables if t.gpu.name != target_gpu]
    raise ValueError(f"unknown tier {tier!r}")


def shrink_pool(
    tables: Sequence[StepTable], target_model: str, tier: str
) -> list[StepTable] | None:
    """Pairs whose pooled fit a per-GPU tier shrinks towards (``None`` for pooled tiers).

    Every GPU's pairs that the tier's own premise allows: for ``seen_gpu`` all models but
    the target, for ``size_extrapolation`` only models up to :data:`SMALL_MODEL_PARAMS`.
    """
    if tier == "seen_gpu":
        return [t for t in tables if t.model.name != target_model]
    if tier == "size_extrapolation":
        return [t for t in tables if t.model.num_params <= SMALL_MODEL_PARAMS]
    return None


def pooled_prior(pool: Sequence[StepTable], measured_peaks: bool = False) -> dict[str, float]:
    """Weights fitted on ``pool`` with the physical prior: the prior of a per-GPU fit."""
    return fit_tier(pool, measured_peaks, intervals=False, ridges=(DEFAULT_RIDGE,)).values


def fit_gpu(
    tables: Sequence[StepTable],
    pool: Sequence[StepTable],
    measured_peaks: bool = False,
    intervals: bool = True,
) -> Coef:
    """Per-GPU fit on ``tables``, shrunk towards the pooled fit on ``pool``.

    How far it may move from the pooled weights is chosen by leave-one-pair-out
    cross-validation over ``tables`` (:data:`POOLED_RIDGE_GRID`).
    """
    coef = fit_tier(
        tables,
        measured_peaks,
        intervals,
        ridges=POOLED_RIDGE_GRID,
        prior=pooled_prior(pool, measured_peaks),
    )
    coef.report["pool_pairs"] = float(len(pool))
    return coef


def fit_for_target(
    tables: Sequence[StepTable],
    target_model: str,
    target_gpu: str,
    tier: str,
    intervals: bool = True,
) -> tuple[Coef, list[StepTable]] | None:
    """The tier's fit for forecasting ``(target_model, target_gpu)`` and its training pairs.

    ``None`` when the tier has fewer than two training pairs for this target.
    """
    train = select_training(tables, target_model, target_gpu, tier)
    if len(train) < 2:
        return None
    measured = tier in MEASURED_PEAK_TIERS
    pool = shrink_pool(tables, target_model, tier)
    if pool is None:
        return fit_tier(train, measured, intervals), train
    return fit_gpu(train, pool, measured, intervals), train
