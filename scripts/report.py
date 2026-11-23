"""Aggregate ``results/eval/*.json`` into the tables the README and figures show."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gpufc.specs import get_model

TIER_LABEL = {
    "seen_gpu": "new model, profiled GPU",
    "size_extrapolation": "larger model (fit on <=8B), profiled GPU",
    "known_model": "profiled model, new GPU (microbenchmark)",
    "unseen_gpu": "new model, new GPU (datasheet)",
    "microbench": "new model, new GPU (1-min microbenchmark)",
}
GPU_LABEL = {
    "h200": "H200",
    "h200-nvl": "H200 NVL",
    "a100-80g": "A100-80GB",
    "a6000": "RTX A6000",
    "t4": "T4",
}


def load(results: Path, name: str) -> Any:
    p = results / "eval" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else []


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{digits}f}%"


def med(rows: list[dict[str, Any]], method: str, key: str) -> float:
    vals = [r[method][key] for r in rows if method in r and key in r[method]]
    return float(np.median(vals)) if vals else math.nan


def mean(rows: list[dict[str, Any]], method: str, key: str) -> float:
    vals = [r[method][key] for r in rows if method in r and key in r[method]]
    return float(np.mean(vals)) if vals else math.nan


def step_table(results: Path) -> list[list[str]]:
    """Step-time error on held-out pairs' real-traffic steps, by method and setting."""
    out = []
    for tier in ("seen_gpu", "size_extrapolation", "known_model", "microbench", "unseen_gpu"):
        rows = load(results, f"steps_{tier}")
        if not rows:
            continue
        for method, label in (("gpufc", "**gpufc**"), ("gbm", "gradient-boosted trees")):
            # Trees ignore the microbenchmark peaks: show them once per training set.
            if not any(method in r for r in rows) or (
                method == "gbm" and tier in ("microbench", "known_model", "size_extrapolation")
            ):
                continue
            out.append(
                [
                    label,
                    TIER_LABEL[tier],
                    str(len(rows)),
                    pct(med(rows, method, "median_ape")),
                    pct(med(rows, method, "time_weighted_ape")),
                    pct(med(rows, method, "p90_ape")),
                ]
            )
    rows = load(results, "steps_seen_gpu")
    if rows:
        out.append(
            [
                "datasheet roofline",
                "hand-set weights, no fit",
                str(len(rows)),
                pct(med(rows, "roofline", "median_ape")),
                pct(med(rows, "roofline", "time_weighted_ape")),
                pct(med(rows, "roofline", "p90_ape")),
            ]
        )
        out.append(
            [
                "oracle (profile the target)",
                "target profiled",
                str(len(rows)),
                pct(med(rows, "oracle", "median_ape")),
                pct(med(rows, "oracle", "time_weighted_ape")),
                pct(med(rows, "oracle", "p90_ape")),
            ]
        )
    return out


E2E_COLS = ("ttft_p50", "ttft_p90", "tpot_p50", "tpot_p90", "e2e_p50", "e2e_p90")


def explained(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replays whose latency the engine accounts for (see gpufc.evaluate.ENGINE_EXPLAINED)."""
    return [r for r in rows if r.get("engine_explained", True) is True]


def e2e_med(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    errs = [
        abs(r[method][metric]["rel_err"])
        for r in rows
        if method in r and metric in r[method] and math.isfinite(r[method][metric]["rel_err"])
    ]
    return float(np.median(errs)) if errs else math.nan


def e2e_table(results: Path) -> list[list[str]]:
    out = []
    for tier in ("seen_gpu", "size_extrapolation", "known_model", "microbench", "unseen_gpu"):
        rows = explained(load(results, f"e2e_{tier}"))
        if not rows:
            continue
        out.append(
            ["**gpufc**", TIER_LABEL[tier], str(len(rows))]
            + [pct(e2e_med(rows, "gpufc", c)) for c in E2E_COLS]
        )
    rows = explained(load(results, "e2e_seen_gpu"))
    if rows:
        out.append(
            ["datasheet roofline", "hand-set weights, no fit", str(len(rows))]
            + [pct(e2e_med(rows, "roofline", c)) for c in E2E_COLS]
        )
        out.append(
            ["oracle (profile the target)", "target profiled", str(len(rows))]
            + [pct(e2e_med(rows, "oracle", c)) for c in E2E_COLS]
        )
    return out


def per_gpu_table(results: Path) -> list[list[str]]:
    rows = load(results, "steps_seen_gpu")
    e2e = explained(load(results, "e2e_seen_gpu"))
    out = []
    for g in sorted(
        {r["gpu"] for r in rows}, key=lambda g: list(GPU_LABEL).index(g) if g in GPU_LABEL else 99
    ):
        rg = [r for r in rows if r["gpu"] == g]
        eg = [r for r in e2e if r["gpu"] == g]
        out.append(
            [
                GPU_LABEL.get(g, g),
                str(len(rg)),
                pct(med(rg, "gpufc", "median_ape")),
                pct(med(rg, "oracle", "median_ape")),
                pct(e2e_med(eg, "gpufc", "ttft_p90")),
                pct(e2e_med(eg, "gpufc", "tpot_p50")),
                pct(e2e_med(eg, "oracle", "tpot_p50")),
            ]
        )
    return out


def new_gpu_table(results: Path) -> list[list[str]]:
    """Per GPU, held out entirely: gpufc (microbenchmark peaks) against the roofline, and
    the fixed host time per step that GPU's own fit finds (not visible to the forecast)."""
    rows = load(results, "steps_microbench")
    e2e = explained(load(results, "e2e_microbench"))
    coefs = load(results, "coef_seen_gpu")
    out = []
    for g in [g for g in GPU_LABEL if any(r["gpu"] == g for r in rows)]:
        rg = [r for r in rows if r["gpu"] == g]
        eg = [r for r in e2e if r["gpu"] == g]
        host = [c["values"]["const"] for c in coefs if c["gpu"] == g]
        out.append(
            [
                GPU_LABEL.get(g, g),
                str(len(rg)),
                pct(med(rg, "gpufc", "time_weighted_ape")),
                pct(med(rg, "roofline", "time_weighted_ape")),
                pct(e2e_med(eg, "gpufc", "ttft_p50")),
                pct(e2e_med(eg, "roofline", "ttft_p50")),
                f"{float(np.median(host)):.2f} ms" if host else "-",
            ]
        )
    return out


def per_model_table(results: Path) -> list[list[str]]:
    rows = load(results, "steps_seen_gpu")
    gpus = [g for g in GPU_LABEL if any(r["gpu"] == g for r in rows)]
    models = sorted({r["model"] for r in rows}, key=lambda m: get_model(m).num_params)
    out = []
    for m in models:
        line = [m]
        for g in gpus:
            hit = [r for r in rows if r["model"] == m and r["gpu"] == g]
            line.append(pct(hit[0]["gpufc"]["median_ape"]) if hit else "-")
        out.append(line)
    return [["model", *[GPU_LABEL[g] for g in gpus]], *out]


def coverage_table(results: Path) -> list[list[str]]:
    out = []
    for tier in ("seen_gpu", "size_extrapolation", "known_model", "microbench", "unseen_gpu"):
        rows = load(results, f"steps_{tier}")
        e2e = explained(load(results, f"e2e_{tier}"))
        if not rows:
            continue
        inside = []
        for r in e2e:
            if "gpufc_fast" not in r:
                continue
            for m in ("ttft_p90", "tpot_p50"):
                if not all(m in r[k] for k in ("gpufc", "gpufc_fast", "gpufc_slow")):
                    continue
                real = r["gpufc"][m]["real"]
                lo = min(r["gpufc_fast"][m]["sim"], r["gpufc_slow"][m]["sim"])
                hi = max(r["gpufc_fast"][m]["sim"], r["gpufc_slow"][m]["sim"])
                inside.append(lo <= real <= hi)
        out.append(
            [
                TIER_LABEL[tier],
                pct(mean(rows, "gpufc", "cov90")),
                pct(float(np.mean(inside)) if inside else math.nan),
            ]
        )
    return out


def memory_table(results: Path) -> tuple[float, float, int]:
    rows = load(results, "memory")
    if not rows:
        return math.nan, math.nan, 0
    errs = [abs(r["rel_err"]) for r in rows]
    return float(np.median(errs)), float(np.max(errs)), len(rows)


def slo_accuracy(results: Path, tier: str, ttft: float = 1.0, tpot: float = 0.05) -> float:
    """Share of replays where the forecast and the measurement agree on SLO pass/fail."""
    rows = explained(load(results, f"e2e_{tier}"))
    hits = []
    for r in rows:
        g = r["gpufc"]
        if "ttft_p90" not in g or "tpot_p90" not in g:
            continue
        real = g["ttft_p90"]["real"] <= ttft and g["tpot_p90"]["real"] <= tpot
        sim = g["ttft_p90"]["sim"] <= ttft and g["tpot_p90"]["sim"] <= tpot
        hits.append(real == sim)
    return float(np.mean(hits)) if hits else math.nan


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)
