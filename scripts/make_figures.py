"""Figures for the README from ``results/eval``: ``python scripts/make_figures.py``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from report import GPU_LABEL, TIER_LABEL, e2e_med, explained, load, med

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#b8b7b0"
GRID = "#e4e3df"
# Reference categorical palette, first three slots (validated for all-pairs use).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GPU_ORDER = ["h200", "h200-nvl", "a100-80g", "a6000"]
MARKERS = {"conv": "o", "code": "^"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def gpu_color(g: str) -> str:
    return SERIES[GPU_ORDER.index(g)] if g in GPU_ORDER else INK2


def parity_steps(results: Path, out: Path) -> None:
    tiers = [t for t in ("seen_gpu", "microbench") if load(results, f"steps_{t}")]
    fig, axes = plt.subplots(1, len(tiers), figsize=(5.2 * len(tiers), 4.6), squeeze=False)
    for ax, tier in zip(axes[0], tiers):
        rows = load(results, f"steps_{tier}")
        lo, hi = np.inf, 0.0
        for g in GPU_ORDER:
            ys = np.concatenate([r["sample"]["y"] for r in rows if r["gpu"] == g] or [[]])
            ps = np.concatenate([r["sample"]["pred"] for r in rows if r["gpu"] == g] or [[]])
            if not len(ys):
                continue
            ax.scatter(
                ys * 1e3,
                ps * 1e3,
                s=9,
                alpha=0.35,
                color=gpu_color(g),
                linewidths=0,
                label=GPU_LABEL[g],
            )
            lo, hi = min(lo, ys.min(), ps.min()), max(hi, ys.max(), ps.max())
        lims = (lo * 1e3 * 0.8, hi * 1e3 * 1.25)
        ax.plot(lims, lims, color=INK2, lw=1)
        for f in (0.9, 1.1):
            ax.plot(lims, [x * f for x in lims], color=MUTED, lw=0.8, ls="--")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=lims,
            ylim=lims,
            xlabel="measured step time (ms)",
            ylabel="forecast step time (ms)",
        )
        mape = med(rows, "gpufc", "median_ape")
        ax.set_title(
            f"{TIER_LABEL[tier]}\nmedian error {100 * mape:.1f}% ({len(rows)} held-out pairs)",
            loc="left",
        )
        leg = ax.legend(loc="upper left", markerscale=2)
        for h in leg.legend_handles:
            h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(out / "parity_steps.png", dpi=150)
    plt.close(fig)


def method_bars(results: Path, out: Path) -> None:
    lomo = load(results, "steps_seen_gpu")
    mb = load(results, "steps_microbench")
    ds = load(results, "steps_unseen_gpu")
    rows = [
        ("gpufc: new model, profiled GPU", med(lomo, "gpufc", "time_weighted_ape"), True),
        ("gpufc: new model + new GPU (microbench)", med(mb, "gpufc", "time_weighted_ape"), True),
        ("gpufc: new model + new GPU (datasheet)", med(ds, "gpufc", "time_weighted_ape"), True),
        ("boosted trees: new model, profiled GPU", med(lomo, "gbm", "time_weighted_ape"), False),
        ("boosted trees: new model + new GPU", med(ds, "gbm", "time_weighted_ape"), False),
        ("datasheet roofline", med(lomo, "roofline", "time_weighted_ape"), False),
        ("oracle: profile the target itself", med(lomo, "oracle", "time_weighted_ape"), False),
    ]
    rows = [r for r in rows if np.isfinite(r[1])]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    vals = [100 * r[1] for r in rows]
    ax.barh(y, vals, height=0.62, color=[SERIES[0] if r[2] else MUTED for r in rows])
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("time-weighted step-time error on held-out pairs (median over pairs, %)")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title("Step-time error on real traffic of pairs never trained on", loc="left")
    fig.tight_layout()
    fig.savefig(out / "methods.png", dpi=150)
    plt.close(fig)


def e2e_parity(results: Path, out: Path) -> None:
    rows = explained(load(results, "e2e_seen_gpu"))
    if not rows:
        return
    metrics = (("ttft_p90", "TTFT p90"), ("tpot_p50", "TPOT p50"), ("e2e_p90", "E2E p90"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (m, label) in zip(axes, metrics):
        lo, hi = np.inf, 0.0
        for g in GPU_ORDER:
            for kind, mk in MARKERS.items():
                pts = [
                    (r["gpufc"][m]["real"], r["gpufc"][m]["sim"])
                    for r in rows
                    if r["gpu"] == g and r["run"].startswith(kind) and m in r["gpufc"]
                ]
                if not pts:
                    continue
                a = np.array(pts) * 1e3
                ax.scatter(
                    a[:, 0],
                    a[:, 1],
                    s=30,
                    marker=mk,
                    color=gpu_color(g),
                    edgecolors=SURFACE,
                    linewidths=0.8,
                    label=f"{GPU_LABEL[g]}, {kind} trace",
                )
                lo, hi = min(lo, a.min()), max(hi, a.max())
        lims = (lo * 0.7, hi * 1.4)
        ax.plot(lims, lims, color=INK2, lw=1)
        for f in (0.8, 1.25):
            ax.plot(lims, [x * f for x in lims], color=MUTED, lw=0.8, ls="--")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=lims,
            ylim=lims,
            xlabel=f"measured {label} (ms)",
            ylabel=f"forecast {label} (ms)",
        )
        ax.set_title(f"{label}: median error {100 * e2e_med(rows, 'gpufc', m):.1f}%", loc="left")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Serving latency of held-out models (new model, profiled GPU): "
        f"{len(rows)} trace replays; dashed lines at ±20–25%",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "e2e_parity.png", dpi=150)
    plt.close(fig)


def efficiency(results: Path, out: Path) -> None:
    """Implied GEMM efficiency of long prefill steps against their duration, per GPU."""
    rows = load(results, "gemm_efficiency")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for g in GPU_ORDER:
        pts = sorted(
            (r["step_ms"], 100 * r["efficiency"], r["model"]) for r in rows if r["gpu"] == g
        )
        if not pts:
            continue
        xs, ys, names = zip(*pts)
        ax.scatter(
            xs,
            ys,
            s=46,
            color=gpu_color(g),
            edgecolors=SURFACE,
            linewidths=1.5,
            label=GPU_LABEL[g],
            zorder=3,
        )
        if g == "h200":
            for x, y, n in ((xs[0], ys[0], names[0]), (xs[-1], ys[-1], names[-1])):
                ax.annotate(
                    n, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK2
                )
            resid = sorted(
                (r["step_ms"], 100 * r["residual_efficiency"]) for r in rows if r["gpu"] == g
            )
            ax.scatter(
                [p[0] for p in resid],
                [p[1] for p in resid],
                s=46,
                facecolors="none",
                edgecolors=gpu_color(g),
                linewidths=1.3,
                label="H200, left after the sustained-load term",
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set(
        xlabel="median duration of the pair's prefill steps with >= 2,048 tokens (ms)",
        ylabel="sustained % of datasheet bf16 peak",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("GEMM efficiency implied by long prefill steps, one point per model", loc="left")
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    parity_steps(results, out)
    method_bars(results, out)
    e2e_parity(results, out)
    efficiency(results, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
