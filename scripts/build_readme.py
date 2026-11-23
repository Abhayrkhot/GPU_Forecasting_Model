"""Render README.md from docs/README.tmpl.md and the numbers in results/.

    python scripts/build_readme.py           # write README.md
    python scripts/build_readme.py --check   # exit 1 if README.md is stale

Every number in the README comes from ``results/`` through this script; the template
holds only prose and ``{{TOKENS}}``.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from report import (
    E2E_COLS,
    GPU_LABEL,
    coverage_table,
    e2e_med,
    e2e_table,
    explained,
    load,
    md_table,
    med,
    memory_table,
    new_gpu_table,
    pct,
    per_gpu_table,
    per_model_table,
    slo_accuracy,
    step_table,
)

ROOT = Path(__file__).resolve().parent.parent


def dataset_stats(results: Path) -> dict[str, str]:
    runs = sorted((results / "runs").glob("*/*.json"))
    steps = 0
    for p in sorted((results / "steps").glob("*/*.csv.gz")):
        with gzip.open(p, "rt") as f:
            steps += sum(1 for _ in f) - 1
    replays = [
        p
        for p in (results / "replays").glob("*/*/*.jsonl.gz")
        if p.name.split(".")[0] not in ("profile", "smoke")
    ]
    requests = 0
    for p in replays:
        with gzip.open(p, "rt") as f:
            requests += sum(1 for line in f if '"summary"' not in line)
    order = list(GPU_LABEL)
    gpus = sorted({p.parent.name for p in runs}, key=lambda g: order.index(g) if g in order else 99)
    models = sorted({p.stem for p in runs})
    return {
        "PAIRS": str(len(runs)),
        "GPUS": ", ".join(GPU_LABEL.get(g, g) for g in gpus),
        "NUM_GPUS": str(len(gpus)),
        "NUM_MODELS": str(len(models)),
        "STEPS": f"{steps:,}",
        "REPLAYS": str(len(replays)),
        "REQUESTS": f"{requests:,}",
    }


def signed_pct(v: float) -> str:
    return f"{100 * v:+.1f}%".replace("-", "\u2212")


def tokens(results: Path) -> dict[str, str]:
    t = dataset_stats(results)
    lomo, mb, ds = (load(results, f"steps_{k}") for k in ("seen_gpu", "microbench", "unseen_gpu"))
    ext = load(results, "steps_size_extrapolation")
    km, e_km = load(results, "steps_known_model"), load(results, "e2e_known_model")
    t["KM_STEP_TW"] = pct(med(km, "gpufc", "time_weighted_ape"))
    t["KM_TTFT_P90"] = pct(e2e_med(explained(e_km), "gpufc", "ttft_p90"))
    t["KM_TPOT_P50"] = pct(e2e_med(explained(e_km), "gpufc", "tpot_p50"))
    t["EXT_STEP_TW"] = pct(med(ext, "gpufc", "time_weighted_ape"))
    t["EXT_N"] = str(len(ext))
    e_lomo, e_mb = load(results, "e2e_seen_gpu"), load(results, "e2e_microbench")
    t["LOMO_STEP_MED"] = pct(med(lomo, "gpufc", "median_ape"))
    t["LOMO_STEP_TW"] = pct(med(lomo, "gpufc", "time_weighted_ape"))
    t["MB_STEP_TW"] = pct(med(mb, "gpufc", "time_weighted_ape"))
    t["DS_STEP_TW"] = pct(med(ds, "gpufc", "time_weighted_ape"))
    t["GBM_STEP_TW"] = pct(med(lomo, "gbm", "time_weighted_ape"))
    t["GBM_DS_STEP_TW"] = pct(med(ds, "gbm", "time_weighted_ape"))
    t["ROOF_STEP_TW"] = pct(med(lomo, "roofline", "time_weighted_ape"))
    t["ORACLE_STEP_TW"] = pct(med(lomo, "oracle", "time_weighted_ape"))
    e_lomo, e_mb = explained(e_lomo), explained(e_mb)
    for m in E2E_COLS:
        t[f"LOMO_{m.upper()}"] = pct(e2e_med(e_lomo, "gpufc", m))
        t[f"MB_{m.upper()}"] = pct(e2e_med(e_mb, "gpufc", m))
        t[f"ORACLE_{m.upper()}"] = pct(e2e_med(e_lomo, "oracle", m))
        t[f"ROOF_{m.upper()}"] = pct(e2e_med(e_lomo, "roofline", m))
    mem_med, mem_max, mem_n = memory_table(results)
    t["MEM_MED"], t["MEM_MAX"], t["MEM_N"] = pct(mem_med), pct(mem_max), str(mem_n)
    t["SLO_LOMO"] = pct(slo_accuracy(results, "seen_gpu"), 0)
    t["SLO_MB"] = pct(slo_accuracy(results, "microbench"), 0)
    t["STEP_TABLE"] = md_table(
        ["method", "setting", "pairs", "median error", "time-weighted", "p90 error"],
        step_table(results),
        "llrrrr",
    )
    t["E2E_TABLE"] = md_table(
        [
            "method",
            "setting",
            "replays",
            "TTFT p50",
            "TTFT p90",
            "TPOT p50",
            "TPOT p90",
            "E2E p50",
            "E2E p90",
        ],
        e2e_table(results),
        "llrrrrrrr",
    )
    t["PER_GPU_TABLE"] = md_table(
        [
            "GPU",
            "held-out models",
            "step error",
            "oracle step error",
            "TTFT p90 error",
            "TPOT p50 error",
            "oracle TPOT p50",
        ],
        per_gpu_table(results),
    )
    ng = new_gpu_table(results)
    t["NEW_GPU_TABLE"] = md_table(
        [
            "held-out GPU",
            "models",
            "gpufc step error",
            "roofline step error",
            "gpufc TTFT p50",
            "roofline TTFT p50",
            "host time per step",
        ],
        ng,
    )
    errs = [float(c.rstrip("%")) for r in ng for c in (r[2], r[4]) if c.endswith("%")]
    if errs:
        t["NEW_GPU_ERR_LO"], t["NEW_GPU_ERR_HI"] = f"{min(errs):.0f}%", f"{max(errs):.0f}%"
    host = {r[0]: float(r[6].split()[0]) for r in ng if r[6] != "-"}
    if host:
        lo, hi = min(host, key=lambda k: host[k]), max(host, key=lambda k: host[k])
        t["HOST_MS_LO"], t["HOST_GPU_LO"] = f"{host[lo]:.1f} ms", lo
        t["HOST_MS_HI"], t["HOST_GPU_HI"] = f"{host[hi]:.1f} ms", hi
    tr = load(results, "transfer")
    if tr:
        if "a100-80g" in tr:
            a = tr["a100-80g"]
            t["A100_TRANSFER_BIAS"] = signed_pct(a["bias"])
            t["A100_SWAP_CONST"] = signed_pct(a["swap"]["const"])
            t["A100_HOST_OWN"] = f"{a['own']['const']:.1f} ms"
            t["A100_HOST_OTHER"] = f"{a['other']['const']:.1f} ms"
        if "a6000" in tr:
            a = tr["a6000"]
            t["A6000_TRANSFER_BIAS"] = signed_pct(a["bias"])
            t["A6000_SWAP_GEMM"] = signed_pct(a["swap"]["gemm_max"])
            t["A6000_GEMM_OWN"] = pct(1 / a["own"]["gemm_max"], 0)
            t["A6000_GEMM_OTHER"] = pct(1 / a["other"]["gemm_max"], 0)
    pm = per_model_table(results)
    t["PER_MODEL_TABLE"] = md_table(pm[0], pm[1:]) if len(pm) > 1 else ""
    t["COVERAGE_TABLE"] = md_table(
        ["setting", "steps inside 90% interval", "replays inside band (TTFT p90, TPOT p50)"],
        coverage_table(results),
        "lrr",
    )
    eff = load(results, "gemm_efficiency")
    for prefix, gpu in (("H200", "h200"), ("A6000", "a6000")):
        rows = sorted((r for r in eff if r["gpu"] == gpu), key=lambda r: r["step_ms"])
        if len(rows) < 2:
            continue
        lo, hi = rows[0], rows[-1]
        t[f"{prefix}_EFF_SHORT"] = pct(lo["efficiency"], 0)
        t[f"{prefix}_EFF_SHORT_MS"] = f"{lo['step_ms']:.0f} ms"
        t[f"{prefix}_EFF_SHORT_MODEL"] = lo["model"]
        t[f"{prefix}_EFF_LONG"] = pct(hi["efficiency"], 0)
        t[f"{prefix}_EFF_LONG_MS"] = f"{hi['step_ms']:.0f} ms"
        t[f"{prefix}_EFF_LONG_MODEL"] = hi["model"]
        resid = [r["residual_efficiency"] for r in rows]
        t[f"{prefix}_RESID_MIN"] = pct(min(resid), 0)
        t[f"{prefix}_RESID_MAX"] = pct(max(resid), 0)
        t[f"{prefix}_RESID_N"] = str(len(rows))
    fe = load(results, "frontend")
    if fe:
        vals = [v["frontend_latency"] * 1e3 for v in fe.values()]
        t["FRONTEND_MS"] = f"{float(np.median(vals)):.1f} ms"
    else:
        t["FRONTEND_MS"] = "n/a"
    t["E2E_LOMO_N"] = str(len(explained(e_lomo)))
    unexplained = [r for r in load(results, "e2e_seen_gpu") if r.get("engine_explained") is False]
    t["E2E_UNEXPLAINED"] = str(len(unexplained))
    t["E2E_UNEXPLAINED_PAIRS"] = (
        ", ".join(
            sorted({f"{r['model']} on {GPU_LABEL.get(r['gpu'], r['gpu'])}" for r in unexplained})
        )
        or "none"
    )
    return t


def render(results: Path, template: Path) -> str:
    text = template.read_text()
    for k, v in tokens(results).items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        missing = sorted({s.split("}}")[0] for s in text.split("{{")[1:]})
        raise SystemExit(f"unfilled tokens: {missing}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--template", default=str(ROOT / "docs" / "README.tmpl.md"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(Path(args.results), Path(args.template))
    readme = ROOT / "README.md"
    if args.check:
        if readme.read_text() != text:
            print("README.md is stale: run python scripts/build_readme.py")
            return 1
        print("README.md is up to date")
        return 0
    readme.write_text(text)
    print("wrote README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
