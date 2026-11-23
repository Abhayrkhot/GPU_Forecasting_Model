"""Run every held-out evaluation on ``results/`` and write ``results/eval/*.json``.

    python scripts/evaluate.py [--workers 16] [--no-gbm]

Protocols (see ``gpufc.evaluate``): ``seen_gpu`` is leave-one-model-out on each GPU;
``size_extrapolation`` fits on models up to 8B and forecasts the larger ones on the same
GPU; ``known_model`` forecasts a pair from the other GPUs (the same model included),
with microbenchmark peaks; ``unseen_gpu`` and ``microbench`` are strict leave-one-GPU-out
(the target model is also excluded from training), from datasheet peaks and from
microbenchmark peaks. Each fold is independent, so folds run in parallel processes.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process: folds already run in parallel.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from gpufc.data import Dataset
from gpufc.evaluate import (
    Fold,
    evaluate_e2e,
    evaluate_steps,
    frontend_table,
    implied_gemm_efficiency,
    memory_lomo,
    transfer_attribution,
)
from gpufc.fit import fit_for_target

TIERS = ("seen_gpu", "size_extrapolation", "known_model", "unseen_gpu", "microbench")
_DS: Dataset | None = None
_FE: dict[tuple[str, str], dict[str, float]] = {}


def _init(results: str) -> None:
    global _DS, _FE
    _DS = Dataset.load(results)
    _FE = {
        tuple(k.split("|")): v
        for k, v in json.loads((Path(results) / "eval" / "frontend.json").read_text()).items()  # type: ignore[misc]
    }


def _run(job: tuple[str, str, str, bool]) -> dict[str, Any]:
    tier, model, gpu, gbm = job
    assert _DS is not None
    tables = list(_DS)
    fitted = fit_for_target(tables, model, gpu, tier)
    if fitted is None:
        return {"steps": [], "e2e": []}
    coef, train = fitted
    fold = Fold(tier, model, gpu, coef, train)
    return {
        "steps": evaluate_steps(_DS, [fold], with_gbm=gbm, keep_sample=400),
        "e2e": evaluate_e2e(_DS, [fold], _FE),
        "coef": {"tier": tier, "model": model, "gpu": gpu, **coef.to_dict()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--tiers", default=",".join(TIERS))
    args = ap.parse_args()
    out = Path(args.results) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = Dataset.load(args.results)
    print(f"{len(ds.tables)} pairs: {len(ds.models)} models x {len(ds.gpus)} GPUs")

    fe = frontend_table(ds)
    (out / "frontend.json").write_text(
        json.dumps({f"{m}|{g}": v for (m, g), v in sorted(fe.items())}, indent=1) + "\n"
    )
    (out / "memory.json").write_text(json.dumps(memory_lomo(ds), indent=1) + "\n")
    (out / "gemm_efficiency.json").write_text(
        json.dumps(implied_gemm_efficiency(ds), indent=1) + "\n"
    )
    (out / "transfer.json").write_text(json.dumps(transfer_attribution(ds), indent=1) + "\n")
    print(f"frontend + memory done in {time.time() - t0:.0f}s")

    for tier in args.tiers.split(","):
        jobs = [(tier, m, g, not args.no_gbm) for (m, g) in sorted(ds.tables)]
        steps: list[dict[str, Any]] = []
        e2e: list[dict[str, Any]] = []
        coefs: list[dict[str, Any]] = []
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.results,)) as ex:
            for res in ex.map(_run, jobs):
                steps += res["steps"]
                e2e += res["e2e"]
                if "coef" in res:
                    coefs.append(res["coef"])
        for name, rows in (("steps", steps), ("e2e", e2e), ("coef", coefs)):
            (out / f"{name}_{tier}.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(f"{tier}: {len(steps)} step folds, {len(e2e)} replays in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
