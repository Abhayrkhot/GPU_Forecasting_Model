from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpufc.data import Dataset, StepTable, parse_server_log
from gpufc.evaluate import (
    evaluate_e2e,
    evaluate_steps,
    folds,
    frontend_table,
    memory_lomo,
    memory_samples,
    summarize_e2e,
    transfer_attribution,
)
from gpufc.fit import (
    Coef,
    design,
    fit_coef,
    fit_for_target,
    fit_gpu,
    fit_tier,
    interval_from_residuals,
    pooled_prior,
    predict,
    select_training,
    shrink_pool,
)
from gpufc.memory import MemoryModel, fit_memory
from gpufc.specs import get_gpu, get_model

from .conftest import GPUS, MODELS, NOISE, TRUE_COEF, TRUE_FRONTEND, TRUE_MEMORY


def test_dataset_loads(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    assert len(ds.tables) == len(MODELS) * len(GPUS)
    assert set(ds.gpus) == set(GPUS)
    assert ds.models[0] == "qwen2.5-0.5b"  # sorted by size
    t = ds.tables[("qwen3-0.6b", "h200")]
    assert len(t.profile()) == 300 and len(t.validation()) == 240
    assert t.gpu.measured_flops > 0  # microbenchmark attached
    assert set(ds.replay_paths("qwen3-0.6b", "h200")) == {"profile", "conv_r0.5", "conv_r2"}
    assert len(ds.pairs(gpu="h200")) == len(MODELS)


def test_step_table_roundtrip(tmp_path: Path, synthetic_results: Path) -> None:
    t = Dataset.load(synthetic_results).tables[("qwen2.5-1.5b", "a6000")]
    p = tmp_path / "x.csv.gz"
    t.to_csv_gz(p)
    back = StepTable.from_csv_gz(p, t.model, t.gpu)
    assert len(back) == len(t)
    assert back.y == pytest.approx(t.y, rel=1e-5)
    assert back.stats[5].prefill_attn == pytest.approx(t.stats[5].prefill_attn, abs=1)
    assert len(t.subsample(10)) == 10 and len(t.subsample(10_000)) == len(t)


def test_fit_recovers_truth_and_transfers(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    tables = list(ds)
    train = select_training(tables, "qwen3-1.7b", "a6000", "unseen_gpu")
    assert all(t.gpu.name != "a6000" and t.model.name != "qwen3-1.7b" for t in train)
    coef = fit_coef(design(train))
    target = ds.tables[("qwen3-1.7b", "a6000")]
    ape = np.abs(predict(coef, target) - target.y) / target.y
    assert np.median(ape) < 2 * NOISE  # unseen model on unseen GPU, noise-level error
    # dense-only training data says nothing about MoE terms: they keep the prior
    from gpufc.workloads import PRIOR_COEF

    assert coef.values["moe_max"] == pytest.approx(PRIOR_COEF["moe_max"])
    assert coef.values["attn_decode"] == pytest.approx(TRUE_COEF["attn_decode"], rel=0.15)
    back = Coef.from_dict(coef.to_dict())
    assert back.values == coef.values
    with pytest.raises(ValueError, match="unknown tier"):
        select_training(tables, "a", "b", "nope")
    with pytest.raises(ValueError, match="no training steps"):
        design([])


def test_conformal_interval_covers(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    train = [t for t in ds if t.gpu.name == "h200" and t.model.name != "qwen3-1.7b"]
    coef = fit_tier(train)
    lo, hi = coef.log_interval["lo90"], coef.log_interval["hi90"]
    assert lo < 0 < hi
    target = ds.tables[("qwen3-1.7b", "h200")]
    r = np.log(target.y / predict(coef, target))
    assert 0.75 < ((r >= lo) & (r <= hi)).mean() <= 1.0
    assert interval_from_residuals(np.zeros(0)) == {}
    sm = coef.bind(target.model, target.gpu, tier="seen_gpu")
    assert sm.meta["tier"] == "seen_gpu" and sm.interval("90")[0] < 1


def test_per_gpu_fit_shrinks_to_pooled(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    tables = list(ds)
    pool = shrink_pool(tables, "qwen3-1.7b", "seen_gpu")
    assert pool is not None and all(t.model.name != "qwen3-1.7b" for t in pool)
    assert shrink_pool(tables, "qwen3-1.7b", "unseen_gpu") is None
    small = shrink_pool(tables, "qwen3-1.7b", "size_extrapolation")
    assert small is not None and len(small) == len(tables)  # every synthetic model is small
    # A very strong ridge pins a per-GPU fit to the pooled weights.
    prior = pooled_prior(pool)
    two = [t for t in pool if t.gpu.name == "h200"][:2]
    pinned = fit_coef(design(two), prior=prior, ridge=1e4)
    assert all(abs(pinned.values[k] - v) <= 0.02 * max(v, 0.1) for k, v in prior.items())
    # Two profiled models are enough to cross-validate the shrinkage and attach intervals.
    coef = fit_gpu(two, pool)
    assert coef.log_interval and "cv_tw_err_ridge_1" in coef.report
    target = ds.tables[("qwen3-1.7b", "h200")].validation()
    assert np.median(np.abs(predict(coef, target) / target.y - 1)) < 2 * NOISE
    fitted = fit_for_target(tables, "qwen3-1.7b", "h200", "seen_gpu")
    assert fitted is not None and fitted[0].report["pool_pairs"] == len(pool)
    assert fit_for_target(tables[:1], "qwen3-1.7b", "h200", "seen_gpu") is None


def test_memory_model() -> None:
    from gpufc.memory import MemorySample

    samples = []
    for g in ("h200", "a100-80g", "a6000"):
        for m in ("qwen2.5-0.5b", "qwen3-1.7b", "qwen3-8b", "qwen2.5-7b"):
            mm, gg = get_model(m), get_gpu(g)
            for bt, ns in ((2048, 256), (8192, 1024)):
                kv = TRUE_MEMORY.kv_tokens(mm, gg, bt, ns)
                samples.append(MemorySample(mm, gg, bt, ns, kv))
    fit = fit_memory(samples)
    for k, v in TRUE_MEMORY.coef.items():
        assert fit.coef[k] == pytest.approx(v, rel=0.05, abs=0.05)
    assert MemoryModel.from_dict(fit.to_dict()).coef == fit.coef
    assert fit.kv_tokens(get_model("qwen2.5-32b"), get_gpu("t4"), 2048, 256) == 0


def test_parse_server_log() -> None:
    text = (
        "INFO Model loading took 0.93 GiB memory and 4.8 seconds\n"
        "INFO Available KV cache memory: 122.1 GiB\n"
        "INFO Graph capturing finished in 3 secs, took 0.34 GiB\n"
        "INFO GPU KV cache size: 10,629,120 tokens, Maximum concurrency ...\n"
        "INFO Initializing a V1 LLM engine (v0.34.0) with config\n"
        "INFO Using FLASH_ATTN backend out of ...\n"
    )
    d = parse_server_log(text)
    assert d == {
        "kv_tokens": 10629120,
        "weights_gib": 0.93,
        "available_kv_gib": 122.1,
        "cudagraph_gib": 0.34,
        "vllm_version": "0.34.0",
        "attention_backend": "FLASH_ATTN",
    }


def test_evaluation_protocols(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    lomo = folds(ds, "seen_gpu", intervals=False)
    assert len(lomo) == len(MODELS) * len(GPUS)
    rows = evaluate_steps(ds, lomo, with_gbm=True)
    med = np.median([r["gpufc"]["median_ape"] for r in rows])
    assert med < 2 * NOISE
    assert all({"roofline", "gbm", "oracle"} <= set(r) for r in rows)
    logo = folds(ds, "microbench", intervals=False)
    rows = evaluate_steps(ds, logo[:3], with_gbm=False)
    assert all(r["tier"] == "microbench" for r in rows)
    mem = memory_lomo(ds)
    assert len(mem) == len(memory_samples(ds))
    # Every GPU shares the true weights, so a fit on the others transfers without bias.
    tr = transfer_attribution(ds, max_eval_steps=1000)
    assert set(tr) == set(GPUS)
    assert all(abs(v["bias"]) < 2 * NOISE and "gemm_max" in v["swap"] for v in tr.values())
    assert max(abs(r["rel_err"]) for r in mem) < 0.02


def test_e2e(synthetic_results: Path) -> None:
    ds = Dataset.load(synthetic_results)
    fe = frontend_table(ds)
    assert len(fe) == len(ds.tables)
    # Requests that find the engine idle see frontend + wake-up latency; the synthetic
    # replays have only the former, so the two estimates must add up to it.
    est = np.median([v["frontend_latency"] + v["idle_launch_latency"] for v in fe.values()])
    assert est == pytest.approx(TRUE_FRONTEND, abs=0.002)
    fl = [f for f in folds(ds, "seen_gpu") if f.gpu == "h200"][:2]
    rows = evaluate_e2e(ds, fl, fe)
    assert rows and all("gpufc_slow" in r for r in rows)
    s = summarize_e2e(rows, "gpufc")
    assert s["tpot_p50"] < 0.1
    assert summarize_e2e(rows, "missing") == {}


def _write_raw_run(d: Path) -> None:
    """A tiny raw run directory in the layout bench/run_combo.sh writes."""
    import json as _json

    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        _json.dumps(
            {
                "model": "qwen3-0.6b",
                "gpu_name": "NVIDIA H200",
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 1024,
                "max_model_len": 8192,
            }
        )
    )
    (d / "server.log").write_text("INFO GPU KV cache size: 1,000,000 tokens\n")
    (d / "gpu.json").write_text(_json.dumps({"gemm_flops": 7.8e14, "copy_bandwidth": 4.2e12}))
    (d / "validation_plan.json").write_text(_json.dumps([{"name": "conv_r1"}]))
    t = 100.0
    steps, runs = [], []
    for i, run in enumerate(("profile", "conv_r1")):
        start = t
        for k in range(50):
            p = [[512, 0]] if k % 5 == 0 else []
            steps.append(
                {
                    "i": len(steps),
                    "cpu": t,
                    "g0": t,
                    "g1": t + 0.004,
                    "T": 8 + 512 * (k % 5 == 0),
                    "S": 8 + (k % 5 == 0),
                    "D": 8,
                    "Dc": 8000,
                    "p": p,
                }
            )
            t += 0.005
        runs.append({"run": run, "start": start - 0.001, "end": t})
        t += 1.0
        recs = [
            {
                "i": j,
                "send": start + 0.01 * j,
                "prompt_len": 100,
                "output_len": 3,
                "tok": [0.02, 0.03, 0.04],
            }
            for j in range(10)
        ]
        recs.append(
            {"i": 10, "send": start, "prompt_len": 100, "output_len": 3, "tok": [], "error": "boom"}
        )
        (d / f"{run}.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in recs) + "\n" + _json.dumps({"summary": {}}) + "\n"
        )
        del i
    (d / "steps.jsonl").write_text("\n".join(_json.dumps(s) for s in steps) + "\n")
    (d / "runs.jsonl").write_text("\n".join(_json.dumps(r) for r in runs) + "\n")


def test_compact_raw_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from batchsim.validation import load_replay

    from gpufc.cli import main
    from gpufc.data import compact_run, load_raw_run

    raw = tmp_path / "raw" / "h200" / "qwen3-0.6b" / "1-000000"
    _write_raw_run(raw)
    info, table = load_raw_run(raw)
    assert info.gpu == "h200" and info.kv_tokens == 1_000_000
    assert len(table) == 100 and set(table.run) == {"profile", "conv_r1"}
    assert len(table.profile()) == 50 and len(table.validation()) == 50
    # back-to-back steps: period = end-to-end distance (5 ms), not the 4 ms GPU span
    assert np.median(table.y) == pytest.approx(0.005, rel=1e-3)
    assert main(["compact", str(raw), "--results", str(tmp_path / "results")]) == 0
    assert "qwen3-0.6b" in capsys.readouterr().out
    ds = Dataset.load(tmp_path / "results")
    assert ds.infos[("qwen3-0.6b", "h200")].gpu_spec().measured_flops == 7.8e14
    meas = load_replay(ds.replay_paths("qwen3-0.6b", "h200")["conv_r1"])
    assert len(meas.send) == 10 and meas.errors == 1
    compact_run(raw, tmp_path / "results2")
    assert (tmp_path / "results2" / "replays" / "h200" / "qwen3-0.6b" / "profile.jsonl.gz").exists()


def test_known_model_tier_and_bounds(synthetic_results: Path) -> None:
    from gpufc.fit import MIN_INV_EFFICIENCY, ROOFLINE_TERMS

    ds = Dataset.load(synthetic_results)
    tables = list(ds)
    train = select_training(tables, "qwen3-1.7b", "a6000", "known_model")
    assert any(t.model.name == "qwen3-1.7b" for t in train)
    assert all(t.gpu.name != "a6000" for t in train)
    big = select_training(tables, "qwen3-1.7b", "h200", "size_extrapolation")
    assert big == []  # the target is not larger than the training cut-off
    # Even fitted on a single pair, no roofline weight claims more than 1/0.8 of peak.
    coef = fit_coef(design(tables[:1]), ridge=0.0)
    assert all(coef.values[t] >= MIN_INV_EFFICIENCY - 1e-12 for t in ROOFLINE_TERMS)
    tiered = fit_tier(train, measured_peaks=True)
    assert tiered.measured_peaks and "cv_tw_err_ridge_0.001" in tiered.report
