from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from batchsim.metrics import SLO
from batchsim.workload import LogNormal, synthetic

from gpufc.bundle import fit_bundle
from gpufc.cli import main
from gpufc.serving import Bundle, forecast, max_rate, plan
from gpufc.specs import get_gpu, get_model
from gpufc.workloads import (
    ServeLimits,
    profile_workload,
    saturation_rate,
    smoke_workload,
    validation_workload,
    vllm_default_limits,
)

from .conftest import TRUE_COEF, true_model


@pytest.fixture(scope="module")
def bundle_path(synthetic_results: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    b = fit_bundle(synthetic_results)
    p = tmp_path_factory.mktemp("bundle") / "bundle.json"
    b.save(p)
    return p


def test_vllm_default_limits() -> None:
    assert vllm_default_limits(get_gpu("h200")) == (8192, 1024)
    assert vllm_default_limits(get_gpu("a100-80g")) == (2048, 256)
    assert vllm_default_limits(get_gpu("a6000")) == (2048, 256)


def test_profile_workload_shapes() -> None:
    m, g = get_model("qwen3-8b"), get_gpu("a6000")
    lim = ServeLimits(100_000, 256, 2048, 8192)
    wl = profile_workload(m, g, lim)
    arr = np.array([s.arrival for s in wl.specs])
    assert (np.diff(arr) >= 0).all()
    assert (wl.prompt_lens + wl.output_lens <= 8192).all()
    # decode ladder present: many simultaneous identical requests with 128 outputs
    _, counts = np.unique(arr[wl.output_lens == 128], return_counts=True)
    assert counts.max() >= 64
    # isolated prefills and a saturated burst present
    assert (wl.output_lens == 4).sum() == 32 and (wl.output_lens == 8).sum() >= 30
    assert profile_workload(m, g, lim).specs == wl.specs  # deterministic
    assert len(smoke_workload()) == 24


def test_validation_workload_duration() -> None:
    base = synthetic(5000, 1.0, LogNormal(500, 0.5, 16, 2000), LogNormal(50, 0.5, 1, 200), seed=1)
    fast = validation_workload(base, "conv", 10.0)
    assert len(fast) == 900 and fast.name == "conv_r10"
    slow = validation_workload(base, "code", 0.1)
    assert len(slow) == 24  # capped near 240 s instead of 60 requests over 600 s
    tiny = validation_workload(base, "code", 0.01)
    assert len(tiny) == 20


def test_saturation_and_forecast() -> None:
    sm = true_model("qwen3-1.7b", "a6000")
    lim = ServeLimits(200_000, 256, 2048, 4096)
    base = synthetic(400, 1.0, LogNormal(800, 0.8, 16, 3000), LogNormal(100, 0.8, 4, 400), seed=3)
    sat = saturation_rate(base, sm, lim, hi=200.0)
    assert sat > 0
    from batchsim.engine import EngineConfig

    eng = EngineConfig(True, 0.004, 0.0, record_steps=False)
    low = forecast(base.scale_to_rate(0.3 * sat), sm, lim, eng)
    high = forecast(base.scale_to_rate(1.5 * sat), sm, lim, eng)
    assert high["ttft_p90"] > low["ttft_p90"]
    cap = max_rate(base, sm, lim, eng, SLO(ttft=2.0), hi=200.0)
    assert cap == pytest.approx(sat, rel=0.1)


def test_bundle(bundle_path: Path) -> None:
    b = Bundle.load(bundle_path)
    assert set(b.per_gpu) == {"h200", "a100-80g", "a6000"}
    assert b.pooled_microbench is not None and b.pooled_microbench.measured_peaks
    m = get_model("qwen3-1.7b")
    assert b.step_model(m, get_gpu("h200")).meta["tier"] == "seen_gpu"
    assert b.step_model(m, get_gpu("l40s")).meta["tier"] == "unseen_gpu"
    assert b.step_model(m, b.gpu(get_gpu("h200")), "microbench").measured_peaks
    with pytest.raises(ValueError, match="microbench tier"):
        b.step_model(m, get_gpu("l40s"), "microbench")
    # Well-identified weights come back (the per-prefill-chunk terms are collinear with
    # each other in this data; only their sum is identified, which prediction tests cover).
    for k in ("const", "gemm_max", "attn_decode"):
        assert b.pooled.values[k] == pytest.approx(TRUE_COEF[k], rel=0.25), k
    assert b.frontend_latency + b.idle_launch_latency == pytest.approx(0.004, abs=0.002)
    lim = b.limits(get_model("qwen2.5-32b"), get_gpu("a6000"))
    assert lim.kv_tokens == 0
    assert Bundle.from_dict(json.loads(bundle_path.read_text())).memory.coef == b.memory.coef


def test_plan(bundle_path: Path) -> None:
    b = Bundle.load(bundle_path)
    base = synthetic(300, 1.0, LogNormal(500, 0.8, 16, 2000), LogNormal(80, 0.8, 4, 300), seed=2)
    rows = plan(
        b,
        get_model("qwen3-8b"),
        base,
        20.0,
        SLO(ttft=1.0),
        ["h200", "a6000", "t4"],
        prices={"h200": 4.0},
    )
    by = {r.gpu: r for r in rows}
    assert not by["t4"].fits
    assert by["h200"].rate_per_gpu > by["a6000"].rate_per_gpu > 0
    assert by["h200"].rate_low <= by["h200"].rate_per_gpu <= by["h200"].rate_high
    assert by["h200"].gpus_needed >= 1 and by["h200"].cost_per_hour is not None
    assert by["a6000"].cost_per_hour is None


def test_cli(bundle_path: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["models"]) == 0
    assert "qwen3-30b-a3b" in capsys.readouterr().out
    assert main(["gpus"]) == 0
    assert "h200" in capsys.readouterr().out
    bp = str(bundle_path)
    assert (
        main(
            [
                "explain",
                "--model",
                "qwen3-8b",
                "--gpu",
                "l40s",
                "--bundle",
                bp,
                "--decode",
                "32",
                "--context",
                "2048",
                "--prefill",
                "512",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "unseen_gpu" in out and "gemm_max" in out
    assert (
        main(
            [
                "predict",
                "--model",
                "qwen3-1.7b",
                "--gpu",
                "a6000",
                "--bundle",
                bp,
                "--rate",
                "2",
                "--requests",
                "200",
            ]
        )
        == 0
    )
    assert "ttft_p90" in capsys.readouterr().out
    assert (
        main(
            [
                "predict",
                "--model",
                "qwen3-1.7b",
                "--gpu",
                "a6000",
                "--bundle",
                bp,
                "--rate",
                "2",
                "--requests",
                "100",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["tier"] == "seen_gpu"
    assert (
        main(
            [
                "capacity",
                "--model",
                "qwen3-1.7b",
                "--gpu",
                "h200",
                "--bundle",
                bp,
                "--requests",
                "200",
                "--slo-ttft",
                "0.5",
            ]
        )
        == 0
    )
    assert "req/s" in capsys.readouterr().out
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"h200": 3.5}))
    assert (
        main(
            [
                "plan",
                "--model",
                "qwen3-4b",
                "--rate",
                "10",
                "--bundle",
                bp,
                "--gpus",
                "h200,a6000,t4",
                "--requests",
                "200",
                "--prices",
                str(prices),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "does not fit" not in out.split("t4")[0] and "h200" in out
    assert (
        main(
            [
                "plan",
                "--model",
                "qwen3-4b",
                "--rate",
                "10",
                "--bundle",
                bp,
                "--gpus",
                "h200",
                "--requests",
                "100",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["gpu"] == "h200"
    with pytest.raises(SystemExit):
        main(["explain", "--model", "qwen3-8b", "--gpu", "h200", "--bundle", bp])
    with pytest.raises(SystemExit):
        main(["predict", "--model", "qwen3-8b", "--gpu", "h200", "--bundle", "/nope.json"])


def test_cli_fit_and_compact(
    synthetic_results: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "b.json"
    assert main(["fit", "--results", str(synthetic_results), "--out", str(out)]) == 0
    assert out.exists() and "per-GPU" in capsys.readouterr().out
