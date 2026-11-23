from __future__ import annotations

import json
import math

import pytest
from batchsim.cost import BatchStats

from gpufc.cost import TERMS, ForecastStepModel, TermCalculator, pad_tokens
from gpufc.specs import GPUS, GPUSpec, ModelSpec, builtin_models, get_gpu, get_model

from .conftest import TRUE_COEF


@pytest.mark.parametrize(
    ("name", "params_b", "kv_kib"),
    [
        ("qwen2.5-0.5b", 0.494, 12.0),
        ("qwen2.5-7b", 7.62, 56.0),
        ("qwen3-8b", 8.19, 144.0),
        ("qwen2.5-32b", 32.76, 256.0),
        ("qwen3-30b-a3b", 30.53, 96.0),
    ],
)
def test_param_counts_match_published(name: str, params_b: float, kv_kib: float) -> None:
    m = get_model(name)
    assert m.num_params / 1e9 == pytest.approx(params_b, rel=0.01)
    assert m.kv_bytes_per_token / 1024 == pytest.approx(kv_kib)


def test_moe_shapes() -> None:
    m = get_model("Qwen/Qwen3-30B-A3B")
    assert m.is_moe and m.num_moe_layers == 48 and m.num_dense_layers == 0
    assert m.dense_mlp_linears() == []
    # ~3.3B active parameters including embeddings (as published), excluding untied head.
    assert (m.active_params + m.embed_params) / 1e9 == pytest.approx(3.35, rel=0.02)
    assert m.expected_experts(0) == 0.0
    assert m.expected_experts(1) == pytest.approx(m.experts_per_token)
    assert m.expected_experts(10_000) == pytest.approx(m.num_experts, rel=1e-6)
    assert m.expected_experts(16) < m.expected_experts(64)


def test_model_lookup_and_json(tmp_path: object) -> None:
    assert get_model("Qwen/Qwen2.5-7B-Instruct").name == "qwen2.5-7b"
    assert len(builtin_models()) == 13
    with pytest.raises(ValueError, match="unknown model"):
        get_model("nope-1b")
    with pytest.raises(ValueError, match="unsupported model_type"):
        ModelSpec.from_hf_config({"model_type": "gpt2", "num_attention_heads": 1})


def test_model_from_hf_config_file(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "torch_dtype": "bfloat16",
    }
    p = tmp_path / "llama-3.1-8b.json"  # type: ignore[operator]
    p.write_text(json.dumps(cfg))
    m = get_model(str(p))
    assert m.num_params / 1e9 == pytest.approx(8.03, rel=0.01)
    assert ModelSpec.from_json(p).head_dim == 128
    dump = tmp_path / "dump.json"  # type: ignore[operator]
    dump.write_text(json.dumps(m.to_dict()))
    assert ModelSpec.from_json(dump) == m


def test_gpu_lookup() -> None:
    assert get_gpu("NVIDIA A100-SXM4-80GB").name == "a100-80g"
    assert get_gpu("NVIDIA RTX A6000").name == "a6000"
    assert get_gpu("Tesla T4").supports_bf16 is False
    assert get_gpu("t4").peak_flops(2, bf16=True) == pytest.approx(65e12)
    h = get_gpu("h200")
    assert h.ridge_point == pytest.approx(989e12 / 4.8e12)
    m = h.with_measurements(7.8e14, 4.3e12)
    assert m.peak_flops(measured=True) == 7.8e14 and m.bandwidth(measured=True) == 4.3e12
    assert m.peak_flops() == h.peak_flops()
    assert GPUSpec.from_dict(m.to_dict()) == m
    with pytest.raises(ValueError, match="unknown GPU"):
        get_gpu("voodoo2")
    assert all(g.memory_bytes > 0 for g in GPUS.values())


def test_pad_tokens() -> None:
    sizes = (1, 2, 4, 8, 16)
    assert pad_tokens(3, sizes) == 4
    assert pad_tokens(16, sizes) == 16
    assert pad_tokens(17, sizes) == 17
    assert pad_tokens(5, ()) == 5


def _decode(b: int, ctx: int) -> BatchStats:
    s = BatchStats()
    for _ in range(b):
        s.add(1, ctx)
    return s


def test_terms_physics() -> None:
    calc = TermCalculator(get_model("qwen3-8b"), get_gpu("h200"))
    t = dict(zip(TERMS, calc.terms(_decode(1, 100))))
    m = get_model("qwen3-8b")
    # one decode token: GEMMs (incl. the vocabulary projection) read every weight once
    lm_head_s = m.embed_params * 2 / 4.8e12
    weight_s = m.dense_params * 2 / 4.8e12 + lm_head_s
    assert t["gemm_max"] == pytest.approx(weight_s, rel=0.01)
    assert t["gemm_min"] < 0.01 * t["gemm_max"]
    assert t["attn_prefill"] == 0 and t["piecewise_layers"] == 0 and t["eager_layers"] == 0
    assert t["moe_max"] == 0
    big = BatchStats()
    big.add(8192, 0)
    tb = dict(zip(TERMS, calc.terms(big)))
    # 8192 tokens: compute bound; compute time ~ 2 * tokens * params / peak
    # (one sequence, so its vocabulary projection is still a weight read)
    assert tb["gemm_max"] == pytest.approx(2 * 8192 * m.dense_params / 989e12 + lm_head_s, rel=0.01)
    assert tb["eager_layers"] > 0 and tb["attn_prefill"] > 0
    # sustained tensor-core load: none for a decode step, linear past 50 ms of compute
    assert t["gemm_sustained"] < 1e-4 * t["gemm_max"]
    cmp = 2 * 8192 * m.dense_params / 989e12
    assert tb["gemm_sustained"] == pytest.approx(cmp * min(1, cmp / 0.05), rel=0.01)
    mixed = _decode(4, 100)
    mixed.add(100, 0)
    assert dict(zip(TERMS, calc.terms(mixed)))["piecewise_layers"] > 0
    # decode attention scales with context
    a = dict(zip(TERMS, calc.terms(_decode(8, 1000))))["attn_decode"]
    b = dict(zip(TERMS, calc.terms(_decode(8, 2000))))["attn_decode"]
    assert b == pytest.approx(2 * a, rel=0.01)
    assert calc.roofline_time(big) > calc.roofline_time(_decode(1, 100))


def test_moe_terms_grow_with_tokens() -> None:
    calc = TermCalculator(get_model("qwen3-30b-a3b"), get_gpu("h200"))
    one = dict(zip(TERMS, calc.terms(_decode(1, 100))))["moe_max"]
    many = dict(zip(TERMS, calc.terms(_decode(64, 100))))["moe_max"]
    assert 0 < one < many


def test_forecast_model_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    sm = ForecastStepModel(
        get_model("qwen3-4b"),
        get_gpu("a100-80g"),
        dict(TRUE_COEF),
        log_interval={"lo90": -0.1, "hi90": 0.2},
    )
    s = _decode(16, 512)
    t = sm.step_time(s)
    assert t > 0
    assert sum(sm.breakdown(s).values()) == pytest.approx(t)
    assert sm.predict([s, s]).tolist() == pytest.approx([t, t])
    assert sm.scaled(2.0).step_time(s) == pytest.approx(2 * t)
    lo, hi = sm.interval("90")
    assert lo == pytest.approx(math.exp(-0.1)) and hi == pytest.approx(math.exp(0.2))
    with pytest.raises(KeyError):
        sm.interval("99")
    p = tmp_path / "m.json"  # type: ignore[operator]
    sm.save(p)
    back = ForecastStepModel.load(p)
    assert back.step_time(s) == pytest.approx(t)
    with pytest.raises(ValueError, match="missing coefficients"):
        ForecastStepModel(get_model("qwen3-4b"), get_gpu("h200"), {"const": 1.0})
