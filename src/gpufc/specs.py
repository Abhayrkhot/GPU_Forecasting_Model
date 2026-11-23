"""Model shapes (from Hugging Face ``config.json``) and GPU datasheet specs.

Everything the forecaster knows about a model or a GPU it has never run comes from here:
layer shapes for the FLOP and byte counts, and datasheet peaks for the roofline. Shapes
cover dense decoder-only transformers (Llama, Qwen2, Qwen3, Mistral) and sparse
mixture-of-experts variants (Qwen3-MoE, Qwen2-MoE style routed experts).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any

GIB = 1 << 30
GB = 10**9

DENSE_ARCHS = {"llama", "qwen2", "qwen3", "mistral"}
MOE_ARCHS = {"qwen3_moe", "qwen2_moe"}


@dataclass(frozen=True)
class Linear:
    """One weight GEMM in a layer: ``[tokens, k] @ [k, n]``. ``count`` layers have it."""

    name: str
    k: int
    n: int
    count: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_embeddings: bool = False
    dtype_bytes: int = 2
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    num_moe_layers: int = 0
    shared_expert_intermediate_size: int = 0
    max_position_embeddings: int = 32768

    # ---- construction -------------------------------------------------------------------

    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], name: str | None = None) -> ModelSpec:
        arch = str(cfg.get("model_type", "llama"))
        if arch not in DENSE_ARCHS | MOE_ARCHS:
            raise ValueError(
                f"unsupported model_type {arch!r}; supported: {sorted(DENSE_ARCHS | MOE_ARCHS)}"
            )
        heads = int(cfg["num_attention_heads"])
        hidden = int(cfg["hidden_size"])
        layers = int(cfg["num_hidden_layers"])
        dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16")
        experts = int(cfg.get("num_experts", 0) or 0) if arch in MOE_ARCHS else 0
        moe_layers = 0
        if experts:
            step = int(cfg.get("decoder_sparse_step", 1) or 1)
            dense_only = set(cfg.get("mlp_only_layers") or [])
            moe_layers = sum(
                1 for i in range(layers) if i not in dense_only and (i + 1) % step == 0
            )
        return cls(
            name=name or str(cfg.get("_name_or_path") or arch),
            arch=arch,
            num_layers=layers,
            hidden_size=hidden,
            num_heads=heads,
            num_kv_heads=int(cfg.get("num_key_value_heads") or heads),
            head_dim=int(cfg.get("head_dim") or hidden // heads),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            dtype_bytes=4 if "float32" in dtype else 2,
            num_experts=experts,
            experts_per_token=int(cfg.get("num_experts_per_tok", 0) or 0) if experts else 0,
            moe_intermediate_size=int(cfg.get("moe_intermediate_size", 0) or 0) if experts else 0,
            num_moe_layers=moe_layers,
            shared_expert_intermediate_size=int(cfg.get("shared_expert_intermediate_size", 0) or 0)
            if experts
            else 0,
            max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
        )

    @classmethod
    def from_json(cls, path: str | Path, name: str | None = None) -> ModelSpec:
        path = Path(path)
        d = json.loads(path.read_text())
        if "num_hidden_layers" not in d:  # a ModelSpec dump, not a HF config
            return cls(**d)
        return cls.from_hf_config(d, name=name or path.stem)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- shapes -------------------------------------------------------------------------

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def num_dense_layers(self) -> int:
        return self.num_layers - self.num_moe_layers

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def attention_linears(self) -> list[Linear]:
        h, n = self.hidden_size, self.num_layers
        return [
            Linear("qkv", h, self.q_dim + 2 * self.kv_dim, n),
            Linear("o", self.q_dim, h, n),
        ]

    def dense_mlp_linears(self) -> list[Linear]:
        """Dense MLP GEMMs (gate and up fused, as vLLM runs them)."""
        h, n, i = self.hidden_size, self.num_dense_layers, self.intermediate_size
        if n == 0:
            return []
        return [Linear("gate_up", h, 2 * i, n), Linear("down", i, h, n)]

    def moe_dense_linears(self) -> list[Linear]:
        """GEMMs every token runs in a MoE layer: the router and any shared expert."""
        if not self.is_moe:
            return []
        h, n = self.hidden_size, self.num_moe_layers
        out = [Linear("router", h, self.num_experts, n)]
        si = self.shared_expert_intermediate_size
        if si:
            out += [Linear("shared_gate_up", h, 2 * si, n), Linear("shared_down", si, h, n)]
        return out

    def dense_linears(self) -> list[Linear]:
        """Every GEMM whose weights all tokens share (everything but routed experts)."""
        return self.attention_linears() + self.dense_mlp_linears() + self.moe_dense_linears()

    @property
    def expert_params(self) -> int:
        """Parameters of one routed expert (gate, up, down)."""
        return 3 * self.hidden_size * self.moe_intermediate_size

    @property
    def dense_params(self) -> int:
        return sum(lin.k * lin.n * lin.count for lin in self.dense_linears())

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def num_params(self) -> int:
        routed = self.num_moe_layers * self.num_experts * self.expert_params
        lm_head = 0 if self.tie_embeddings else self.embed_params
        return self.dense_params + routed + self.embed_params + lm_head

    @property
    def active_params(self) -> int:
        """Parameters one token touches (excluding the embedding lookup)."""
        routed = self.num_moe_layers * self.experts_per_token * self.expert_params
        return self.dense_params + routed + self.embed_params

    @property
    def weight_bytes(self) -> int:
        return self.num_params * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V for every layer, one token."""
        return 2 * self.num_layers * self.kv_dim * self.dtype_bytes

    def expected_experts(self, tokens: float) -> float:
        """Distinct routed experts per MoE layer that ``tokens`` tokens touch.

        Assumes each token picks ``experts_per_token`` experts uniformly at random, the
        balance the router's auxiliary loss trains towards.
        """
        if not self.is_moe or tokens <= 0:
            return 0.0
        e, k = self.num_experts, self.experts_per_token
        return float(e * (1.0 - (1.0 - k / e) ** tokens))


@dataclass(frozen=True)
class GPUSpec:
    """Datasheet numbers for one GPU (dense tensor-core throughput, no sparsity).

    ``measured_*`` fields are filled by ``gpufc probe-gpu`` microbenchmarks (a large GEMM
    and a device-to-device copy) and, when present, replace the datasheet peaks for the
    ``microbench`` forecasting tier.
    """

    name: str
    arch: str
    compute_capability: float
    sm_count: int
    memory_bytes: int
    mem_bandwidth: float  # bytes/s
    bf16_flops: float  # dense FLOP/s, 0 if unsupported
    fp16_flops: float
    l2_bytes: int
    form_factor: str = ""
    measured_flops: float = 0.0
    measured_bandwidth: float = 0.0
    aliases: tuple[str, ...] = field(default=(), compare=False)

    def peak_flops(self, dtype_bytes: int = 2, bf16: bool = True, measured: bool = False) -> float:
        if measured and self.measured_flops:
            return self.measured_flops
        if dtype_bytes == 4:
            return self.fp16_flops / 8  # TF32-class, rarely served; rough
        return self.bf16_flops if bf16 and self.bf16_flops else self.fp16_flops

    def bandwidth(self, measured: bool = False) -> float:
        if measured and self.measured_bandwidth:
            return self.measured_bandwidth
        return self.mem_bandwidth

    @property
    def supports_bf16(self) -> bool:
        return self.bf16_flops > 0

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte at which the datasheet roofline turns compute-bound."""
        return self.peak_flops() / self.mem_bandwidth

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aliases"] = list(self.aliases)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPUSpec:
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["aliases"] = tuple(d.get("aliases", ()))
        return cls(**d)

    def with_measurements(self, flops: float, bandwidth: float) -> GPUSpec:
        return replace(self, measured_flops=flops, measured_bandwidth=bandwidth)


def _tf(x: float) -> float:
    return x * 1e12


# Datasheet values from NVIDIA product briefs and architecture whitepapers: dense
# tensor-core throughput (no structured sparsity) for bf16 and fp16 with FP32 accumulate.
# Memory is the total nvidia-smi reports (what vLLM sizes its KV cache from), in MiB.
# name, arch, compute capability, SMs, MiB, bytes/s, bf16 TFLOP/s, fp16 TFLOP/s, L2 MiB,
# form factor, nvidia-smi names
# fmt: off
_CATALOG: tuple[tuple[Any, ...], ...] = (
    ("h200", "hopper", 9.0, 132, 143771, 4.8e12, 989, 989, 50, "sxm", ("nvidia h200",)),
    ("h200-nvl", "hopper", 9.0, 132, 143771, 4.8e12, 835, 835, 50, "pcie",
     ("nvidia h200 nvl",)),
    ("h100-sxm", "hopper", 9.0, 132, 81559, 3.35e12, 989, 989, 50, "sxm",
     ("nvidia h100 80gb hbm3",)),
    ("h100-pcie", "hopper", 9.0, 114, 81559, 2.0e12, 756, 756, 50, "pcie",
     ("nvidia h100 pcie",)),
    ("a100-80g", "ampere", 8.0, 108, 81920, 2.039e12, 312, 312, 40, "sxm",
     ("nvidia a100-sxm4-80gb",)),
    ("a100-80g-pcie", "ampere", 8.0, 108, 81920, 1.935e12, 312, 312, 40, "pcie",
     ("nvidia a100 80gb pcie",)),
    ("a100-40g", "ampere", 8.0, 108, 40960, 1.555e12, 312, 312, 40, "sxm",
     ("nvidia a100-sxm4-40gb", "nvidia a100-pcie-40gb")),
    ("a30", "ampere", 8.0, 56, 24576, 0.933e12, 165, 165, 24, "pcie", ("nvidia a30",)),
    ("l40s", "ada", 8.9, 142, 46068, 0.864e12, 362, 362, 96, "pcie", ("nvidia l40s",)),
    ("l40", "ada", 8.9, 142, 46068, 0.864e12, 181, 181, 96, "pcie", ("nvidia l40",)),
    ("a6000", "ampere", 8.6, 84, 49140, 0.768e12, 154.8, 154.8, 6, "pcie",
     ("nvidia rtx a6000",)),
    ("a5000", "ampere", 8.6, 64, 24564, 0.768e12, 111.1, 111.1, 6, "pcie",
     ("nvidia rtx a5000",)),
    ("t4", "turing", 7.5, 40, 15360, 0.32e12, 0, 65, 4, "pcie", ("tesla t4",)),
    ("b200", "blackwell", 10.0, 148, 183359, 8.0e12, 2250, 2250, 126, "sxm",
     ("nvidia b200",)),
)
# fmt: on
GPUS: dict[str, GPUSpec] = {
    r[0]: GPUSpec(
        r[0],
        r[1],
        r[2],
        r[3],
        r[4] << 20,
        r[5],
        _tf(r[6]),
        _tf(r[7]),
        r[8] << 20,
        r[9],
        aliases=r[10],
    )
    for r in _CATALOG
}


def get_gpu(name: str) -> GPUSpec:
    """Look up a GPU by catalog key or by the name ``nvidia-smi`` reports."""
    key = name.strip().lower()
    if key in GPUS:
        return GPUS[key]
    for g in GPUS.values():
        if key in g.aliases:
            return g
    path = Path(name)
    if path.suffix == ".json" and path.exists():
        return GPUSpec.from_dict(json.loads(path.read_text()))
    raise ValueError(f"unknown GPU {name!r}; known: {sorted(GPUS)} (or pass a spec .json)")


def builtin_models() -> dict[str, ModelSpec]:
    """Model shapes shipped with the package (``gpufc/data/models/*.json`` HF configs)."""
    out = {}
    root = resources.files("gpufc") / "data" / "models"
    for entry in root.iterdir():
        if entry.name.endswith(".json"):
            name = entry.name[: -len(".json")]
            out[name] = ModelSpec.from_hf_config(json.loads(entry.read_text()), name=name)
    return out


def get_model(name: str) -> ModelSpec:
    """A built-in model by short name (``qwen3-8b``) or HF id, or a ``config.json`` path."""
    path = Path(name)
    if path.suffix == ".json" and path.exists():
        return ModelSpec.from_json(path)
    models = builtin_models()
    key = name.strip().lower().split("/")[-1].removesuffix("-instruct")
    if key in models:
        return models[key]
    raise ValueError(f"unknown model {name!r}; known: {sorted(models)} (or pass a config.json)")
