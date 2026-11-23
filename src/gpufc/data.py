"""Measured runs: loading raw GPU run directories and the compact tables kept in ``results/``.

A raw run directory (``bench/run_combo.sh``) holds the vLLM step probe log, the client
records of every replay, the server log and a GPU microbenchmark. :func:`compact_run`
reduces it to what the forecaster and the evaluation need:

``results/steps/<gpu>/<model>.csv.gz``
    One row per engine step: which replay it belongs to, its batch shape and its measured
    period (seconds from the end of the previous back-to-back step, see
    :func:`batchsim.calibrate.periods`).
``results/runs/<gpu>/<model>.json``
    Run metadata, server facts parsed from the log (KV cache size, memory), the
    microbenchmark and the validation plan.
``results/replays/<gpu>/<model>/<run>.jsonl.gz``
    Per-request send time, lengths and first/last token offsets (batchsim's compact
    replay format), for end-to-end validation.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from batchsim.calibrate import exists, load_probe, open_text, periods
from batchsim.cost import BatchStats
from numpy.typing import NDArray

from gpufc.specs import GPUSpec, ModelSpec, get_gpu, get_model

STEP_COLUMNS = ("run", "T", "S", "D", "Dc", "P", "PT", "PA", "y")
FloatArray = NDArray[np.float64]


def gpu_key(name: str) -> str:
    """Catalog key for an ``nvidia-smi`` GPU name (``NVIDIA A100-SXM4-80GB`` -> ``a100-80g``)."""
    return get_gpu(name).name


def parse_server_log(text: str) -> dict[str, Any]:
    """Facts vLLM logs at startup: KV cache size, memory split, CUDA graph memory."""
    out: dict[str, Any] = {}
    pats = {
        "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "weights_gib": r"Model loading took ([\d.]+) GiB",
        "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
        "cudagraph_gib": r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB",
        "vllm_version": r"V1 LLM engine \(v([\w.+]+)\)",
        "attention_backend": r"Using (\w+) backend",
    }
    for key, pat in pats.items():
        m = re.findall(pat, text)
        if not m:
            continue
        v = m[-1]
        if key == "kv_tokens":
            out[key] = int(v.replace(",", ""))
        elif key.endswith("_gib"):
            out[key] = float(v)
        else:
            out[key] = v
    return out


def stats_row(s: BatchStats) -> tuple[int, int, int, float, int, int, float]:
    return (
        s.num_tokens,
        s.num_seqs,
        s.num_decode,
        s.decode_ctx,
        s.num_prefill,
        s.prefill_tokens,
        s.prefill_attn,
    )


def stats_from_row(t: int, s: int, d: int, dc: float, p: int, pt: int, pa: float) -> BatchStats:
    return BatchStats(t, s, d, p, pt, pa, dc)


@dataclass
class StepTable:
    """Steps of one (model, GPU) pair."""

    model: ModelSpec
    gpu: GPUSpec
    run: NDArray[np.str_]
    stats: list[BatchStats]
    y: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def key(self) -> tuple[str, str]:
        return (self.model.name, self.gpu.name)

    def select(self, mask: NDArray[np.bool_]) -> StepTable:
        idx = np.flatnonzero(mask)
        return StepTable(
            self.model,
            self.gpu,
            self.run[idx],
            [self.stats[i] for i in idx],
            self.y[idx],
            self.meta,
        )

    def runs(self, names: Iterable[str]) -> StepTable:
        names = set(names)
        return self.select(np.array([r in names for r in self.run], dtype=bool))

    def profile(self) -> StepTable:
        """Steps the forecaster may train on: the synthetic profile and smoke replays."""
        return self.runs(("profile", "smoke"))

    def validation(self) -> StepTable:
        """Steps recorded while replaying the Azure traces (never trained on)."""
        return self.select(np.array([r not in ("profile", "smoke") for r in self.run]))

    def subsample(self, n: int, seed: int = 0) -> StepTable:
        if len(self) <= n:
            return self
        rng = np.random.default_rng(seed)
        mask = np.zeros(len(self), dtype=bool)
        mask[rng.choice(len(self), size=n, replace=False)] = True
        return self.select(mask)

    def to_csv_gz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(STEP_COLUMNS)
        for r, s, y in zip(self.run, self.stats, self.y):
            t, n, d, dc, p, pt, pa = stats_row(s)
            w.writerow((r, t, n, d, f"{dc:.0f}", p, pt, f"{pa:.0f}", f"{y:.7f}"))
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(buf.getvalue())

    @classmethod
    def from_csv_gz(cls, path: Path, model: ModelSpec, gpu: GPUSpec) -> StepTable:
        runs, stats, ys = [], [], []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                runs.append(row["run"])
                stats.append(
                    stats_from_row(
                        int(row["T"]),
                        int(row["S"]),
                        int(row["D"]),
                        float(row["Dc"]),
                        int(row["P"]),
                        int(row["PT"]),
                        float(row["PA"]),
                    )
                )
                ys.append(float(row["y"]))
        return cls(model, gpu, np.array(runs), stats, np.array(ys, dtype=np.float64))


@dataclass
class RunInfo:
    """Facts about one (model, GPU) run that are not per-step."""

    model: str
    gpu: str
    meta: dict[str, Any]
    server: dict[str, Any]
    microbench: dict[str, Any]
    plan: list[dict[str, Any]]
    windows: dict[str, tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "gpu": self.gpu,
            "meta": self.meta,
            "server": self.server,
            "microbench": self.microbench,
            "plan": self.plan,
            "windows": {k: list(v) for k, v in self.windows.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunInfo:
        return cls(
            d["model"],
            d["gpu"],
            d["meta"],
            d["server"],
            d.get("microbench", {}),
            d.get("plan", []),
            {k: (float(v[0]), float(v[1])) for k, v in d.get("windows", {}).items()},
        )

    def gpu_spec(self) -> GPUSpec:
        g = get_gpu(self.gpu)
        mb = self.microbench
        if mb.get("gemm_flops") and mb.get("copy_bandwidth"):
            g = g.with_measurements(float(mb["gemm_flops"]), float(mb["copy_bandwidth"]))
        return g

    @property
    def max_num_seqs(self) -> int:
        return int(self.meta["max_num_seqs"])

    @property
    def max_num_batched_tokens(self) -> int:
        return int(self.meta["max_num_batched_tokens"])

    @property
    def max_model_len(self) -> int:
        return int(self.meta["max_model_len"])

    @property
    def kv_tokens(self) -> int:
        return int(self.server["kv_tokens"])


def _windows(run_dir: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    p = run_dir / "runs.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["run"]] = (float(r["start"]), float(r["end"]))
    return out


def load_raw_run(run_dir: Path) -> tuple[RunInfo, StepTable]:
    """Read a raw run directory written by ``bench/run_combo.sh``."""
    meta = json.loads((run_dir / "meta.json").read_text())
    if "sif" in meta:  # keep the image name, not the cluster path (it names the user)
        meta["sif"] = Path(meta["sif"]).name
    with open_text(run_dir / "server.log") as f:
        server = parse_server_log(f.read())
    mb_path = run_dir / "gpu.json"
    microbench = json.loads(mb_path.read_text()) if mb_path.exists() else {}
    plan_path = run_dir / "validation_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else []
    windows = _windows(run_dir)
    model = get_model(meta["model"])
    gpu = get_gpu(meta["gpu_name"])
    info = RunInfo(model.name, gpu.name, meta, server, microbench, plan, windows)
    steps = load_probe(run_dir / "steps.jsonl")
    y = periods(steps)
    labels = np.full(len(steps), "", dtype=object)
    g0 = np.array([s.g0 for s in steps])
    for name, (t0, t1) in windows.items():
        labels[(g0 >= t0) & (g0 <= t1)] = name
    keep = labels != ""
    idx = np.flatnonzero(keep)
    table = StepTable(
        model,
        gpu,
        labels[idx].astype(str),
        [steps[i].stats for i in idx],
        y[idx],
        {"run_dir": run_dir.name},
    )
    return info, table


def compact_replay(src: Path, dst: Path) -> None:
    """Per-request compact record: send, lengths, token count, first and last token."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_text(src) as f, gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as out:
        for line in f:
            rec = json.loads(line)
            if "summary" in rec:
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                continue
            toks = rec.get("tok") or []
            c = {
                "i": rec["i"],
                "send": rec["send"],
                "prompt_len": rec["prompt_len"],
                "output_len": rec["output_len"],
                "ntok": len(toks),
                "first": toks[0] if toks else None,
                "last": toks[-1] if toks else None,
            }
            if rec.get("error"):
                c["error"] = rec["error"]
            out.write(json.dumps(c, separators=(",", ":")) + "\n")


def compact_run(run_dir: Path, results: Path) -> tuple[RunInfo, StepTable]:
    """Write the compact tables for one raw run into ``results``."""
    info, table = load_raw_run(run_dir)
    table.to_csv_gz(results / "steps" / info.gpu / f"{info.model}.csv.gz")
    p = results / "runs" / info.gpu / f"{info.model}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info.to_dict(), indent=2, sort_keys=True) + "\n")
    for name in info.windows:  # profile and smoke too: they calibrate frontend latency
        src = run_dir / f"{name}.jsonl"
        if exists(src):
            compact_replay(src, results / "replays" / info.gpu / info.model / f"{name}.jsonl.gz")
    return info, table


@dataclass
class Dataset:
    """Every measured (model, GPU) pair under ``results/``."""

    tables: dict[tuple[str, str], StepTable]
    infos: dict[tuple[str, str], RunInfo]
    root: Path

    @classmethod
    def load(cls, results: str | Path, measured_peaks: bool = False) -> Dataset:
        root = Path(results)
        tables: dict[tuple[str, str], StepTable] = {}
        infos: dict[tuple[str, str], RunInfo] = {}
        for p in sorted((root / "runs").glob("*/*.json")):
            info = RunInfo.from_dict(json.loads(p.read_text()))
            steps = root / "steps" / info.gpu / f"{info.model}.csv.gz"
            if not steps.exists():
                continue
            gpu = info.gpu_spec()
            tables[(info.model, info.gpu)] = StepTable.from_csv_gz(
                steps, get_model(info.model), gpu
            )
            infos[(info.model, info.gpu)] = info
        return cls(tables, infos, root)

    @property
    def models(self) -> list[str]:
        return sorted({m for m, _ in self.tables}, key=lambda m: get_model(m).num_params)

    @property
    def gpus(self) -> list[str]:
        return sorted({g for _, g in self.tables})

    def pairs(self, model: str | None = None, gpu: str | None = None) -> list[tuple[str, str]]:
        return [
            k
            for k in self.tables
            if (model is None or k[0] == model) and (gpu is None or k[1] == gpu)
        ]

    def replay_paths(self, model: str, gpu: str) -> dict[str, Path]:
        d = self.root / "replays" / gpu / model
        return {p.name.removesuffix(".jsonl.gz"): p for p in sorted(d.glob("*.jsonl.gz"))}

    def __iter__(self) -> Iterator[StepTable]:
        return iter(self.tables.values())


def concat(tables: Sequence[StepTable]) -> tuple[list[BatchStats], FloatArray]:
    stats: list[BatchStats] = []
    ys = []
    for t in tables:
        stats.extend(t.stats)
        ys.append(t.y)
    return stats, (np.concatenate(ys) if ys else np.zeros(0))
