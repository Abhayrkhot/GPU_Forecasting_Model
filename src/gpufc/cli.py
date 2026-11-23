"""Command line: ``gpufc <command>``.

gpufc models | gpus                      what the built-in catalogs contain
gpufc explain  --model M --gpu G ...      one step's time, broken into its terms
gpufc predict  --model M --gpu G ...      TTFT/TPOT/E2E for a workload at a rate
gpufc capacity --model M --gpu G ...      highest rate that meets an SLO
gpufc plan     --model M --rate R ...     GPUs needed per GPU type for a target rate
gpufc compact  RUN_DIR... --results DIR   raw GPU runs -> compact tables (results/)
gpufc fit      --results DIR --out F      fit the forecasting bundle on measured pairs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import resources
from pathlib import Path

from batchsim.metrics import SLO
from batchsim.workload import LogNormal, Workload, azure, synthetic

from gpufc.cost import ForecastStepModel
from gpufc.evaluate import stats_from_shapes
from gpufc.serving import Bundle, forecast, max_rate, plan
from gpufc.specs import GPUS, builtin_models, get_gpu, get_model
from gpufc.workloads import MAX_MODEL_LEN


def default_bundle() -> Path:
    return Path(str(resources.files("gpufc") / "data" / "bundle.json"))


def load_bundle(path: str | None) -> Bundle:
    p = Path(path) if path else default_bundle()
    if not p.exists():
        sys.exit(f"no forecasting bundle at {p}; run `gpufc fit` first")
    return Bundle.load(p)


def make_workload(args: argparse.Namespace) -> Workload:
    if args.workload in ("azure-conv", "azure-code"):
        base = azure(args.workload.split("-")[1], args.data_dir).clip(MAX_MODEL_LEN)
        n = int(min(args.requests, len(base)))
        return base.head(n).scale_to_rate(args.rate)
    return synthetic(
        args.requests,
        args.rate,
        LogNormal(args.prompt_mean, 1.0, 16, MAX_MODEL_LEN - 1024),
        LogNormal(args.output_mean, 1.0, 1, 1024),
        seed=args.seed,
    )


def add_workload_args(p: argparse.ArgumentParser, rate: bool = True) -> None:
    p.add_argument(
        "--workload", default="synthetic", choices=("synthetic", "azure-conv", "azure-code")
    )
    if rate:
        p.add_argument("--rate", type=float, default=2.0, help="requests per second")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--prompt-mean", type=float, default=1000.0)
    p.add_argument("--output-mean", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data", help="where Azure traces are cached")


def add_target_args(p: argparse.ArgumentParser, gpu: bool = True) -> None:
    p.add_argument("--model", required=True, help="built-in name, HF id, or config.json")
    if gpu:
        p.add_argument("--gpu", required=True, help="catalog name or nvidia-smi name")
    p.add_argument("--bundle", help="forecasting bundle (default: the packaged one)")
    p.add_argument("--tier", choices=("seen_gpu", "unseen_gpu", "microbench"))


def fmt_ms(x: float) -> str:
    return f"{x * 1e3:8.1f} ms" if math.isfinite(x) else "     n/a"


def cmd_models(_: argparse.Namespace) -> int:
    for name, m in sorted(builtin_models().items(), key=lambda kv: kv[1].num_params):
        kind = f"MoE {m.num_experts}x{m.experts_per_token}" if m.is_moe else "dense"
        print(
            f"{name:16s} {m.num_params / 1e9:6.2f}B  {kind:12s} "
            f"KV {m.kv_bytes_per_token / 1024:6.1f} KiB/token"
        )
    return 0


def cmd_gpus(_: argparse.Namespace) -> int:
    for g in GPUS.values():
        print(
            f"{g.name:10s} {g.arch:9s} {g.memory_bytes / 2**30:6.1f} GiB "
            f"{g.mem_bandwidth / 1e12:5.2f} TB/s {g.peak_flops() / 1e12:6.0f} TFLOP/s"
        )
    return 0


def _step_model(args: argparse.Namespace) -> tuple[Bundle, ForecastStepModel]:
    b = load_bundle(args.bundle)
    model, gpu = get_model(args.model), b.gpu(get_gpu(args.gpu))
    return b, b.step_model(model, gpu, args.tier)


def cmd_explain(args: argparse.Namespace) -> int:
    _, sm = _step_model(args)
    shapes = [(1, args.context)] * args.decode
    shapes += [(args.prefill, args.prefill_context)] if args.prefill else []
    if not shapes:
        sys.exit("give --decode and/or --prefill")
    s = stats_from_shapes(shapes)
    t = sm.step_time(s)
    print(f"{sm.model.name} on {sm.gpu.name} ({sm.meta.get('tier')}): {t * 1e3:.3f} ms")
    try:
        lo, hi = sm.interval("90")
        print(f"  90% interval: {t * lo * 1e3:.3f} - {t * hi * 1e3:.3f} ms")
    except KeyError:
        pass
    for k, v in sorted(sm.breakdown(s).items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v * 1e3:9.3f} ms  {v / t:6.1%}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    b, sm = _step_model(args)
    lim = b.limits(sm.model, sm.gpu)
    if lim.kv_tokens <= 0:
        sys.exit(f"{sm.model.name} does not fit on {sm.gpu.name}")
    wl = make_workload(args)
    eng = b.engine_config()
    res = forecast(wl, sm, lim, eng)
    out = {
        "model": sm.model.name,
        "gpu": sm.gpu.name,
        "tier": sm.meta.get("tier"),
        "kv_tokens": lim.kv_tokens,
        "rate": args.rate,
        "forecast": res,
    }
    try:
        lo, hi = sm.interval("90")
        out["optimistic"] = forecast(wl, sm.scaled(lo), lim, eng)
        out["pessimistic"] = forecast(wl, sm.scaled(hi), lim, eng)
    except KeyError:
        pass
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(
        f"{sm.model.name} on {sm.gpu.name} [{out['tier']}], {args.workload} at "
        f"{args.rate:g} req/s, KV cache {lim.kv_tokens:,} tokens"
    )
    for k in (
        "ttft_p50",
        "ttft_p90",
        "ttft_p99",
        "tpot_p50",
        "tpot_p90",
        "tpot_p99",
        "e2e_p50",
        "e2e_p99",
    ):
        band = ""
        if "optimistic" in out:
            band = f"   [{fmt_ms(out['optimistic'][k]).strip()} .. "
            band += f"{fmt_ms(out['pessimistic'][k]).strip()}]"
        print(f"  {k:9s} {fmt_ms(res[k])}{band}")
    print(f"  output   {res['output_tok_s']:9.0f} tok/s")
    return 0


def _slo(args: argparse.Namespace) -> SLO:
    return SLO(ttft=args.slo_ttft, tpot=args.slo_tpot / 1e3 if args.slo_tpot else math.inf)


def cmd_capacity(args: argparse.Namespace) -> int:
    b, sm = _step_model(args)
    lim = b.limits(sm.model, sm.gpu)
    args.rate = 1.0
    wl = make_workload(args)
    slo = _slo(args)
    cap = max_rate(wl, sm, lim, b.engine_config(), slo, args.attainment)
    print(
        f"{sm.model.name} on {sm.gpu.name}: {cap:.2f} req/s with {args.attainment:.0%} of "
        f"requests meeting {slo}"
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    b = load_bundle(args.bundle)
    model = get_model(args.model)
    args_rate = args.rate
    args.rate = 1.0
    wl = make_workload(args)
    prices = json.loads(Path(args.prices).read_text()) if args.prices else None
    gpus = args.gpus.split(",") if args.gpus else None
    rows = plan(b, model, wl, args_rate, _slo(args), gpus, prices, args.attainment)
    if args.json:
        print(json.dumps([r.to_dict() for r in rows], indent=2))
        return 0
    print(
        f"{model.name}: {args_rate:g} req/s of {args.workload} traffic, SLO {_slo(args)} "
        f"for {args.attainment:.0%} of requests"
    )
    print(
        f"{'gpu':10s} {'tier':11s} {'req/s/GPU':>10s} {'90% band':>15s} {'GPUs':>5s} "
        f"{'worst':>6s} {'$/h':>8s}"
    )
    for r in sorted(rows, key=lambda r: (not r.fits, r.gpus_needed or 1e9)):
        if not r.fits:
            print(f"{r.gpu:10s} does not fit (KV {r.kv_tokens:,} tokens)")
            continue
        cost = f"{r.cost_per_hour:8.2f}" if r.cost_per_hour is not None else "       -"
        print(
            f"{r.gpu:10s} {r.tier:11s} {r.rate_per_gpu:10.2f} "
            f"{r.rate_low:7.2f}-{r.rate_high:<7.2f} {r.gpus_needed:5d} {r.gpus_needed_worst:6d}"
            f" {cost}"
        )
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    from gpufc.data import compact_run

    for d in args.run_dirs:
        info, table = compact_run(Path(d), Path(args.results))
        print(
            f"{info.model:15s} {info.gpu:9s} {len(table):7d} steps, "
            f"{len(info.windows)} replays, kv {info.server.get('kv_tokens')}"
        )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from gpufc.bundle import fit_bundle

    b = fit_bundle(Path(args.results))
    b.save(args.out)
    print(
        f"bundle: {len(b.per_gpu)} per-GPU fits, pooled on {len(b.pooled.trained_on)} pairs "
        f"-> {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gpufc", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="built-in model shapes").set_defaults(fn=cmd_models)
    sub.add_parser("gpus", help="GPU datasheet catalog").set_defaults(fn=cmd_gpus)

    p = sub.add_parser("explain", help="break one step's forecast into its terms")
    add_target_args(p)
    p.add_argument("--decode", type=int, default=0, help="decode sequences in the batch")
    p.add_argument("--context", type=int, default=1024, help="context of each decode")
    p.add_argument("--prefill", type=int, default=0, help="prefill chunk tokens")
    p.add_argument("--prefill-context", type=int, default=0)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("predict", help="TTFT/TPOT/E2E forecast for a workload")
    add_target_args(p)
    add_workload_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_predict)

    for name, fn, help_ in (
        ("capacity", cmd_capacity, "highest rate meeting an SLO"),
        ("plan", cmd_plan, "GPUs needed per GPU type for a target rate"),
    ):
        p = sub.add_parser(name, help=help_)
        add_target_args(p, gpu=(name == "capacity"))
        add_workload_args(p, rate=False)
        if name == "plan":
            p.add_argument("--rate", type=float, required=True, help="target req/s")
            p.add_argument("--gpus", help="comma-separated catalog names (default: all)")
            p.add_argument("--prices", help="JSON {gpu: $/hour} for a cost column")
            p.add_argument("--json", action="store_true")
        p.add_argument("--slo-ttft", type=float, default=1.0, help="seconds")
        p.add_argument("--slo-tpot", type=float, default=0.0, help="milliseconds (0: none)")
        p.add_argument("--attainment", type=float, default=0.9)
        p.set_defaults(fn=fn)

    p = sub.add_parser("compact", help="raw GPU run directories -> results tables")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--results", default="results")
    p.set_defaults(fn=cmd_compact)

    p = sub.add_parser("fit", help="fit the forecasting bundle on measured pairs")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=str(default_bundle()))
    p.set_defaults(fn=cmd_fit)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.fn(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
