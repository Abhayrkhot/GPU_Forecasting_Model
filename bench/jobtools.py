"""Helpers the GPU job runs inside the vLLM image (``bench/run_combo.sh``).

    python3 bench/jobtools.py microbench --out gpu.json
    python3 bench/jobtools.py profile --run-dir DIR --model qwen3-8b
    python3 bench/jobtools.py plan --run-dir DIR --model qwen3-8b --traces DATA_DIR

``profile`` writes the smoke and profiling workloads, sized from the model, the GPU and
the KV-cache size the server logged. ``plan`` fits a per-run step model (batchsim's
calibrated linear model, the "oracle") on the profiling steps only, finds the request rate
at which each Azure trace saturates the server, and writes validation workloads at fixed
fractions of it. The oracle only places the load levels; forecasts never see it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HF_IDS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}
CONV_FRACS = (0.4, 0.7, 0.9)
CODE_FRACS = (0.5, 0.85)
MAX_RATE = 80.0  # one replay client process stays accurate up to about this


def microbench(out: Path) -> None:
    """Datasheet-free peaks: a large bf16/fp16 GEMM and a 1 GiB device copy."""
    import torch

    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    dt = torch.bfloat16 if cc[0] >= 8 else torch.float16

    def timeit(fn: object, reps: int) -> float:
        for _ in range(3):
            fn()  # type: ignore[operator]
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps):
                fn()  # type: ignore[operator]
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t) / reps)
        return best

    res: dict[str, object] = {"name": dev, "capability": list(cc), "dtype": str(dt)}
    flops = {}
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        flops[n] = 2 * n**3 / timeit(lambda a=a, b=b: a @ b, 20)
        del a, b
    x = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    bw = 2 * (1 << 30) / timeit(lambda x=x, y=y: y.copy_(x), 20)
    del x, y
    torch.cuda.empty_cache()
    res["gemm_flops"] = max(flops.values())
    res["gemm_flops_by_n"] = {str(k): v for k, v in flops.items()}
    res["copy_bandwidth"] = bw
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))


def _limits(run_dir: Path):  # type: ignore[no-untyped-def]
    from batchsim.validation import parse_server_log

    from gpufc.workloads import ServeLimits

    meta = json.loads((run_dir / "meta.json").read_text())
    server = parse_server_log(run_dir / "server.log")
    if "kv_tokens" not in server:
        raise SystemExit("KV cache size not found in server.log")
    return meta, ServeLimits(
        kv_tokens=int(server["kv_tokens"]),
        max_num_seqs=int(meta["max_num_seqs"]),
        max_num_batched_tokens=int(meta["max_num_batched_tokens"]),
        max_model_len=int(meta["max_model_len"]),
    )


def profile(run_dir: Path, model_key: str) -> None:
    from gpufc.specs import get_gpu, get_model
    from gpufc.workloads import profile_workload, smoke_workload

    meta, limits = _limits(run_dir)
    model, gpu = get_model(model_key), get_gpu(meta["gpu_name"])
    smoke_workload().to_csv(run_dir / "smoke.csv")
    wl = profile_workload(model, gpu, limits)
    wl.to_csv(run_dir / "profile.csv")
    print(f"profile: {len(wl)} requests over {wl.duration:.0f}s, kv_tokens={limits.kv_tokens}")


def plan(run_dir: Path, model_key: str, traces: Path) -> None:
    from batchsim.calibrate import fit_linear, load_probe, periods
    from batchsim.workload import load_azure_trace

    from gpufc.workloads import saturation_rate, validation_workload

    _, limits = _limits(run_dir)
    windows = [json.loads(x) for x in (run_dir / "runs.jsonl").read_text().splitlines() if x]
    prof = next(w for w in windows if w["run"] == "profile")
    steps = load_probe(run_dir / "steps.jsonl")
    y = periods(steps)
    keep = [i for i, s in enumerate(steps) if prof["start"] <= s.g0 <= prof["end"]]
    oracle, rep = fit_linear([steps[i].stats for i in keep], y[keep])
    print(f"oracle fit on {len(keep)} profile steps: MAPE {rep.mape:.3f}")
    manifest = []
    for kind, fracs in (("conv", CONV_FRACS), ("code", CODE_FRACS)):
        base = load_azure_trace(traces / f"AzureLLMInferenceTrace_{kind}.csv", f"azure-{kind}")
        base = base.clip(limits.max_model_len)
        # The code trace arrives in large bursts of long prompts: judge its knee by a
        # looser TTFT bound, or slow GPUs would get no code replay at all.
        slo = 2.0 if kind == "conv" else 10.0
        sat = saturation_rate(base.head(4000), oracle, limits, ttft_slo=slo)
        # Above what one client process can drive, scale the ladder down as a whole so the
        # load levels stay distinct.
        ref = min(sat, MAX_RATE / max(fracs))
        print(f"{kind}: saturation ~{sat:.2f} req/s (ladder reference {ref:.2f})")
        for f in fracs:
            rate = round(f * ref, 3)
            if rate <= 0:
                continue
            wl = validation_workload(base, kind, rate)
            wl.to_csv(run_dir / f"{wl.name}.csv")
            manifest.append(
                {
                    "name": wl.name,
                    "kind": kind,
                    "frac": f,
                    "sat": sat,
                    "rate": rate,
                    "requests": len(wl),
                    "seconds": wl.duration,
                }
            )
    (run_dir / "validation_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(" ".join(m["name"] for m in manifest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("microbench")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("profile")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--traces", type=Path, required=True)
    p = sub.add_parser("hf-id")
    p.add_argument("model")
    args = ap.parse_args()
    if args.cmd == "microbench":
        microbench(args.out)
    elif args.cmd == "profile":
        profile(args.run_dir, args.model)
    elif args.cmd == "plan":
        plan(args.run_dir, args.model, args.traces)
    elif args.cmd == "hf-id":
        if args.model not in HF_IDS:
            sys.exit(f"unknown model {args.model}")
        print(HF_IDS[args.model])


if __name__ == "__main__":
    main()
