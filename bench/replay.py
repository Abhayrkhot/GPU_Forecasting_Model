"""Replay a workload CSV against an OpenAI-compatible completions endpoint, open loop.

Each request is sent at its ``arrival`` offset from the start, regardless of how many are
outstanding, with a prompt of exactly ``prompt_len`` random token ids and
``max_tokens = min_tokens = output_len`` and ``ignore_eos``, so the engine does exactly the
work the simulator models. Responses are streamed and every token's arrival time is kept.

Output: one JSON line per request with wall-clock ``send`` time (``time.time()``),
token arrival offsets ``tok`` (seconds after ``send``), the token count the server reported,
and any error. A final ``{"summary": ...}`` line records lateness of the sender itself.

Stdlib plus aiohttp (which vLLM already depends on), so it runs inside the vLLM image.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp


def load(path: Path) -> list[tuple[float, int, int]]:
    with path.open(newline="") as f:
        return [
            (float(r["arrival"]), int(r["prompt_len"]), int(r["output_len"]))
            for r in csv.DictReader(f)
        ]


HEADERS = {"Content-Type": "application/json"}


def request_body(model: str, prompt: list[int], out_len: int) -> bytes:
    return json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": out_len,
            "min_tokens": out_len,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        },
        separators=(",", ":"),
    ).encode()


async def one(
    session: aiohttp.ClientSession,
    url: str,
    idx: int,
    body: bytes,
    prompt_len: int,
    out_len: int,
    t_target: float,
    t0_perf: float,
    t0_wall: float,
) -> dict[str, Any]:
    delay = t0_perf + t_target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    send_perf = time.perf_counter()
    rec: dict[str, Any] = {
        "i": idx,
        "target": t_target,
        "send": t0_wall + (send_perf - t0_perf),
        "prompt_len": prompt_len,
        "output_len": out_len,
    }
    toks: list[float] = []
    usage = None
    try:
        async with session.post(url, data=body, headers=HEADERS) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return rec
            buf = b""
            async for chunk in resp.content.iter_any():
                now = time.perf_counter() - send_perf
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data: "):
                            continue
                        data = line[6:]
                        if data == b"[DONE]":
                            continue
                        msg = json.loads(data)
                        if msg.get("usage"):
                            usage = msg["usage"]
                        for ch in msg.get("choices") or ():
                            ids = ch.get("token_ids")
                            k = len(ids) if ids is not None else (1 if ch.get("text") else 0)
                            toks.extend([round(now, 6)] * k)
    except Exception as e:  # keep going; the analysis drops failed requests
        rec["error"] = repr(e)[:300]
    rec["tok"] = toks
    rec["completion_tokens"] = usage.get("completion_tokens") if usage else None
    return rec


async def main_async(args: argparse.Namespace) -> None:
    rows = load(args.workload)
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    # Bodies are serialized before the clock starts, and records are kept in memory until
    # the end, so the event loop does no bulk JSON work while requests are in flight.
    bodies = [
        request_body(
            args.model, [rng.randrange(args.vocab_low, args.vocab_high) for _ in range(p)], o
        )
        for _, p, o in rows
    ]
    url = args.url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.timeout)
    # uvicorn closes idle keep-alive connections after 5 s; reusing one it is closing
    # fails the request, so idle connections are dropped well before that.
    conn = aiohttp.TCPConnector(limit=0, keepalive_timeout=2.0)
    records: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        t0_perf = time.perf_counter() + 0.5
        t0_wall = time.time() + 0.5
        tasks = [
            asyncio.create_task(one(session, url, i, bodies[i], p, o, a, t0_perf, t0_wall))
            for i, (a, p, o) in enumerate(rows)
        ]
        for done, fut in enumerate(asyncio.as_completed(tasks), 1):
            records.append(await fut)
            if done % 500 == 0:
                print(f"  {done}/{len(rows)} done", file=sys.stderr, flush=True)
    lateness = sorted(r["send"] - (t0_wall + r["target"]) for r in records)
    errors = sum(1 for r in records if r.get("error"))
    summary = {
        "summary": {
            "requests": len(rows),
            "errors": errors,
            "t0_wall": t0_wall,
            "send_lateness_p50": lateness[len(lateness) // 2],
            "send_lateness_p99": lateness[int(len(lateness) * 0.99)],
            "send_lateness_max": lateness[-1],
            "workload": str(args.workload),
        }
    }
    with args.out.open("w") as out:
        for r in sorted(records, key=lambda r: r["i"]):
            out.write(json.dumps(r, separators=(",", ":")) + "\n")
        out.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-low", type=int, default=1000)
    ap.add_argument("--vocab-high", type=int, default=100000)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
