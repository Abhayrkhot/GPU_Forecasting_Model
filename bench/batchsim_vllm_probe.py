"""vLLM plugin that records every engine iteration's batch shape and GPU time.

Loaded through vLLM's ``vllm.general_plugins`` entry point (``bench/install_probe.sh``
writes the metadata), it is inert unless ``BATCHSIM_STEP_LOG`` names an output file. It
wraps ``Worker.execute_model`` and ``Worker.sample_tokens`` (vLLM 0.34) and records CUDA
events on the current stream before the forward pass and after sampling. Events are read
lazily, once they have completed, so no step ever waits on the probe and asynchronous
scheduling is undisturbed. A failure inside the probe disables it; it never fails a step.

Each JSON line describes one iteration:

``i``      iteration index
``cpu``    wall-clock time (``time.time()``) when the worker received the step
``g0/g1``  wall-clock times of the GPU start/end events
``T,S``    scheduled tokens and sequences
``D,Dc``   decode sequences (one new token) and the sum of their context lengths
``p``      prefill chunks as ``[new_tokens, context]`` pairs

GPU times are chained: each step's start is measured from the previous step's start, and
its end from its own start. ``cudaEventElapsedTime`` returns float32 milliseconds, so
measuring everything from one reference event would lose resolution as the server runs
(0.25 ms after 35 minutes); short intervals keep microsecond resolution.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any

_STATE: dict[str, Any] = {}
_LOG = logging.getLogger("batchsim_probe")


def _composition(so: Any) -> dict[str, Any]:
    sched = so.num_scheduled_tokens
    ctx: dict[str, int] = {}
    for nr in so.scheduled_new_reqs:
        ctx[nr.req_id] = int(nr.num_computed_tokens)
    cached = so.scheduled_cached_reqs
    for rid, c in zip(cached.req_ids, cached.num_computed_tokens):
        ctx[rid] = int(c)
    dec = 0
    dec_ctx = 0
    prefill = []
    for rid, n in sched.items():
        c = ctx.get(rid, 0)
        if n == 1:
            dec += 1
            dec_ctx += c
        else:
            prefill.append([int(n), c])
    return {
        "T": int(so.total_num_scheduled_tokens),
        "S": len(sched),
        "D": dec,
        "Dc": dec_ctx,
        "p": prefill,
    }


def _disable(err: BaseException) -> None:
    _STATE["disabled"] = True
    _LOG.warning("batchsim probe disabled after error: %r", err)


def _drain(block: bool = False) -> None:
    st = _STATE
    pending: deque[tuple[dict[str, Any], Any, Any]] = st["pending"]
    out = st["file"]
    while pending:
        rec, ev0, ev1 = pending[0]
        if block:
            ev1.synchronize()
        elif not ev1.query():  # ev0 precedes ev1 on the same stream, so it is done too
            break
        pending.popleft()
        anchor_ev, anchor_wall = st["anchor"]
        g0 = anchor_wall + anchor_ev.elapsed_time(ev0) / 1e3
        g1 = g0 + ev0.elapsed_time(ev1) / 1e3
        st["anchor"] = (ev0, g0)
        rec["g0"] = round(g0, 6)
        rec["g1"] = round(g1, 6)
        out.write(json.dumps(rec, separators=(",", ":")) + "\n")
    now = time.monotonic()
    if block or now - st["flushed"] > 2.0:  # bound what a hard kill can lose
        out.flush()
        st["flushed"] = now


def _flush_at_exit() -> None:
    if "file" not in _STATE or _STATE.get("closed"):
        return
    try:
        with _STATE["lock"]:
            if not _STATE.get("disabled"):
                _drain(block=True)
    except Exception as e:  # never raise during interpreter shutdown
        _LOG.warning("batchsim probe final flush failed: %r", e)
    finally:
        _STATE["file"].close()
        _STATE["closed"] = True


def _ensure_started() -> None:
    if "file" in _STATE:
        return
    import torch

    path = os.environ["BATCHSIM_STEP_LOG"]
    torch.cuda.synchronize()
    ref = torch.cuda.Event(enable_timing=True)
    ref.record()
    ref.synchronize()
    _STATE.update(
        file=open(path, "a", buffering=1 << 16),  # noqa: SIM115 (closed at exit)
        anchor=(ref, time.time()),
        pending=deque(),
        flushed=time.monotonic(),
        index=0,
        lock=threading.Lock(),
        open_rec=None,
    )
    atexit.register(_flush_at_exit)
    # atexit does not run in a fork-started multiprocessing child; Finalize does.
    from multiprocessing import util

    util.Finalize(None, _flush_at_exit, exitpriority=100)


def register() -> None:
    if not os.environ.get("BATCHSIM_STEP_LOG"):
        return
    try:
        from vllm.v1.worker import gpu_worker
    except Exception:  # API-server processes may not have CUDA set up; nothing to patch
        return
    worker_cls = gpu_worker.Worker
    if getattr(worker_cls, "_batchsim_probe", False):
        return
    orig_execute = worker_cls.execute_model
    orig_sample = worker_cls.sample_tokens

    def _end_step() -> None:
        import torch

        rec_ev = _STATE["open_rec"]
        if rec_ev is None:
            return
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        rec, ev0 = rec_ev
        _STATE["pending"].append((rec, ev0, ev1))
        _STATE["open_rec"] = None
        _drain()

    def _begin(scheduler_output: Any) -> None:
        import torch

        _ensure_started()
        with _STATE["lock"]:
            if _STATE["open_rec"] is not None:  # previous step returned without sampling
                _end_step()
            rec = {"i": _STATE["index"], "cpu": round(time.time(), 6)}
            rec.update(_composition(scheduler_output))
            _STATE["index"] += 1
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            _STATE["open_rec"] = (rec, ev0)

    def _finish() -> None:
        if "file" in _STATE and not _STATE.get("disabled"):
            with _STATE["lock"]:
                _end_step()

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        active = not _STATE.get("disabled") and scheduler_output.total_num_scheduled_tokens > 0
        if active:
            try:
                _begin(scheduler_output)
            except Exception as e:
                _disable(e)
                active = False
        out = orig_execute(self, scheduler_output, *args, **kwargs)
        if active and out is not None:  # this step needs no separate sample_tokens call
            try:
                _finish()
            except Exception as e:
                _disable(e)
        return out

    def sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = orig_sample(self, *args, **kwargs)
        try:
            _finish()
        except Exception as e:
            _disable(e)
        return out

    worker_cls.execute_model = execute_model
    worker_cls.sample_tokens = sample_tokens
    worker_cls._batchsim_probe = True
