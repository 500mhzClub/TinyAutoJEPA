#!/usr/bin/env python3
"""
tajeppa_required_diagnostics.py

Runs a target training script under torch.profiler and emits:
  (1) GPU batch shape (best-effort: first "batch-like" CUDA tensor; fallback largest)
  (2) profiler table sorted by self_cuda_time_total

Stops after N *optimizer steps* reliably by intercepting:
  - torch.amp.GradScaler.step (AMP path; your script uses this via torch.cuda.amp.GradScaler)
  - torch.cuda.amp.GradScaler.step (deprecated alias used by many scripts)
  - optimizer.step for common subclasses (non-AMP path)

Usage:
  python3 tajeppa_required_diagnostics.py --script train_encoder.py --steps 10
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


class StopAfterSteps(RuntimeError):
    pass


@dataclass
class Candidate:
    shape: tuple[int, ...]
    dtype: str
    device: str
    contig: bool
    channels_last: bool
    numel: int


@dataclass
class DiagState:
    printed_gpu_batch: bool = False
    steps_seen: int = 0
    started_at: float = 0.0
    best_candidate: Optional[Candidate] = None


def _tensor_candidate(t: torch.Tensor) -> Optional[Candidate]:
    try:
        if not isinstance(t, torch.Tensor):
            return None
        if not t.is_cuda:
            return None
        # ignore tiny tensors
        if t.numel() < 4096:
            return None
        return Candidate(
            shape=tuple(int(x) for x in t.shape),
            dtype=str(t.dtype),
            device=str(t.device),
            contig=bool(t.is_contiguous()),
            channels_last=bool(t.is_contiguous(memory_format=torch.channels_last)),
            numel=int(t.numel()),
        )
    except Exception:
        return None


def _looks_batch_like(c: Candidate) -> bool:
    # Broad heuristic: 4D (images) or 3D (tokens) large tensors.
    if len(c.shape) == 4:
        n = c.shape[0]
        h = c.shape[-2]
        w = c.shape[-1]
        return (1 <= n <= 16384) and (h >= 16) and (w >= 16)
    if len(c.shape) == 3:
        return c.numel >= 256 * 256
    return False


def _fmt(prefix: str, c: Candidate) -> str:
    return (
        f"{prefix} shape={c.shape} dtype={c.dtype} device={c.device} "
        f"contig={c.contig} channels_last={c.channels_last} numel={c.numel}"
    )


def _make_profiler(**kwargs):
    # torch.profiler.profile signature varies; handle acc_events gracefully.
    from torch.profiler import profile
    try:
        return profile(acc_events=True, **kwargs)
    except TypeError:
        return profile(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train_encoder.py")
    ap.add_argument("--steps", type=int, default=10, help="Stop after this many optimizer steps")
    ap.add_argument("--row-limit", type=int, default=30)
    ap.add_argument("--print-top", type=int, default=15)
    ap.add_argument("--out", default="profile.self_cuda.txt")
    ap.add_argument("--profile-shapes", action="store_true")
    ap.add_argument("--profile-memory", action="store_true")
    ap.add_argument("--no-exit", action="store_true", help="Do not stop early (debug only)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    args = ap.parse_args()

    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v

    if not torch.cuda.is_available():
        print("[DIAG] ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    print(f"[DIAG] torch={torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}")
    print(f"[DIAG] script={args.script} steps={args.steps} out={args.out}")

    from torch.profiler import ProfilerActivity, schedule

    state = DiagState(printed_gpu_batch=False, steps_seen=0, started_at=time.time(), best_candidate=None)

    # --- Save originals for unpatching ---
    orig_tensor_to = torch.Tensor.to
    orig_tensor_cuda = torch.Tensor.cuda

    orig_opt_steps: Dict[type, Callable[..., Any]] = {}

    # GradScaler may exist in both namespaces
    orig_amp_scaler_step = getattr(getattr(torch, "amp", None), "GradScaler", None)
    orig_cuda_amp_scaler_step = getattr(getattr(torch.cuda, "amp", None), "GradScaler", None)

    saved_amp_step = None
    saved_cuda_amp_step = None

    prof_ref: dict[str, Any] = {"prof": None}

    def observe(out: Any):
        if state.printed_gpu_batch:
            return
        if not isinstance(out, torch.Tensor):
            return
        cand = _tensor_candidate(out)
        if cand is None:
            return
        if state.best_candidate is None or cand.numel > state.best_candidate.numel:
            state.best_candidate = cand
        if _looks_batch_like(cand):
            print(_fmt("GPU batch:", cand))
            state.printed_gpu_batch = True

    def patched_tensor_to(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_tensor_to(self, *a, **kw)
        try:
            observe(out)
        except Exception:
            pass
        return out

    def patched_tensor_cuda(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_tensor_cuda(self, *a, **kw)
        try:
            observe(out)
        except Exception:
            pass
        return out

    def step_tick():
        state.steps_seen += 1
        if prof_ref["prof"] is not None:
            try:
                prof_ref["prof"].step()
            except Exception:
                pass
        if (not args.no_exit) and state.steps_seen >= args.steps:
            raise StopAfterSteps(f"Stopped after {state.steps_seen} optimizer steps (requested {args.steps}).")

    # Patch AMP scaler step (best interception point when AMP is used)
    def patch_gradscaler_step(GradScalerCls):
        nonlocal saved_amp_step, saved_cuda_amp_step
        if GradScalerCls is None:
            return
        if not hasattr(GradScalerCls, "step"):
            return

        orig = GradScalerCls.step

        def wrapped(self, optimizer, *a, **kw):
            out = orig(self, optimizer, *a, **kw)
            step_tick()
            return out

        # Assign and remember where it came from
        GradScalerCls.step = wrapped  # type: ignore[assignment]
        return orig

    # Patch optimizer subclasses step() for non-AMP (and as fallback)
    def patch_all_optimizer_steps():
        import inspect
        import torch.optim as optim

        for name, cls in inspect.getmembers(optim, inspect.isclass):
            try:
                if not issubclass(cls, optim.Optimizer):
                    continue
                if not hasattr(cls, "step"):
                    continue
                if cls in orig_opt_steps:
                    continue
                orig = cls.step

                def make_wrapped(orig_step):
                    def wrapped(self, *a, **kw):
                        out = orig_step(self, *a, **kw)
                        step_tick()
                        return out
                    return wrapped

                cls.step = make_wrapped(orig)  # type: ignore[assignment]
                orig_opt_steps[cls] = orig
            except Exception:
                continue

    # --- Apply patches ---
    torch.Tensor.to = patched_tensor_to  # type: ignore[assignment]
    torch.Tensor.cuda = patched_tensor_cuda  # type: ignore[assignment]

    # Patch scaler steps (both namespaces)
    if orig_amp_scaler_step is not None:
        saved_amp_step = patch_gradscaler_step(orig_amp_scaler_step)
    if orig_cuda_amp_scaler_step is not None:
        saved_cuda_amp_step = patch_gradscaler_step(orig_cuda_amp_scaler_step)

    # Patch optimizers too (covers non-AMP and any custom stepping)
    patch_all_optimizer_steps()

    exc: Optional[BaseException] = None
    table: str = ""

    try:
        sched = schedule(wait=0, warmup=0, active=max(args.steps, 1), repeat=1)
        with _make_profiler(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=bool(args.profile_shapes),
            profile_memory=bool(args.profile_memory),
        ) as prof:
            prof_ref["prof"] = prof
            try:
                runpy.run_path(args.script, run_name="__main__")
            except StopAfterSteps as e:
                print(f"[DIAG] {e}")
            except SystemExit as e:
                print(f"[DIAG] Target script SystemExit: {e}")
            except BaseException as e:
                exc = e

        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit)

    finally:
        # --- Unpatch everything ---
        torch.Tensor.to = orig_tensor_to  # type: ignore[assignment]
        torch.Tensor.cuda = orig_tensor_cuda  # type: ignore[assignment]

        # restore scaler steps
        try:
            if orig_amp_scaler_step is not None and saved_amp_step is not None:
                orig_amp_scaler_step.step = saved_amp_step  # type: ignore[assignment]
        except Exception:
            pass
        try:
            if orig_cuda_amp_scaler_step is not None and saved_cuda_amp_step is not None:
                orig_cuda_amp_scaler_step.step = saved_cuda_amp_step  # type: ignore[assignment]
        except Exception:
            pass

        # restore optimizer steps
        for cls, orig in orig_opt_steps.items():
            try:
                cls.step = orig  # type: ignore[assignment]
            except Exception:
                pass

    if exc is not None:
        print("\n[DIAG] Target script raised an exception (traceback):\n", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    if not state.printed_gpu_batch:
        if state.best_candidate is not None:
            print(_fmt("GPU batch (fallback-largest):", state.best_candidate))
        else:
            print("[DIAG] WARNING: No CUDA tensors observed before stopping.", file=sys.stderr)

    print("\n[DIAG] Profiler table (sorted by self_cuda_time_total):\n")
    print(table)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n[DIAG] Wrote profiler table to: {args.out}")
    except Exception as e:
        print(f"[DIAG] Failed to write profiler output: {e}", file=sys.stderr)

    print(f"\n[DIAG] Top {args.print_top} lines (for pasting):\n")
    lines = table.splitlines()
    k = min(len(lines), 4 + args.print_top)
    print("\n".join(lines[:k]))

    elapsed = time.time() - state.started_at
    print(f"\n[DIAG] Done. steps_seen={state.steps_seen} elapsed_s={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
