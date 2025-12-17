#!/usr/bin/env python3
"""
tajeppa_required_diagnostics.py

Runs an existing training script (default: train_encoder.py) under torch.profiler and
emits the two required diagnostics:

  (1) GPU batch shape (best-effort: first "large" CUDA tensor observed; fallback to largest)
  (2) top profiler rows sorted by self_cuda_time_total

No changes needed to your training script.

Mechanism:
- Monkeypatches torch.Tensor.to / .cuda to observe CUDA transfers and capture candidates.
- Monkeypatches torch.optim.Optimizer.step to count optimizer steps and call prof.step().
- Stops after N optimizer steps via a controlled exception, then prints/writes profiler table.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    optimizer_steps: int = 0
    started_at: float = 0.0
    best_candidate: Optional[Candidate] = None


def _tensor_candidate(t: torch.Tensor) -> Optional[Candidate]:
    try:
        if not isinstance(t, torch.Tensor):
            return None
        if not t.is_cuda:
            return None
        # Ignore tiny tensors (weights, scalars, etc.)
        if t.numel() < 32 * 32:  # ~1k elements
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


def _is_probably_batch_like(c: Candidate) -> bool:
    """
    Broad heuristic:
    - Prefer 4D tensors (N,C,H,W or N,H,W,C)
    - Otherwise accept 3D (tokens) if very large
    """
    try:
        if len(c.shape) == 4:
            n = c.shape[0]
            h = c.shape[2]
            w = c.shape[3]
            # Accept wide range; just avoid pathological tiny images
            return (1 <= n <= 8192) and (h >= 16) and (w >= 16)
        if len(c.shape) == 3:
            # token-ish batch: (B, T, D) etc.
            return c.numel >= 256 * 256
        return False
    except Exception:
        return False


def _format_candidate(prefix: str, c: Candidate) -> str:
    return (
        f"{prefix} shape={c.shape} dtype={c.dtype} device={c.device} "
        f"contig={c.contig} channels_last={c.channels_last} numel={c.numel}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train_encoder.py", help="Path to training script to run")
    ap.add_argument("--steps", type=int, default=10, help="Stop after this many optimizer steps")
    ap.add_argument("--row-limit", type=int, default=30, help="Profiler table row_limit")
    ap.add_argument("--out", default="profile.self_cuda.txt", help="Write profiler table here")
    ap.add_argument("--print-top", type=int, default=15, help="How many top lines to print from the table")
    ap.add_argument("--profile-shapes", action="store_true", help="Record shapes (more overhead, more detail)")
    ap.add_argument("--profile-memory", action="store_true", help="Profile memory (more overhead)")
    ap.add_argument("--no-exit", action="store_true", help="Do not stop early; run full script")
    ap.add_argument("--env", action="append", default=[], help="Extra env vars KEY=VALUE (repeatable)")
    args = ap.parse_args()

    # Apply env vars requested via CLI
    for kv in args.env:
        if "=" not in kv:
            print(f"[DIAG] Ignoring malformed --env (expected KEY=VALUE): {kv}", file=sys.stderr)
            continue
        k, v = kv.split("=", 1)
        os.environ[k] = v

    if not torch.cuda.is_available():
        print("[DIAG] ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    print(f"[DIAG] torch={torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}")
    print(f"[DIAG] script={args.script} steps={args.steps} row_limit={args.row_limit} out={args.out}")

    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception as e:
        print(f"[DIAG] ERROR: torch.profiler unavailable: {e}", file=sys.stderr)
        return 3

    state = DiagState(printed_gpu_batch=False, optimizer_steps=0, started_at=time.time(), best_candidate=None)

    # Keep originals for unpatching
    orig_tensor_to = torch.Tensor.to
    orig_tensor_cuda = torch.Tensor.cuda
    orig_opt_step = torch.optim.Optimizer.step

    prof_ref: dict[str, Any] = {"prof": None}

    def observe_tensor(out: Any):
        if state.printed_gpu_batch:
            return
        if not isinstance(out, torch.Tensor):
            return
        cand = _tensor_candidate(out)
        if cand is None:
            return

        # Track best (largest) candidate as fallback
        if (state.best_candidate is None) or (cand.numel > state.best_candidate.numel):
            state.best_candidate = cand

        # Print first batch-like candidate immediately
        if _is_probably_batch_like(cand):
            print(_format_candidate("GPU batch:", cand))
            state.printed_gpu_batch = True

    def patched_tensor_to(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_tensor_to(self, *a, **kw)
        try:
            observe_tensor(out)
        except Exception:
            pass
        return out

    def patched_tensor_cuda(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_tensor_cuda(self, *a, **kw)
        try:
            observe_tensor(out)
        except Exception:
            pass
        return out

    def patched_opt_step(self: torch.optim.Optimizer, closure: Optional[Callable[[], Any]] = None) -> Any:
        out = orig_opt_step(self, closure=closure) if closure is not None else orig_opt_step(self)
        try:
            state.optimizer_steps += 1
            if prof_ref["prof"] is not None:
                try:
                    prof_ref["prof"].step()
                except Exception:
                    pass

            if (not args.no_exit) and state.optimizer_steps >= args.steps:
                raise StopAfterSteps(f"Stopped after {state.optimizer_steps} optimizer steps (requested {args.steps}).")
        except StopAfterSteps:
            raise
        except Exception:
            pass
        return out

    # Apply patches
    torch.Tensor.to = patched_tensor_to  # type: ignore[assignment]
    torch.Tensor.cuda = patched_tensor_cuda  # type: ignore[assignment]
    torch.optim.Optimizer.step = patched_opt_step  # type: ignore[assignment]

    exc: Optional[BaseException] = None
    table: str = ""

    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
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
        # Unpatch no matter what
        torch.Tensor.to = orig_tensor_to  # type: ignore[assignment]
        torch.Tensor.cuda = orig_tensor_cuda  # type: ignore[assignment]
        torch.optim.Optimizer.step = orig_opt_step  # type: ignore[assignment]

    if exc is not None:
        print("\n[DIAG] Target script raised an exception (traceback):\n", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    # If we never printed a batch line, print best fallback candidate
    if not state.printed_gpu_batch:
        if state.best_candidate is not None:
            print(_format_candidate("GPU batch (fallback-largest):", state.best_candidate))
            state.printed_gpu_batch = True
        else:
            print(
                "[DIAG] WARNING: No CUDA tensor candidates observed. "
                "This usually means your script never moved tensors to GPU before it stopped.",
                file=sys.stderr,
            )

    print("\n[DIAG] Profiler table (sorted by self_cuda_time_total):\n")
    print(table)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n[DIAG] Wrote profiler table to: {args.out}")
    except Exception as e:
        print(f"[DIAG] Failed to write profiler output to {args.out}: {e}", file=sys.stderr)

    # Convenience: print top lines for pasting (include header + separator)
    print(f"\n[DIAG] Top {args.print-top if False else args.print_top} lines (for pasting):\n")
    lines = table.splitlines()
    k = min(len(lines), 4 + args.print_top)
    print("\n".join(lines[:k]))

    elapsed = time.time() - state.started_at
    print(f"\n[DIAG] Done. optimizer_steps_seen={state.optimizer_steps} elapsed_s={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
