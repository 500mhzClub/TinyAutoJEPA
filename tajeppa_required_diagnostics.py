#!/usr/bin/env python3
"""
tajeppa_required_diagnostics.py

Runs an existing training script (default: train_encoder.py) under torch.profiler and
prints two required diagnostics:

  (1) GPU batch shape (first likely image batch moved to CUDA)
  (2) top profiler rows sorted by self_cuda_time_total

It does NOT require editing your training script.

How it works:
- Monkeypatches torch.Tensor.to / .cuda to detect first CUDA transfer of a 4D "image-like" tensor.
- Monkeypatches torch.optim.Optimizer.step to count optimizer steps and call prof.step().
- Stops after N optimizer steps by raising a controlled exception, then prints/writes profiler table.

Usage:
  python3 tajeppa_required_diagnostics.py --script train_encoder.py --steps 10 --row-limit 30

Output:
  - prints "GPU batch: (...)" line
  - prints profiler table
  - writes profiler table to profile.self_cuda.txt (configurable)
  - optionally writes a full log file via shell redirection
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


class StopAfterSteps(RuntimeError):
    pass


@dataclass
class DiagState:
    printed_gpu_batch: bool = False
    optimizer_steps: int = 0
    started_at: float = 0.0


def _is_cuda_device_spec(x: Any) -> bool:
    try:
        if x is None:
            return False
        if isinstance(x, torch.device):
            return x.type == "cuda"
        if isinstance(x, str):
            # Accept "cuda", "cuda:0", etc.
            return x.strip().lower().startswith("cuda")
        return False
    except Exception:
        return False


def _looks_like_image_batch(t: torch.Tensor) -> bool:
    """
    Heuristic for "GPU batch shape" detection:
    - 4D tensor on CUDA (N, C, H, W) or NHWC-ish still shows as 4D
    - C in {1,3,4} is typical for images
    - H/W reasonably large
    - N reasonable
    """
    try:
        if not isinstance(t, torch.Tensor):
            return False
        if not t.is_cuda:
            return False
        if t.dim() != 4:
            return False

        n, c, h, w = t.shape[0], t.shape[1], t.shape[2], t.shape[3]
        if n < 1 or n > 4096:
            return False
        if c not in (1, 3, 4):
            # some pipelines may use C=8/16 etc; adjust if needed
            return False
        if h < 32 or w < 32:
            return False
        if h > 8192 or w > 8192:
            return False
        return True
    except Exception:
        return False


def _format_tensor_brief(t: torch.Tensor) -> str:
    return (
        f"GPU batch: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"contig={t.is_contiguous()} channels_last={t.is_contiguous(memory_format=torch.channels_last)}"
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
    ap.add_argument("--no-exit", action="store_true", help="Do not stop early; run full script (still prints batch + table at end)")
    ap.add_argument("--env", action="append", default=[], help="Extra env vars KEY=VALUE (can be repeated)")
    args = ap.parse_args()

    # Apply user-provided env vars
    for kv in args.env:
        if "=" not in kv:
            print(f"[DIAG] Ignoring malformed --env (expected KEY=VALUE): {kv}", file=sys.stderr)
            continue
        k, v = kv.split("=", 1)
        os.environ[k] = v

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("[DIAG] ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    print(f"[DIAG] torch={torch.__version__} hip={torch.version.hip} device={device_name}")
    print(f"[DIAG] script={args.script} steps={args.steps} row_limit={args.row_limit} out={args.out}")
    print("[DIAG] NOTE: This will print GPU batch shape once it detects the first image-like CUDA transfer.")

    # Import profiler lazily (some builds may not include it)
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception as e:
        print(f"[DIAG] ERROR: torch.profiler unavailable: {e}", file=sys.stderr)
        return 3

    state = DiagState(printed_gpu_batch=False, optimizer_steps=0, started_at=time.time())

    # Keep originals for unpatching
    orig_tensor_to = torch.Tensor.to
    orig_tensor_cuda = torch.Tensor.cuda
    orig_opt_step = torch.optim.Optimizer.step

    prof_ref: dict[str, Any] = {"prof": None}

    def patched_tensor_to(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        # Call original
        out = orig_tensor_to(self, *a, **kw)

        # Detect CUDA transfer for batch shape
        try:
            if not state.printed_gpu_batch:
                # If the result is CUDA and looks like an image batch, print it.
                if _looks_like_image_batch(out):
                    print(_format_tensor_brief(out))
                    state.printed_gpu_batch = True
        except Exception:
            pass
        return out

    def patched_tensor_cuda(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_tensor_cuda(self, *a, **kw)
        try:
            if not state.printed_gpu_batch and _looks_like_image_batch(out):
                print(_format_tensor_brief(out))
                state.printed_gpu_batch = True
        except Exception:
            pass
        return out

    def patched_opt_step(self: torch.optim.Optimizer, closure: Optional[Callable[[], Any]] = None) -> Any:
        # Call original optimizer step
        out = orig_opt_step(self, closure=closure) if closure is not None else orig_opt_step(self)

        # Count steps + advance profiler step (gives cleaner segmentation)
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
            # Never break training due to diagnostics
            pass

        return out

    # Apply patches
    torch.Tensor.to = patched_tensor_to  # type: ignore[assignment]
    torch.Tensor.cuda = patched_tensor_cuda  # type: ignore[assignment]
    torch.optim.Optimizer.step = patched_opt_step  # type: ignore[assignment]

    # Run the target script under profiler
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
                # Execute the training script as if run directly.
                # This keeps its globals, argparse, etc., intact.
                runpy.run_path(args.script, run_name="__main__")
            except StopAfterSteps as e:
                print(f"[DIAG] {e}")
            except SystemExit as e:
                # If the script exits (argparse, etc.), keep going to print what we captured
                print(f"[DIAG] Target script SystemExit: {e}")
            except BaseException as e:
                exc = e

        # Build profiler table (sorted as requested)
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit)

    finally:
        # Always unpatch
        torch.Tensor.to = orig_tensor_to  # type: ignore[assignment]
        torch.Tensor.cuda = orig_tensor_cuda  # type: ignore[assignment]
        torch.optim.Optimizer.step = orig_opt_step  # type: ignore[assignment]

    # If the script threw, show it (but still print diagnostics we captured)
    if exc is not None:
        print("\n[DIAG] Target script raised an exception (showing traceback):\n", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    # Ensure we emitted the GPU batch line
    if not state.printed_gpu_batch:
        print(
            "[DIAG] WARNING: Did not detect an image-like 4D CUDA transfer. "
            "If your pipeline uses non-(N,C,H,W) inputs or C not in {1,3,4}, "
            "edit _looks_like_image_batch() heuristics in this script.",
            file=sys.stderr,
        )

    # Print the table and write it out
    print("\n[DIAG] Profiler table (sorted by self_cuda_time_total):\n")
    print(table)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n[DIAG] Wrote profiler table to: {args.out}")
    except Exception as e:
        print(f"[DIAG] Failed to write profiler output to {args.out}: {e}", file=sys.stderr)

    # Print "top ~15 lines" convenience (exact earlier request)
    print(f"\n[DIAG] Top {args.print_top} lines (for pasting):\n")
    lines = table.splitlines()
    # keep header + separator + top rows; typical table has ~4 header lines
    # We'll print the first (4 + args.print_top) lines to include headers.
    k = min(len(lines), 4 + args.print_top)
    print("\n".join(lines[:k]))

    elapsed = time.time() - state.started_at
    print(f"\n[DIAG] Done. optimizer_steps_seen={state.optimizer_steps} elapsed_s={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
