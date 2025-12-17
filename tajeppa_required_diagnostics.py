#!/usr/bin/env python3
"""
tajeppa_required_diagnostics.py (safe for ROCm + DataLoader)

Produces:
  (1) GPU batch line (first batch-like CUDA tensor; fallback largest)
  (2) profiler table sorted by self_cuda_time_total

Stops after N optimizer steps by intercepting:
  - torch.amp.GradScaler.step
  - torch.cuda.amp.GradScaler.step
  - optimizer.step for common optimizer subclasses

IMPORTANT:
- To avoid fork-after-GPU-init crashes, defaults to forcing DataLoader(num_workers=0).
- When num_workers==0, also forces prefetch_factor=None and persistent_workers=False
  (PyTorch requires that).
Override workers:
  export TAJEPA_FORCE_NUM_WORKERS=4
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
class State:
    printed: bool = False
    steps: int = 0
    best: Optional[Candidate] = None
    t0: float = 0.0


def cand_from(t: torch.Tensor) -> Optional[Candidate]:
    try:
        if not t.is_cuda:
            return None
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


def looks_batch_like(c: Candidate) -> bool:
    if len(c.shape) == 4:
        n = c.shape[0]
        h = c.shape[-2]
        w = c.shape[-1]
        return (1 <= n <= 65536) and (h >= 8) and (w >= 8)
    if len(c.shape) == 3:
        return c.numel >= 256 * 256
    return False


def fmt(prefix: str, c: Candidate) -> str:
    return (
        f"{prefix} shape={c.shape} dtype={c.dtype} device={c.device} "
        f"contig={c.contig} channels_last={c.channels_last} numel={c.numel}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train_encoder.py")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--row-limit", type=int, default=30)
    ap.add_argument("--print-top", type=int, default=15)
    ap.add_argument("--out", default="profile.self_cuda.txt")
    ap.add_argument("--record-shapes", action="store_true")
    ap.add_argument("--profile-memory", action="store_true")
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

    st = State(printed=False, steps=0, best=None, t0=time.time())

    # --- Patch DataLoader to force num_workers=N and fix related args ---
    from torch.utils.data import DataLoader as _DataLoader

    orig_dl_init = _DataLoader.__init__
    forced_workers = int(os.getenv("TAJEPA_FORCE_NUM_WORKERS", "0"))

    def patched_dl_init(self, *a, **kw):
        # Force num_workers (positional index 5 or kw)
        if "num_workers" in kw:
            kw["num_workers"] = forced_workers
        else:
            if len(a) > 5:
                a = list(a)
                a[5] = forced_workers
                a = tuple(a)
            else:
                kw["num_workers"] = forced_workers

        # When single-process loading, PyTorch requires:
        # - prefetch_factor must be None / unspecified
        # - persistent_workers must be False
        if forced_workers == 0:
            if "prefetch_factor" in kw:
                kw["prefetch_factor"] = None
            if "persistent_workers" in kw:
                kw["persistent_workers"] = False

        return orig_dl_init(self, *a, **kw)

    _DataLoader.__init__ = patched_dl_init  # type: ignore[assignment]

    # --- Patch Tensor.to / .cuda to capture batch tensor ---
    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda

    def observe(out: Any):
        if not isinstance(out, torch.Tensor):
            return
        c = cand_from(out)
        if c is None:
            return
        if st.best is None or c.numel > st.best.numel:
            st.best = c
        if (not st.printed) and looks_batch_like(c):
            print(fmt("GPU batch:", c))
            st.printed = True

    def patched_to(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_to(self, *a, **kw)
        try:
            observe(out)
        except Exception:
            pass
        return out

    def patched_cuda(self: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        out = orig_cuda(self, *a, **kw)
        try:
            observe(out)
        except Exception:
            pass
        return out

    torch.Tensor.to = patched_to  # type: ignore[assignment]
    torch.Tensor.cuda = patched_cuda  # type: ignore[assignment]

    # --- Patch optimizer/scaler steps to stop after N ---
    orig_opt_steps: Dict[type, Callable[..., Any]] = {}

    def tick():
        st.steps += 1
        if st.steps >= args.steps:
            raise StopAfterSteps(f"Stopped after {st.steps} optimizer steps.")

    saved_amp_step = None
    saved_cuda_amp_step = None

    def patch_scaler(cls):
        if cls is None or not hasattr(cls, "step"):
            return None
        orig = cls.step

        def wrapped(self, optimizer, *a, **kw):
            out = orig(self, optimizer, *a, **kw)
            tick()
            return out

        cls.step = wrapped  # type: ignore[assignment]
        return orig

    amp_cls = getattr(getattr(torch, "amp", None), "GradScaler", None)
    cuda_amp_cls = getattr(getattr(torch.cuda, "amp", None), "GradScaler", None)
    saved_amp_step = patch_scaler(amp_cls)
    saved_cuda_amp_step = patch_scaler(cuda_amp_cls)

    import inspect
    import torch.optim as optim

    for _, cls in inspect.getmembers(optim, inspect.isclass):
        try:
            if not issubclass(cls, optim.Optimizer):
                continue
            if not hasattr(cls, "step"):
                continue
            orig = cls.step

            def make_wrapped(orig_step):
                def wrapped(self, *a, **kw):
                    out = orig_step(self, *a, **kw)
                    tick()
                    return out

                return wrapped

            cls.step = make_wrapped(orig)  # type: ignore[assignment]
            orig_opt_steps[cls] = orig
        except Exception:
            continue

    # --- Run under autograd profiler (no roctracer) ---
    exc: Optional[BaseException] = None
    prof_table = ""

    try:
        with torch.autograd.profiler.profile(
            use_device="cuda",
            record_shapes=bool(args.record_shapes),
            profile_memory=bool(args.profile_memory),
        ) as prof:
            try:
                runpy.run_path(args.script, run_name="__main__")
            except StopAfterSteps as e:
                print(f"[DIAG] {e}")
            except SystemExit as e:
                print(f"[DIAG] Target script SystemExit: {e}")
            except BaseException as e:
                exc = e

        prof_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit)

    finally:
        # Unpatch tensor
        torch.Tensor.to = orig_to  # type: ignore[assignment]
        torch.Tensor.cuda = orig_cuda  # type: ignore[assignment]

        # Restore DataLoader
        _DataLoader.__init__ = orig_dl_init  # type: ignore[assignment]

        # Restore scaler steps
        try:
            if amp_cls is not None and saved_amp_step is not None:
                amp_cls.step = saved_amp_step  # type: ignore[assignment]
        except Exception:
            pass
        try:
            if cuda_amp_cls is not None and saved_cuda_amp_step is not None:
                cuda_amp_cls.step = saved_cuda_amp_step  # type: ignore[assignment]
        except Exception:
            pass

        # Restore optimizers
        for cls, orig in orig_opt_steps.items():
            try:
                cls.step = orig  # type: ignore[assignment]
            except Exception:
                pass

    if exc is not None:
        print("\n[DIAG] Target script raised an exception:\n", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    if not st.printed and st.best is not None:
        print(fmt("GPU batch (fallback-largest):", st.best))

    print("\n[DIAG] Profiler table (sorted by self_cuda_time_total):\n")
    print(prof_table)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(prof_table)
        print(f"\n[DIAG] Wrote profiler table to: {args.out}")
    except Exception as e:
        print(f"[DIAG] Failed to write profiler output: {e}", file=sys.stderr)

    print(f"\n[DIAG] Top {args.print_top} lines (for pasting):\n")
    lines = prof_table.splitlines()
    k = min(len(lines), 4 + args.print_top)
    print("\n".join(lines[:k]))

    print(f"\n[DIAG] Done. steps_seen={st.steps} elapsed_s={time.time()-st.t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
