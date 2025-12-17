#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


RELEVANT_ENV_KEYS = [
    "ROCM_PATH",
    "HIP_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_ENABLE_SDMA",
    "HSA_FORCE_FINE_GRAIN_PCIE",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_SYSTEM_DB_PATH",
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "MIOPEN_COMPILE_PARALLEL_LEVEL",
    "TORCH_LOGS",
    "TORCHDYNAMO_DISABLE",
    "TORCHINDUCTOR_DISABLE",
    "PYTORCH_HIP_ALLOC_CONF",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
]


def run_cmd(cmd: list[str], timeout_s: int = 10) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 999, f"ERROR running {cmd}: {e}"


def du(path: str) -> str:
    # portable fallback: if "du" exists use it, else compute rough size.
    if shutil.which("du"):
        rc, out = run_cmd(["du", "-sh", path], timeout_s=5)
        if rc == 0 and out:
            return out.split()[0]
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    # human-ish
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024 or unit == "TB":
            return f"{total:.1f}{unit}" if unit != "B" else f"{int(total)}B"
        total /= 1024.0
    return "n/a"


def count_files(path: str) -> int:
    n = 0
    for _, _, files in os.walk(path):
        n += len(files)
    return n


def torch_dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


@dataclass
class BenchResult:
    mode: str
    dtype: str
    channels_last: bool
    compile: bool
    batch: int
    image_hw: int
    iters: int
    warmup: int
    fwd_ms: float
    bwd_opt_ms: float
    step_ms: float
    imgs_per_s: float
    max_mem_mb: float
    notes: str = ""


class TinyConvNet(nn.Module):
    """
    Simple, stable op-mix: conv/bn/relu + a couple blocks + classifier.
    This is intentionally not "fastest possible"; it's used to expose
    kernel/layout/dtype path issues clearly.
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 1024):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


def cuda_sync():
    # ROCm uses torch.cuda namespace; synchronize is valid.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_train_loop(
    model: nn.Module,
    device: torch.device,
    batch: int,
    image_hw: int,
    dtype: torch.dtype,
    use_amp: bool,
    channels_last: bool,
    compile_model: bool,
    iters: int,
    warmup: int,
    lr: float,
) -> BenchResult:
    torch.cuda.reset_peak_memory_stats(device=device)

    # synthetic, fixed-shape batch to remove dataloader/augmentation noise
    x = torch.randn(batch, 3, image_hw, image_hw, device=device, dtype=torch.float32)
    y = torch.randint(0, 1024, (batch,), device=device)

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    model = model.to(device)
    model.train(True)

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Keep weights FP32 for AMP; autocast will cast ops.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Optional compile (will be ignored if not supported / disabled)
    compiled = False
    if compile_model:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            compiled = True
        except Exception as e:
            compiled = False

    # AMP context
    amp_dtype_name = str(dtype).replace("torch.", "")
    mode_name = "amp" if use_amp else "fp32"

    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                out = model(x)
                loss = F.cross_entropy(out, y)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
        cuda_sync()

    # timed
    fwd_total = 0.0
    bwd_total = 0.0
    step_total = 0.0

    # Use events for GPU timing; fall back to perf_counter if events unavailable.
    use_events = True
    try:
        start_ev = torch.cuda.Event(enable_timing=True)
        mid_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
    except Exception:
        use_events = False

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)

        if use_events:
            start_ev.record()
        t0 = time.perf_counter()

        # forward + loss
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                out = model(x)
                loss = F.cross_entropy(out, y)
        else:
            out = model(x)
            loss = F.cross_entropy(out, y)

        if use_events:
            mid_ev.record()
        t1 = time.perf_counter()

        # backward + opt
        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if use_events:
            end_ev.record()
            cuda_sync()
            fwd_ms = start_ev.elapsed_time(mid_ev)
            step_ms = start_ev.elapsed_time(end_ev)
            bwd_ms = step_ms - fwd_ms
        else:
            cuda_sync()
            t2 = time.perf_counter()
            fwd_ms = (t1 - t0) * 1000.0
            step_ms = (t2 - t0) * 1000.0
            bwd_ms = step_ms - fwd_ms

        fwd_total += fwd_ms
        bwd_total += bwd_ms
        step_total += step_ms

    fwd_avg = fwd_total / iters
    bwd_avg = bwd_total / iters
    step_avg = step_total / iters
    imgs_per_s = (batch * 1000.0) / step_avg

    peak_mem = torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0)

    notes = ""
    if compile_model and not compiled:
        notes = "torch.compile requested but not active (unsupported/failed/disabled)."

    return BenchResult(
        mode=mode_name,
        dtype=amp_dtype_name,
        channels_last=channels_last,
        compile=compiled,
        batch=batch,
        image_hw=image_hw,
        iters=iters,
        warmup=warmup,
        fwd_ms=fwd_avg,
        bwd_opt_ms=bwd_avg,
        step_ms=step_avg,
        imgs_per_s=imgs_per_s,
        max_mem_mb=peak_mem,
        notes=notes,
    )


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="ROCm/PyTorch diagnostic + micro-bench")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hw", type=int, default=224, help="Image H=W")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--compile", action="store_true", help="Try torch.compile (if available)")
    ap.add_argument("--profile", action="store_true", help="Run a short torch.profiler table (10 steps)")
    ap.add_argument("--out", type=str, default="rocm_diag.json", help="Write JSON summary")
    args = ap.parse_args()

    summary: Dict[str, Any] = {}

    print_header("System / Environment")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", os.getcwd())
    print("env (selected):")
    env_sel = {k: os.environ.get(k, "") for k in RELEVANT_ENV_KEYS if os.environ.get(k) is not None}
    for k in sorted(env_sel.keys()):
        print(f"  {k}={env_sel[k]}")
    summary["env"] = env_sel

    print_header("ROCm CLI tools (best-effort)")
    if shutil.which("rocminfo"):
        rc, out = run_cmd(["rocminfo"], timeout_s=15)
        print("rocminfo rc:", rc)
        # keep it short
        lines = out.splitlines()
        for ln in lines[:120]:
            print(ln)
        summary["rocminfo_head"] = "\n".join(lines[:200])
    else:
        print("rocminfo: not found in PATH")
        summary["rocminfo_head"] = None

    if shutil.which("rocm-smi"):
        rc, out = run_cmd(["rocm-smi"], timeout_s=10)
        print("\nrocm-smi rc:", rc)
        lines = out.splitlines()
        for ln in lines[:80]:
            print(ln)
        summary["rocm_smi_head"] = "\n".join(lines[:120])
    else:
        print("rocm-smi: not found in PATH")
        summary["rocm_smi_head"] = None

    print_header("PyTorch / Device")
    print("torch:", torch.__version__)
    print("torch.version.hip:", torch.version.hip)
    print("cuda namespace available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if not torch.cuda.is_available():
        print("\nERROR: torch.cuda.is_available() is False. Aborting benchmarks.")
        sys.exit(2)

    device = torch.device("cuda:0")
    print("device[0]:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_mem_gb:", round(props.total_memory / (1024**3), 2))
    summary["torch"] = {
        "version": torch.__version__,
        "hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "total_mem_bytes": props.total_memory,
    }

    print_header("Cache directories (best-effort)")
    miopen_root = os.path.expanduser("~/.cache/miopen")
    miopen_userdb = os.path.expanduser("~/.cache/miopen/userdb")
    inductor_root = os.path.expanduser("~/.cache/torch/inductor")
    tex_root = os.path.expanduser("~/.cache/torch_extensions")

    def cache_stat(path: str) -> Dict[str, Any]:
        return {
            "path": path,
            "exists": os.path.exists(path),
            "size": du(path) if os.path.exists(path) else "0",
            "files": count_files(path) if os.path.exists(path) else 0,
        }

    caches = {
        "miopen": cache_stat(miopen_root),
        "miopen_userdb": cache_stat(miopen_userdb),
        "inductor": cache_stat(inductor_root),
        "torch_extensions": cache_stat(tex_root),
    }
    for k, v in caches.items():
        print(f"{k}: exists={v['exists']} size={v['size']} files={v['files']} ({v['path']})")
    summary["caches"] = caches

    print_header("Smoke test: matmul")
    # Keep modest to avoid long runs; enough to validate compute correctness.
    x = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    cuda_sync()
    t0 = time.perf_counter()
    y = x @ x.t()
    cuda_sync()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"matmul 4096x4096 fp16: {dt:.2f} ms  | y={tuple(y.shape)} {y.dtype}")
    summary["smoke_matmul_ms"] = dt

    print_header("Benchmarks: train-step (synthetic fixed batch)")
    results: list[BenchResult] = []

    model = TinyConvNet(in_ch=3, num_classes=1024)

    # Benchmark matrix: NCHW vs channels_last, FP32 vs AMP FP16, plus AMP BF16 if supported.
    bench_specs = []

    bench_specs.append(("fp32", torch.float32, False))
    bench_specs.append(("fp32", torch.float32, True))

    bench_specs.append(("amp_fp16", torch.float16, False))
    bench_specs.append(("amp_fp16", torch.float16, True))

    # BF16 support varies; test it but handle failures.
    bench_specs.append(("amp_bf16", torch.bfloat16, False))
    bench_specs.append(("amp_bf16", torch.bfloat16, True))

    for tag, amp_dtype, cl in bench_specs:
        use_amp = tag.startswith("amp_")
        dtype_name = str(amp_dtype).replace("torch.", "")

        print("\n---")
        print(f"case: tag={tag} use_amp={use_amp} dtype={dtype_name} channels_last={cl} compile={args.compile}")
        try:
            r = timed_train_loop(
                model=TinyConvNet(3, 1024),  # fresh instance per case
                device=device,
                batch=args.batch,
                image_hw=args.hw,
                dtype=amp_dtype,
                use_amp=use_amp,
                channels_last=cl,
                compile_model=args.compile,
                iters=args.iters,
                warmup=args.warmup,
                lr=args.lr,
            )
            print(f"fwd_ms={r.fwd_ms:.2f}  bwd+opt_ms={r.bwd_opt_ms:.2f}  step_ms={r.step_ms:.2f}  imgs/s={r.imgs_per_s:.2f}  peak_mem={r.max_mem_mb:.1f}MB")
            if r.notes:
                print("notes:", r.notes)
            results.append(r)
        except Exception as e:
            print(f"FAILED case tag={tag} dtype={dtype_name} channels_last={cl}: {e}")
            results.append(
                BenchResult(
                    mode="amp" if use_amp else "fp32",
                    dtype=dtype_name,
                    channels_last=cl,
                    compile=False,
                    batch=args.batch,
                    image_hw=args.hw,
                    iters=args.iters,
                    warmup=args.warmup,
                    fwd_ms=float("nan"),
                    bwd_opt_ms=float("nan"),
                    step_ms=float("nan"),
                    imgs_per_s=0.0,
                    max_mem_mb=float("nan"),
                    notes=f"FAILED: {e}",
                )
            )

    summary["benchmarks"] = [asdict(r) for r in results]

    if args.profile:
        print_header("Profiler (10 steps, best-effort)")
        try:
            from torch.profiler import profile, ProfilerActivity

            prof_model = TinyConvNet(3, 1024).to(device).train(True)
            x = torch.randn(args.batch, 3, args.hw, args.hw, device=device, dtype=torch.float32)
            y = torch.randint(0, 1024, (args.batch,), device=device)
            opt = torch.optim.AdamW(prof_model.parameters(), lr=args.lr)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                for i in range(10):
                    opt.zero_grad(set_to_none=True)
                    out = prof_model(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    opt.step()
                    cuda_sync()

            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
            print(table)
            summary["profiler_table"] = table
        except Exception as e:
            print("Profiler failed:", e)
            summary["profiler_table"] = f"FAILED: {e}"

    print_header("Write summary")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")

    print_header("Done")


if __name__ == "__main__":
    main()
