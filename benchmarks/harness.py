"""Shared measurement, provenance, and result helpers for benchmark CLIs."""

import csv
import gc
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence, TypeVar

import torch
import triton

ResultT = TypeVar("ResultT")


PROVENANCE_FIELDS = [
    "timestamp_utc",
    "git_sha",
    "git_dirty",
    "python_version",
    "torch_version",
    "triton_version",
    "cuda_runtime_version",
    "driver_version",
    "gpu_name",
    "gpu_cc",
    "gpu_sms",
    "gpu_total_mib",
    "gpu_pstate",
    "gpu_clock_sm_mhz",
    "gpu_clock_mem_mhz",
    "gpu_power_limit_w",
]


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_device(requested: str, *, purpose: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested for {purpose}, but no CUDA device is available")
    return device


def dtype_from_arg(raw: str, *, allowed: Iterable[str]) -> torch.dtype:
    allowed = tuple(allowed)
    if raw not in allowed:
        raise ValueError(f"dtype must be one of {allowed}, got {raw!r}")
    return DTYPES[raw]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _command_output(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip().splitlines()
    return value[0].strip() if value else None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def runtime_provenance(device: torch.device | None = None) -> dict[str, object]:
    git_sha = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--porcelain"])
    result: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": bool(git_status),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "driver_version": None,
        "gpu_name": None,
        "gpu_cc": None,
        "gpu_sms": None,
        "gpu_total_mib": None,
        "gpu_pstate": None,
        "gpu_clock_sm_mhz": None,
        "gpu_clock_mem_mhz": None,
        "gpu_power_limit_w": None,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        state = _command_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=pstate,clocks.sm,clocks.mem,power.limit",
                "--format=csv,noheader,nounits",
            ]
        )
        state_parts = [part.strip() for part in state.split(",")] if state else []
        result.update(
            {
                "gpu_name": props.name,
                "gpu_cc": f"{props.major}.{props.minor}",
                "gpu_sms": props.multi_processor_count,
                "gpu_total_mib": props.total_memory / 1024**2,
                "driver_version": _command_output(
                    [
                        "nvidia-smi",
                        f"--id={index}",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                ),
                "gpu_pstate": state_parts[0] if len(state_parts) > 0 else None,
                "gpu_clock_sm_mhz": _optional_float(state_parts[1] if len(state_parts) > 1 else None),
                "gpu_clock_mem_mhz": _optional_float(state_parts[2] if len(state_parts) > 2 else None),
                "gpu_power_limit_w": _optional_float(state_parts[3] if len(state_parts) > 3 else None),
            }
        )
    return result


def l2_rel(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> float:
    denom = torch.linalg.vector_norm(a.float()).clamp_min(eps)
    return float((torch.linalg.vector_norm((a - b).float()) / denom).item())


def format_float(value, *, width: int, prec: int = 3, sci: bool = False) -> str:
    if value is None:
        return f"{'skipped':>{width}}"
    if sci:
        return f"{value:{width}.{prec}e}"
    return f"{value:{width}.{prec}f}"


def format_ratio(value, *, width: int, prec: int = 2) -> str:
    if value is None:
        return f"{'skipped':>{width}}"
    return f"{value:{width}.{prec}f}x"


def write_rows(
    rows: list[dict[str, object]],
    *,
    path: str,
    fieldnames: list[str],
    append: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = (
        not append
        or not output_path.exists()
        or output_path.stat().st_size == 0
    )
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_error(row: dict[str, object], *, phase: str, error: str | None) -> None:
    if error is None:
        return
    detail = f"{phase}:{error}"
    existing = row.get("error")
    row["error"] = detail if existing in (None, "") else f"{existing}; {detail}"


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round(fraction * (len(ordered) - 1))]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_with_oom_capture(
    fn: Callable[[], ResultT], *, device: torch.device
) -> tuple[ResultT | None, str | None]:
    try:
        return fn(), None
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        cleanup_device(device)
        return None, " ".join(str(exc).strip().splitlines())


def time_once(fn: Callable[[], object], device: torch.device) -> float:
    """Milliseconds for one call, for comparisons that alternate arms.

    `measure_time_and_peak` owns the closed warmup-and-repeat loop. A paired
    comparison has to interleave its arms sample by sample, so it needs the
    single-shot primitive instead.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    sync_device(device)
    return float(start.elapsed_time(end))


def measure_time_and_peak(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            sync_device(device)
            times.append(start.elapsed_time(end))
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        times = []
        for _ in range(iters):
            started = perf_counter()
            fn()
            times.append((perf_counter() - started) * 1000.0)
        peak_mib = 0.0

    times.sort()
    return {
        "ms": times[len(times) // 2],
        "ms_p20": percentile(times, 0.20),
        "ms_p80": percentile(times, 0.80),
        "ms_samples": ";".join(f"{sample:.6f}" for sample in times),
        "peak_mib": peak_mib,
        "warmup_iters": warmup,
        "rep_iters": iters,
    }


def time_and_peak(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    """Compatibility wrapper for callers that need only median and peak memory."""

    measured = measure_time_and_peak(fn, warmup=warmup, iters=iters, device=device)
    return float(measured["ms"]), float(measured["peak_mib"])


def _command_bytes(repo_root: Path, command: list[str]) -> bytes | None:
    try:
        process = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return process.stdout if process.returncode == 0 else None


def source_provenance(repo_root: Path) -> dict[str, str | None]:
    """Hash the library, benchmarks, and validation tests used by a run."""

    paths = [repo_root / "pyproject.toml"]
    for directory in ("src/bob", "benchmarks", "tests"):
        paths.extend(sorted((repo_root / directory).rglob("*.py")))

    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    diff = _command_bytes(
        repo_root,
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
    )
    status = _command_bytes(
        repo_root,
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    return {
        "source_sha256": digest.hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "git_status_sha256": (
            hashlib.sha256(status).hexdigest() if status is not None else None
        ),
    }
