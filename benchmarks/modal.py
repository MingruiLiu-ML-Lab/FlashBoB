"""Bounded H100 validation for the maintained Triton implementation.

Invoke this file as a module so its name does not shadow the Modal package:

    modal run -m benchmarks.modal::validate
"""

if not __package__:
    raise RuntimeError("invoke with 'modal run -m benchmarks.modal::<function>'")

import os
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]

app = modal.App("flashbob-triton-validation")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.8.0", "triton==3.4.0")
    .pip_install(
        "packaging==25.0", "ninja==1.13.0", "wheel==0.45.1", "numpy==2.1.3"
    )
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .add_local_dir(ROOT / "src", remote_path="/workspace/repo/src", copy=True)
    .add_local_file(
        ROOT / "benchmarks" / "__init__.py",
        remote_path="/workspace/repo/benchmarks/__init__.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "benchmarks" / "harness.py",
        remote_path="/workspace/repo/benchmarks/harness.py",
        copy=True,
    )
    .add_local_dir(
        ROOT / "benchmarks" / "attention",
        remote_path="/workspace/repo/benchmarks/attention",
        copy=True,
    )
    .add_local_file(
        ROOT / "benchmarks" / "baselines" / "__init__.py",
        remote_path="/workspace/repo/benchmarks/baselines/__init__.py",
        copy=True,
    )
)


def _run(*args: str) -> str:
    process = subprocess.run(
        args,
        cwd="/workspace/repo",
        env={**os.environ, "PYTHONPATH": "/workspace/repo/src:/workspace/repo"},
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    return process.stdout


@app.function(gpu="H100!", image=image, timeout=30 * 60, single_use_containers=True)
def validate() -> str:
    """Run square, GQA, and rectangular correctness checks and timings."""

    outputs = []
    for arguments in (
        ("--mode", "square", "--n", "256"),
        ("--mode", "gqa", "--n", "256", "--h", "8", "--h-kv", "2"),
        ("--mode", "rect", "--m", "128", "--n-kv", "256", "--window", "128"),
    ):
        outputs.append(
            _run(
                "python",
                "-m",
                "benchmarks.attention",
                *arguments,
                "--check",
                "--warmup",
                "2",
                "--iters",
                "5",
            )
        )
    return "\n".join(outputs)
