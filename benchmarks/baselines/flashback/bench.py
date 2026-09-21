import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def bootstrap_runtime(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--xla-preallocate", choices=("true", "false"), default=None)
    parser.add_argument("--xla-mem-fraction", type=str, default=None)
    parser.add_argument("--xla-allocator", choices=("default", "platform"), default=None)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Run FlashBack autotuning instead of the default first valid kernel config.",
    )
    args, _ = parser.parse_known_args(argv)

    if args.cuda_visible_devices is not None:
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif "JAX_CUDA_VISIBLE_DEVICES" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.xla_preallocate is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = args.xla_preallocate
    if args.xla_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.xla_mem_fraction
    if args.xla_allocator is not None and args.xla_allocator != "default":
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = args.xla_allocator
    if args.autotune:
        os.environ["SKIP_AUTOTUNER"] = "false"
    elif "SKIP_AUTOTUNER" not in os.environ:
        os.environ["SKIP_AUTOTUNER"] = "true"


bootstrap_runtime(sys.argv[1:])

import jax
import jax.numpy as jnp
import numpy as np

from flashback.ops import softmax_mha
from flashback.pallas_utils import Precision


@dataclass(frozen=True)
class AttentionShape:
    batch_size: int
    num_heads: int
    head_dim: int
    causal: bool
    label: str


PRESET_SHAPES = {
    "gpt2": AttentionShape(batch_size=1, num_heads=12, head_dim=64, causal=True, label="GPT-2 Small"),
    "gpt2-small": AttentionShape(batch_size=1, num_heads=12, head_dim=64, causal=True, label="GPT-2 Small"),
    "single-head": AttentionShape(batch_size=1, num_heads=1, head_dim=64, causal=False, label="Single Head"),
}

PRECISIONS = {
    "bf16": (Precision.BF16, jnp.bfloat16),
    "fp16": (Precision.FP16, jnp.float16),
    "fp32": (Precision.FP32, jnp.float32),
    "tf32": (Precision.TF32_ROUND, jnp.float32),
}

BENCH_FIELDNAMES = [
    "implementation",
    "attention",
    "order",
    "shape_label",
    "batch_size",
    "num_heads",
    "head_dim",
    "causal",
    "n",
    "precision",
    "seed",
    "device",
    "status",
    "compile_autotune_s",
    "ms",
    "warmup",
    "iters",
    "error",
]


def resolve_attention_shape(
    preset: str,
    batch_size: int | None,
    num_heads: int | None,
    head_dim: int | None,
    causal: bool | None,
) -> AttentionShape:
    if preset not in PRESET_SHAPES:
        raise ValueError(f"Unknown preset: {preset}")

    base = PRESET_SHAPES[preset]
    return AttentionShape(
        batch_size=base.batch_size if batch_size is None else batch_size,
        num_heads=base.num_heads if num_heads is None else num_heads,
        head_dim=base.head_dim if head_dim is None else head_dim,
        causal=base.causal if causal is None else causal,
        label=base.label,
    )


def powers_of_two_seq_lengths(max_seq_len: int, min_seq_len: int = 1) -> list[int]:
    if min_seq_len < 1:
        raise ValueError("min_seq_len must be >= 1")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be >= min_seq_len")

    seq_lengths = []
    seq_len = 1
    while seq_len < min_seq_len:
        seq_len <<= 1

    while seq_len <= max_seq_len:
        seq_lengths.append(seq_len)
        seq_len <<= 1

    return seq_lengths


def make_shared_inputs(
    seq_len: int,
    shape: AttentionShape,
    *,
    seed: int,
    dtype,
) -> dict[str, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(seed), 7)
    tensor_shape = (shape.batch_size, seq_len, shape.num_heads, shape.head_dim)
    names = ("Q", "K", "V", "dO", "dQ_seed", "dK_seed", "dV_seed")
    return {
        name: jax.random.normal(key, tensor_shape, dtype=jnp.float32).astype(dtype)
        for name, key in zip(names, keys)
    }


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()
    return tree


def make_flashback_bob_fn(
    dO,
    dQ_seed,
    dK_seed,
    dV_seed,
    *,
    causal: bool,
    precision: Precision,
):
    def fwd(q, k, v):
        sm_scale = float(q.shape[-1]) ** -0.5
        return softmax_mha(q, k, v, sm_scale=sm_scale, causal=causal, precision=precision)

    def first_order(q, k, v, dout):
        return jax.grad(
            lambda qq, kk, vv: jnp.sum(fwd(qq, kk, vv) * dout),
            argnums=(0, 1, 2),
        )(q, k, v)

    def phi(q, k, v, dout):
        gq, gk, gv = first_order(q, k, v, dout)
        return jnp.sum(gq * dQ_seed) + jnp.sum(gk * dK_seed) + jnp.sum(gv * dV_seed)

    return jax.jit(jax.grad(phi, argnums=(0, 1, 2, 3)))


def time_first_call_s(fn) -> float:
    start = time.perf_counter()
    block_until_ready(fn())
    return float(time.perf_counter() - start)


def median_runtime_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        block_until_ready(fn())

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(times))


def run_flashback_single(
    seq_len: int,
    shape: AttentionShape,
    *,
    precision_name: str,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    precision, dtype = PRECISIONS[precision_name]
    inputs = make_shared_inputs(seq_len, shape, seed=seed, dtype=dtype)
    run_bob = make_flashback_bob_fn(
        inputs["dO"],
        inputs["dQ_seed"],
        inputs["dK_seed"],
        inputs["dV_seed"],
        causal=shape.causal,
        precision=precision,
    )

    def benchmark_step():
        return run_bob(inputs["Q"], inputs["K"], inputs["V"], inputs["dO"])

    compile_autotune_s = time_first_call_s(benchmark_step)
    ms = median_runtime_ms(benchmark_step, warmup=warmup, iters=iters)
    return {
        "implementation": "flashback",
        "attention": "softmax",
        "order": "second",
        "shape_label": shape.label,
        "batch_size": shape.batch_size,
        "num_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "causal": shape.causal,
        "n": seq_len,
        "precision": precision_name,
        "seed": seed,
        "device": jax.default_backend(),
        "status": "ok",
        "compile_autotune_s": compile_autotune_s,
        "ms": ms,
        "warmup": warmup,
        "iters": iters,
        "error": None,
    }


def write_rows(rows: list[dict[str, object]], *, path: str, append: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCH_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def child_error_message(proc: subprocess.CompletedProcess[str]) -> str:
    for stream in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"child exited with code {proc.returncode}"


def is_oom_text(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "out of memory",
        "resource_exhausted",
        "resource exhausted",
        "cuda_error_out_of_memory",
        "cuda out of memory",
    )
    return any(marker in lowered for marker in markers)


def crash_row(
    seq_len: int,
    shape: AttentionShape,
    *,
    precision_name: str,
    seed: int,
    warmup: int,
    iters: int,
    error: str,
) -> dict[str, object]:
    return {
        "implementation": "flashback",
        "attention": "softmax",
        "order": "second",
        "shape_label": shape.label,
        "batch_size": shape.batch_size,
        "num_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "causal": shape.causal,
        "n": seq_len,
        "precision": precision_name,
        "seed": seed,
        "device": "unknown",
        "status": "oom" if is_oom_text(error) else "crashed",
        "compile_autotune_s": None,
        "ms": None,
        "warmup": warmup,
        "iters": iters,
        "error": error,
    }


def read_json_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return payload


def child_command(
    args: argparse.Namespace,
    seq_len: int,
    json_path: Path,
    shape: AttentionShape,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--json-out",
        str(json_path),
        "--preset",
        args.preset,
        "--batch-size",
        str(shape.batch_size),
        "--num-heads",
        str(shape.num_heads),
        "--head-dim",
        str(shape.head_dim),
        "--seq-lens",
        str(seq_len),
        "--precision",
        args.precision,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    cmd.append("--causal" if shape.causal else "--non-causal")
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.xla_preallocate is not None:
        cmd.extend(["--xla-preallocate", args.xla_preallocate])
    if args.xla_mem_fraction is not None:
        cmd.extend(["--xla-mem-fraction", args.xla_mem_fraction])
    if args.xla_allocator is not None:
        cmd.extend(["--xla-allocator", args.xla_allocator])
    if args.autotune:
        cmd.append("--autotune")
    return cmd


def run_flashback_benchmark_isolated(
    seq_lengths: list[int],
    shape: AttentionShape,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    if args.csv:
        write_rows([], path=args.csv)

    for seq_len in seq_lengths:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "rows.json"
            proc = subprocess.run(
                child_command(args, seq_len, json_path, shape),
                capture_output=True,
                text=True,
                check=False,
            )
            child_rows = read_json_rows(json_path)

        if proc.returncode == 0 and child_rows:
            rows.extend(child_rows)
            if args.csv:
                write_rows(child_rows, path=args.csv, append=True)
            continue

        row = crash_row(
            seq_len,
            shape,
            precision_name=args.precision,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            error=child_error_message(proc),
        )
        rows.append(row)
        if args.csv:
            write_rows([row], path=args.csv, append=True)

    return rows


def print_flashback_benchmark(rows: list[dict[str, object]], shape: AttentionShape) -> None:
    print(
        "\n"
        f"FlashBack softmax BoB: {shape.label}, "
        f"B={shape.batch_size}, H={shape.num_heads}, D={shape.head_dim}, Causal={shape.causal}"
    )
    print(f"{'N':>7} {'status':>10} {'compile+s':>12} {'ms':>10} {'precision':>10}")
    print("-" * 55)
    for row in rows:
        compile_s = row["compile_autotune_s"]
        ms = row["ms"]
        compile_text = f"{compile_s:12.3f}" if compile_s is not None else f"{row['status']:>12}"
        ms_text = f"{ms:10.3f}" if ms is not None else f"{row['status']:>10}"
        print(
            f"{int(row['n']):7d} {str(row['status']):>10} "
            f"{compile_text} {ms_text} {str(row['precision']):>10}"
        )

    error_rows = [row for row in rows if row["error"]]
    if error_rows:
        print("\nErrors:")
        for row in error_rows:
            print(f"N={row['n']}: {row['status']} - {row['error']}")


def build_parser(description: str = "FlashBack softmax BoB benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--preset", choices=sorted(PRESET_SHAPES), default="gpt2-small")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--head-dim", "--d", dest="head_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--causal",
        dest="causal",
        action="store_const",
        const=True,
        default=None,
        help="Force causal attention",
    )
    parser.add_argument(
        "--non-causal",
        dest="causal",
        action="store_const",
        const=False,
        help="Force non-causal attention",
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--powers-of-two", action="store_true")
    parser.add_argument("--min-seq-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--precision", choices=sorted(PRECISIONS), default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--no-isolated", action="store_true")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--xla-preallocate", choices=("true", "false"), default=None)
    parser.add_argument("--xla-mem-fraction", type=str, default=None)
    parser.add_argument("--xla-allocator", choices=("default", "platform"), default=None)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Run FlashBack autotuning instead of the default first valid kernel config.",
    )
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-out", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    shape = resolve_attention_shape(
        preset=args.preset,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        causal=args.causal,
    )
    seq_lengths = args.seq_lens
    if args.powers_of_two:
        seq_lengths = powers_of_two_seq_lengths(
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
        )

    if args.child_run or args.no_isolated:
        rows = [
            run_flashback_single(
                seq_len,
                shape,
                precision_name=args.precision,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
            )
            for seq_len in seq_lengths
        ]
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(rows), encoding="utf-8")
        if args.csv and not args.child_run:
            write_rows(rows, path=args.csv)
        if not args.child_run:
            print_flashback_benchmark(rows, shape)
        return rows

    rows = run_flashback_benchmark_isolated(seq_lengths, shape, args)
    print_flashback_benchmark(rows, shape)
    return rows


if __name__ == "__main__":
    main()
