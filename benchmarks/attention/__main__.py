"""Benchmark square, grouped-query, and rectangular second-order attention."""

import argparse
import csv
from pathlib import Path

import torch

from benchmarks.attention import BACKENDS, OUTPUT_NAMES, derivative_outputs, make_case
from benchmarks.harness import (
    cleanup_device,
    is_oom_error,
    l2_rel,
    measure_time_and_peak,
    runtime_provenance,
    source_provenance,
)


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _shape(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.mode == "square":
        return args.h, args.h, args.n, args.n
    if args.mode == "gqa":
        return args.h, args.h_kv, args.n, args.n
    return args.h, args.h, args.m, args.n_kv


def _backend_supported(name: str, args: argparse.Namespace) -> None:
    if name.startswith("hvp-") and (
        args.mode != "square" or args.window or not args.causal
    ):
        raise ValueError(f"{name} supports only dense causal square MHA")


def _errors(
    trial: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in names:
        left = trial[name].float()
        right = reference[name].float()
        metrics[f"{name}_max_abs"] = float((left - right).abs().max())
        metrics[f"{name}_l2_rel"] = l2_rel(left, right)
    return metrics


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    requested = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = DTYPES[args.dtype]
    hq, hkv, query_length, key_length = _shape(args)
    q_offset = args.q_offset
    if q_offset is None and args.mode == "rect":
        q_offset = key_length - query_length
    kwargs = {
        "is_causal": args.causal,
        "scale": args.scale,
        "window_size": args.window,
        "q_offset": q_offset,
        "k_offset": args.k_offset,
    }
    names = OUTPUT_NAMES[: 8 if args.order == "second" else 4]
    provenance = {**runtime_provenance(device), **source_provenance(REPO_ROOT)}

    def new_case():
        return make_case(
            batch=args.b,
            query_heads=hq,
            key_heads=hkv,
            query_length=query_length,
            key_length=key_length,
            head_dim=args.d,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )

    reference = None
    if args.check:
        ref_inputs, ref_bars = new_case()
        reference = derivative_outputs(
            BACKENDS["math"], ref_inputs, ref_bars, order=args.order, **kwargs
        )

    rows: list[dict[str, object]] = []
    for backend_name in args.backends:
        _backend_supported(backend_name, args)
        row: dict[str, object] = {
            "mode": args.mode,
            "backend": backend_name,
            "order": args.order,
            "B": args.b,
            "H_Q": hq,
            "H_KV": hkv,
            "M": query_length,
            "N": key_length,
            "D": args.d,
            "causal": args.causal,
            "window": args.window,
            "dtype": args.dtype,
            "status": "pending",
            "error": None,
            **provenance,
        }
        try:
            if reference is not None:
                inputs, bars = new_case()
                trial = derivative_outputs(
                    BACKENDS[backend_name], inputs, bars, order=args.order, **kwargs
                )
                row.update(_errors(trial, reference, names))
                for name in names:
                    torch.testing.assert_close(
                        trial[name].float(),
                        reference[name].float(),
                        atol=args.atol,
                        rtol=args.rtol,
                        msg=f"{backend_name} {name} failed the correctness check",
                    )

            inputs, bars = new_case()

            def step():
                derivative_outputs(
                    BACKENDS[backend_name], inputs, bars, order=args.order, **kwargs
                )

            row.update(
                measure_time_and_peak(
                    step,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
            )
            row["status"] = "ok"
        except RuntimeError as error:
            if not is_oom_error(error):
                raise
            row["status"] = "oom"
            row["error"] = " ".join(str(error).splitlines())
        finally:
            cleanup_device(device)
        rows.append(row)

    baseline = next(
        (
            float(row["ms"])
            for row in rows
            if row["backend"] == "math" and row["status"] == "ok"
        ),
        None,
    )
    for row in rows:
        row["speedup_vs_math"] = (
            baseline / float(row["ms"])
            if baseline is not None and row["status"] == "ok"
            else None
        )
        latency = (
            f"{float(row['ms']):.3f} ms"
            if row["status"] == "ok"
            else str(row["status"])
        )
        print(
            f"{row['mode']:>6} {row['backend']:>15} "
            f"B={row['B']} H={row['H_Q']}/{row['H_KV']} M={row['M']} N={row['N']} "
            f"D={row['D']} {latency}"
        )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("square", "gqa", "rect"), default="square")
    parser.add_argument(
        "--backends", nargs="+", choices=tuple(BACKENDS), default=("math", "bob")
    )
    parser.add_argument("--order", choices=("first", "second"), default="second")
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--h", type=int, default=8, help="query heads; also key heads outside GQA")
    parser.add_argument("--h-kv", type=int, default=2)
    parser.add_argument("--n", type=int, default=512, help="square sequence length")
    parser.add_argument("--m", type=int, default=128, help="rectangular query length")
    parser.add_argument("--n-kv", type=int, default=512, help="rectangular key/value length")
    parser.add_argument("--d", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--q-offset", type=int)
    parser.add_argument("--k-offset", type=int, default=0)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
