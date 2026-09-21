"""GPT-2 SWA benchmark for the rectangular BoB attention path."""

import argparse
import copy
from pathlib import Path

import torch

from benchmarks.harness import (
    PROVENANCE_FIELDS,
    append_error,
    dtype_from_arg,
    dtype_name,
    format_float,
    format_ratio,
    resolve_device,
    runtime_provenance,
    source_provenance,
    write_rows,
)
from benchmarks.harness import cleanup_device, run_with_oom_capture
from benchmarks.models.language_model_benchmark import (
    make_lm_batch,
    make_model,
    parameter_summary,
    gpt2_config,
    run_one,
)


SWA_SEQ_LENS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SWA_BACKENDS = ("bob", "math")
SWA_DTYPES = ("bf16", "fp16")
SWA_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
}
GPT_PRESETS = ("mini", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
SWA_FIELDNAMES = [
    "mode",
    "preset",
    "order",
    "batch_size",
    "seq_len",
    "window",
    "candidate_backend",
    "ref_backend",
    "device",
    "dtype",
    "candidate_status",
    "ref_status",
    "error",
    "candidate_ms",
    "ref_ms",
    "speedup",
    "candidate_peak_mib",
    "ref_peak_mib",
    "mem_ratio",
    "total_params",
    "attn_params",
    "attn_pct",
    "source_sha256",
    "git_diff_sha256",
    "git_status_sha256",
    *PROVENANCE_FIELDS,
]


def parse_seq_lens(raw: list[str] | None) -> list[int]:
    if raw is None:
        return list(SWA_SEQ_LENS)

    seq_lens = [
        int(piece)
        for chunk in raw
        for piece in chunk.split(",")
        if piece.strip()
    ]
    if not seq_lens:
        raise ValueError("at least one seq_len is required")
    return seq_lens


def parse_refs(raw: str, *, candidate_backend: str) -> list[str]:
    refs = [piece.strip() for piece in raw.split(",") if piece.strip()]
    forbidden = [ref for ref in refs if ref not in SWA_BACKENDS or ref == candidate_backend]
    if forbidden:
        raise ValueError(f"unknown/forbidden refs for rectangular SWA: {forbidden}")
    if not refs:
        raise ValueError("at least one reference backend is required")
    return refs


def resolve_window(seq_len: int, *, window_size: int | None, window_ratio: int) -> int:
    if window_size is not None:
        return int(window_size)
    return max(1, seq_len // int(window_ratio))


def _run_backend(
    *,
    cfg,
    backend: str,
    idx: torch.Tensor,
    targets: torch.Tensor,
    order: str,
    probe_seed: int,
    warmup_ms: int,
    rep_ms: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    window_size: int,
    summarize: bool = False,
) -> tuple[dict[str, float], dict[str, float] | None]:
    model = None
    try:
        model = make_model(
            cfg,
            backend,
            device,
            dtype,
            seed,
            window_size=window_size,
        )
        summary = parameter_summary(model) if summarize else None
        return run_one(model, idx, targets, order, probe_seed, warmup_ms, rep_ms), summary
    finally:
        del model
        cleanup_device(device)


def _empty_row(
    *,
    args: argparse.Namespace,
    seq_len: int,
    window_size: int,
    ref: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    return {
        "mode": "swa",
        "preset": args.preset,
        "order": args.order,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "window": window_size,
        "candidate_backend": args.candidate_backend,
        "ref_backend": ref,
        "device": device.type,
        "dtype": dtype_name(dtype),
        "candidate_status": "pending",
        "ref_status": "pending",
        "candidate_ms": None,
        "ref_ms": None,
        "speedup": None,
        "candidate_peak_mib": None,
        "ref_peak_mib": None,
        "mem_ratio": None,
        "total_params": None,
        "attn_params": None,
        "attn_pct": None,
        "error": None,
    }


def _attach_summary(row: dict[str, object], summary: dict[str, float] | None) -> None:
    if summary is None:
        return
    row["total_params"] = summary["total"]
    row["attn_params"] = summary["attn"]
    row["attn_pct"] = summary["attn_pct"]


def _fill_candidate(row: dict[str, object], stats: dict[str, float]) -> None:
    row["candidate_status"] = "ok"
    row["candidate_ms"] = stats["ms"]
    row["candidate_peak_mib"] = stats["peak_mb"]


def _fill_ref(row: dict[str, object], stats: dict[str, float] | None, status: str) -> None:
    row["ref_status"] = status
    if stats is None:
        return

    row["ref_ms"] = stats["ms"]
    row["ref_peak_mib"] = stats["peak_mb"]
    cand_ms = row["candidate_ms"]
    cand_peak = row["candidate_peak_mib"]
    if cand_ms is not None and float(cand_ms) > 0.0:
        row["speedup"] = float(stats["ms"]) / float(cand_ms)
    if cand_peak is not None and float(cand_peak) > 0.0:
        row["mem_ratio"] = float(stats["peak_mb"]) / float(cand_peak)


def run_swa_bench(
    *,
    args: argparse.Namespace,
    seq_lens: list[int],
    refs: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    provenance = {
        **runtime_provenance(device),
        **source_provenance(Path(__file__).resolve().parents[2]),
    }
    base_cfg = gpt2_config(args.preset)
    base_cfg.vocab_size = args.vocab_size
    first_summary: dict[str, float] | None = None

    for row_idx, seq_len in enumerate(seq_lens):
        cfg = copy.deepcopy(base_cfg)
        cfg.block_size = seq_len
        window_size = resolve_window(
            seq_len,
            window_size=args.window_size,
            window_ratio=args.window_ratio,
        )
        batch_seed = args.seed + row_idx
        probe_seed = args.seed + 10_000 + row_idx

        def run_candidate():
            idx, targets = make_lm_batch(
                args.batch_size,
                seq_len,
                cfg.vocab_size,
                device,
                batch_seed,
            )
            stats, summary = _run_backend(
                cfg=cfg,
                backend=args.candidate_backend,
                idx=idx,
                targets=targets,
                order=args.order,
                probe_seed=probe_seed,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                device=device,
                dtype=dtype,
                seed=args.seed,
                window_size=window_size,
                summarize=True,
            )
            return idx, targets, stats, summary

        candidate_payload, candidate_error = run_with_oom_capture(run_candidate, device=device)
        if candidate_error is not None:
            for ref in refs:
                row = _empty_row(
                    args=args,
                    seq_len=seq_len,
                    window_size=window_size,
                    ref=ref,
                    device=device,
                    dtype=dtype,
                )
                row.update(provenance)
                row["candidate_status"] = "oom"
                row["ref_status"] = "skipped"
                append_error(row, phase="candidate", error=candidate_error)
                rows.append(row)
            continue

        idx, targets, candidate_stats, summary = candidate_payload
        first_summary = first_summary or summary
        try:
            for ref in refs:
                row = _empty_row(
                    args=args,
                    seq_len=seq_len,
                    window_size=window_size,
                    ref=ref,
                    device=device,
                    dtype=dtype,
                )
                row.update(provenance)
                _attach_summary(row, first_summary)
                _fill_candidate(row, candidate_stats)

                if args.reference_max_seq_len is not None and seq_len > args.reference_max_seq_len:
                    _fill_ref(row, None, "capped")
                    rows.append(row)
                    continue

                ref_payload, ref_error = run_with_oom_capture(
                    lambda ref=ref: _run_backend(
                        cfg=cfg,
                        backend=ref,
                        idx=idx,
                        targets=targets,
                        order=args.order,
                        probe_seed=probe_seed,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        device=device,
                        dtype=dtype,
                        seed=args.seed,
                        window_size=window_size,
                    ),
                    device=device,
                )
                if ref_error is not None:
                    _fill_ref(row, None, "oom")
                    append_error(row, phase=f"ref_{ref}", error=ref_error)
                else:
                    ref_stats, _ = ref_payload
                    _fill_ref(row, ref_stats, "ok")
                rows.append(row)
        finally:
            cleanup_device(device)

    return rows


def print_swa_bench(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    first = rows[0]
    summary = next((row for row in rows if row.get("total_params") is not None), None)
    print(
        f"\n=== GPT-2 rectangular SWA sweep: preset={first['preset']}, "
        f"order={first['order']}, candidate={first['candidate_backend']}, "
        f"dtype={first['dtype']} ==="
    )
    if summary is not None:
        print(
            f"total_params={int(summary['total_params']):,}  "
            f"attn_params={int(summary['attn_params']):,} ({float(summary['attn_pct']):.2f}%)"
        )
    print(
        f"{'N':>7} {'W':>7} {'ref':>8} {'cand':>8} {'ref':>8} | "
        f"{'cand ms':>10} {'ref ms':>10} {'speedup':>9} | "
        f"{'cand MiB':>10} {'ref MiB':>10} {'mem x':>8}"
    )
    print("-" * 106)
    for row in rows:
        print(
            f"{int(row['seq_len']):7d} {int(row['window']):7d} "
            f"{str(row['ref_backend']):>8} {str(row['candidate_status']):>8} {str(row['ref_status']):>8} | "
            f"{format_float(row.get('candidate_ms'), width=10, prec=3)} "
            f"{format_float(row.get('ref_ms'), width=10, prec=3)} "
            f"{format_ratio(row.get('speedup'), width=8, prec=2)} | "
            f"{format_float(row.get('candidate_peak_mib'), width=10, prec=1)} "
            f"{format_float(row.get('ref_peak_mib'), width=10, prec=1)} "
            f"{format_ratio(row.get('mem_ratio'), width=7, prec=2)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-2 sliding-window attention benchmark.")
    parser.add_argument("--preset", choices=GPT_PRESETS, default="gpt2")
    parser.add_argument("--seq-lens", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--order", choices=("first", "second"), default="second")
    parser.add_argument("--candidate-backend", choices=SWA_BACKENDS, default="bob")
    parser.add_argument("--refs", default="math")
    parser.add_argument("--window-ratio", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument(
        "--dtype",
        choices=(*SWA_DTYPES, *SWA_DTYPE_ALIASES),
        default="bf16",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--reference-max-seq-len",
        "--ref-max-seq-len",
        dest="reference_max_seq_len",
        type=int,
        default=None,
        help="Skip reference backends above this seq_len and continue the candidate run.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, object]]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.window_ratio < 1:
        parser.error("--window-ratio must be >= 1")
    if args.window_size is not None and args.window_size < 1:
        parser.error("--window-size must be >= 1")

    seq_lens = parse_seq_lens(args.seq_lens)
    refs = parse_refs(args.refs, candidate_backend=args.candidate_backend)
    device = resolve_device(args.device, purpose="swa benchmark")
    args.dtype = SWA_DTYPE_ALIASES.get(args.dtype, args.dtype)
    dtype = dtype_from_arg(args.dtype, allowed=SWA_DTYPES)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = run_swa_bench(
        args=args,
        seq_lens=seq_lens,
        refs=refs,
        device=device,
        dtype=dtype,
    )
    print_swa_bench(rows)
    if args.csv:
        write_rows(rows, path=args.csv, fieldnames=SWA_FIELDNAMES)
    return rows


if __name__ == "__main__":
    main()
