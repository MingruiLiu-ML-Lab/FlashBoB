"""Evidence for the dense front-end threshold in `bob.attention`.

`CUDNN_MIN_COMPUTE_CAPABILITY` decides whether dense attention runs on cuDNN or
on aten FlashAttention. This measures both, changing exactly one factor: the
fused operator supplying the forward and the first backward. The Triton second
backward, the tensors, and the timing order are identical across arms, so a
difference here is attributable to the front end alone.

Run it on each new architecture before using the threshold there:

    python -m benchmarks.front_end --csv runs/front-end.csv
"""

import argparse
import statistics
from pathlib import Path

import torch

import bob
from bob import attention
from benchmarks.harness import (
    PROVENANCE_FIELDS,
    l2_rel,
    runtime_provenance,
    source_provenance,
    time_once,
    write_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.device("cuda")
ROW_FIELDS = [
    "seq_len", "heads", "head_dim", "order",
    "cudnn_ms", "flash_ms", "cudnn_gain", "agreement_l2_rel", "selected",
    "source_sha256", "git_diff_sha256", "git_status_sha256",
    *PROVENANCE_FIELDS,
]


def _force(use_cudnn: bool):
    """Override the shipped compute-capability choice for one arm."""
    attention.dense_front_end_is_cudnn = lambda _index: use_cudnn


def _case(seq_len, heads, head_dim, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, heads, seq_len, head_dim)

    def rand():
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    q, k, v = (rand().requires_grad_(True) for _ in range(3))
    return (q, k, v), rand(), (rand(), rand(), rand())


def _first_order(tensors, grad_out):
    q, k, v = tensors
    return torch.autograd.grad(bob.sdpa_bob(q, k, v, is_causal=True), (q, k, v), grad_out)


def _second_order(tensors, grad_out, bars):
    q, k, v = tensors
    out = bob.sdpa_bob(q, k, v, is_causal=True)
    first = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=True)
    return torch.autograd.grad(first, (q, k, v), bars)


def measure(seq_len, heads, head_dim, warmup, iters) -> list[dict]:
    tensors, grad_out, bars = _case(seq_len, heads, head_dim)
    steps = {
        "first": lambda: _first_order(tensors, grad_out),
        "second": lambda: _second_order(tensors, grad_out, bars),
    }

    rows = []
    for order, step in steps.items():
        samples = {"cudnn": [], "flash": []}
        for use_cudnn in (True, False):
            _force(use_cudnn)
            for _ in range(warmup):
                step()
        # alternate arms so clock drift and thermal throttling hit both equally
        for index in range(iters):
            arms = ((True, "cudnn"), (False, "flash"))
            for use_cudnn, name in (arms if index % 2 == 0 else arms[::-1]):
                _force(use_cudnn)
                samples[name].append(time_once(step, CUDA))
        cudnn_ms = statistics.median(samples["cudnn"])
        flash_ms = statistics.median(samples["flash"])
        rows.append({
            "seq_len": seq_len, "heads": heads, "head_dim": head_dim, "order": order,
            "cudnn_ms": cudnn_ms, "flash_ms": flash_ms,
            "cudnn_gain": flash_ms / cudnn_ms,
        })

    # Equivalent outputs are required for a valid latency comparison.
    _force(True)
    reference = _second_order(tensors, grad_out, bars)
    _force(False)
    other = _second_order(tensors, grad_out, bars)
    agreement = max(l2_rel(a, b) for a, b in zip(other, reference))
    for row in rows:
        row["agreement_l2_rel"] = agreement
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--csv", type=str)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the front-end comparison requires CUDA")

    shipped = attention.dense_front_end_is_cudnn(torch.cuda.current_device())
    selected = "cudnn" if shipped else "flash"
    provenance = {**runtime_provenance(torch.device("cuda")), **source_provenance(REPO_ROOT)}
    print(f"{provenance['gpu_name']} cc{provenance['gpu_cc']} | "
          f"torch {provenance['torch_version']} | shipped selection: {selected}")
    print(f"{'N':>7}{'order':>8}{'cudnn ms':>11}{'flash ms':>11}"
          f"{'cudnn gain':>12}{'agreement':>12}")

    rows = []
    for seq_len in args.seq_lens:
        try:
            measured = measure(seq_len, args.heads, args.head_dim, args.warmup, args.iters)
        except torch.OutOfMemoryError:
            print(f"{seq_len:>7}  out of memory")
            torch.cuda.empty_cache()
            continue
        for row in measured:
            row["selected"] = selected
            print(f"{row['seq_len']:>7}{row['order']:>8}{row['cudnn_ms']:>11.3f}"
                  f"{row['flash_ms']:>11.3f}{row['cudnn_gain']:>12.3f}"
                  f"{row['agreement_l2_rel']:>12.2e}")
        rows.extend({**row, **provenance} for row in measured)
        torch.cuda.empty_cache()

    print("cudnn gain above 1 means cuDNN is the faster front end at that length")
    if args.csv:
        write_rows(rows, path=args.csv, fieldnames=ROW_FIELDS)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
