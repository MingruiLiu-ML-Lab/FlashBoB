# Experiments

All experiments use the installed `bob.sdpa_bob` interface. The `minimal` and
`full` profiles change run size only.

## Setup

```bash
pip install -e '.[benchmarks,native,test]'
```

Install the additional dependencies before running FlashBack or Modal:

```bash
pip install -e '.[benchmarks,native,test,flashback]'
pip install -e '.[modal]'
```

Run commands from the repository root. Outputs go to `runs/` unless `OUT_DIR`
is set.

## Run the standard suite

The standard suite has a bounded `minimal` profile and a `full` profile with
the configured sweeps:

```bash
./benchmarks/reproduce.sh --profile minimal all
./benchmarks/reproduce.sh --profile full all
```

`all` covers `check`, `attention`, `models`, and `swa`. To run every local
target, including the expensive or optional experiments, use:

```bash
./benchmarks/reproduce.sh --profile minimal all front-end flashback sophia
./benchmarks/reproduce.sh --profile full all front-end flashback sophia
```

The full Sophia-H target launches eight-process training jobs. FlashBack needs
the `flashback` optional dependencies. Modal runs remotely and therefore has a
separate command below.

## Run one experiment group

Each target can be launched independently. These commands use the bounded
profile:

```bash
./benchmarks/reproduce.sh --profile minimal check
./benchmarks/reproduce.sh --profile minimal attention
./benchmarks/reproduce.sh --profile minimal models
./benchmarks/reproduce.sh --profile minimal swa
./benchmarks/reproduce.sh --profile minimal front-end
./benchmarks/reproduce.sh --profile minimal flashback
./benchmarks/reproduce.sh --profile minimal sophia
```

These commands run the complete sweeps or campaigns:

```bash
./benchmarks/reproduce.sh --profile full check
./benchmarks/reproduce.sh --profile full attention
./benchmarks/reproduce.sh --profile full models
./benchmarks/reproduce.sh --profile full swa
./benchmarks/reproduce.sh --profile full front-end
./benchmarks/reproduce.sh --profile full flashback
./benchmarks/reproduce.sh --profile full sophia
```

The maintained targets are:

| Target | Coverage |
|---|---|
| `check` | CPU tests, CUDA tests, and MHA/GQA/MQA/rectangular correctness checks |
| `attention` | D32/D64/D128 MHA, GQA, MQA, rectangular attention, and HVP baselines |
| `flashback` | Preserved FlashBack D64 and D128 runs |
| `models` | GPT-2, Pythia, Llama, SmolLM2, Qwen2.5, and Granite 3.1 |
| `swa` | GPT-2 sliding-window model benchmark |
| `sophia` | Sophia-H routing checks or the full FineWeb training campaign |
| `front-end` | cuDNN against aten flash for the dense forward and first backward |

Each row records runtime and source provenance; correctness failures stop the
run. Keep the raw output with every reported timing.

## Correctness checks

Run the unit, benchmark, CUDA, and bounded operator correctness checks with:

```bash
./benchmarks/reproduce.sh --profile minimal check
```

## Attention

One CLI covers all geometries and baselines:

```bash
python -m benchmarks.attention --mode square --n 1024 --d 64
python -m benchmarks.attention --mode gqa --h 24 --h-kv 8 --n 1024 --d 128
python -m benchmarks.attention --mode gqa --h 8 --h-kv 1 --n 1024 --d 64
python -m benchmarks.attention --mode rect --m 128 --n-kv 512 --window 256
python -m benchmarks.attention --mode square --n 1024 \
  --backends math hvp-manual hvp-semi-manual bob
```

`--check` is enabled by default. Use `--no-check` only when a matching
correctness artifact already exists and the materialized reference would make
a long-context capacity run impossible.

## Models

```bash
python -m benchmarks.models \
  --preset granite-3.1-1b-a400m \
  --seq-lens 512,1024,2048 \
  --candidate-backend bob --refs math \
  --device cuda --csv runs/granite.csv
```

The Llama, SmolLM2, Pythia, Qwen, and Granite presets reproduce architecture
shapes with random initialization. They do not download or evaluate pretrained
weights. Granite 3.1 1B-A400M uses 24 layers, hidden size 1024, 16 query heads,
8 key/value heads, head dimension 64, and 1,334,625,280 parameters.

## Sliding-window attention

Run the maintained GPT-2 sliding-window sweep with:

```bash
./benchmarks/reproduce.sh --profile full swa
```

To run one custom sliding-window case directly:

```bash
python -m benchmarks.models.swa \
  --preset gpt2 --seq-lens 2048 4096 --window-ratio 4 \
  --candidate-backend bob --refs math --device cuda \
  --csv runs/swa.csv
```

## Sophia-H

The full FineWeb campaign is explicit because it is an eight-process training
run:

```bash
./benchmarks/reproduce.sh --profile full sophia
```

The suites are `fineweb`, `pythia`, `pythia-d128`, `llama`, and `all`.
Together they cover the five-point AdamW sweep, matched GPT-2 Sophia-H arms,
Pythia D64/D128 Hessian intervals 10/5/2, and the six-step Llama 3.2 1B
systems check. Pass ordinary trainer arguments after the script name to reduce
the token budget or alter logging. Set `WANDB=1` to enable Weights & Biases.

To launch one Sophia-H suite directly, set its name explicitly:

```bash
SOPHIA_SUITE=pythia SOPHIA_OUT_DIR=runs/sophia-pythia \
  ./benchmarks/models/sophia/run.sh
```

## Dense front end

`bob.attention.CUDNN_MIN_COMPUTE_CAPABILITY` decides whether dense attention
runs on cuDNN or on aten FlashAttention. Re-measure it on any architecture the
threshold has not been checked on:

```bash
python -m benchmarks.front_end --seq-lens 512 2048 8192 --csv runs/front-end.csv
```

Exactly one factor changes between the two measurements. The command also
reports the relative L2 difference between the resulting second-order outputs.
If that difference exceeds the expected bfloat16 numerical variation, the
latency comparison does not measure equivalent computations.
[docs/support.md](../docs/support.md) records the current threshold and its
basis.

## Comparison implementations

The FlashBack and Hessian-vector-product (HVP) comparison sources are preserved
without local modifications. First-party adapters may normalize their call
interfaces, but files under
`benchmarks/baselines/flashback/` and
`benchmarks/baselines/hvp_baselines/` are not edited.

```bash
./benchmarks/reproduce.sh --profile full flashback
```

## Modal

Invoke the Modal file as a module. File-mode invocation would shadow the
installed `modal` package.

```bash
modal run -w runs/modal-validation.json -m benchmarks.modal::validate
```

These are bounded H100 checks. They do not replace full experiment runs.

The generated rows include source and runtime provenance. Modal validation is
limited to the shapes executed by this command.
