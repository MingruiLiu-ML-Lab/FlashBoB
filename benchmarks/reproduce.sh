#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
PROFILE="minimal"
if [[ "${1:-}" == "--profile" ]]; then
  PROFILE="${2:-}"
  shift 2
fi
if [[ "${PROFILE}" != "minimal" && "${PROFILE}" != "full" ]]; then
  echo "--profile must be minimal or full" >&2
  exit 2
fi
OUT_DIR="${OUT_DIR:-runs/${PROFILE}}"
mkdir -p "${OUT_DIR}"

usage() {
  cat <<'EOF'
Usage: ./benchmarks/reproduce.sh [--profile minimal|full] [TARGET ...]

Targets:
  check       Run tests and bounded operator correctness checks.
  attention   Run MHA, GQA, MQA, head-dimension, and rectangular sweeps.
  flashback   Run the preserved FlashBack comparison.
  front-end   Measure cuDNN against aten flash for the dense forward pair.
  models      Run GPT-2, Pythia, Llama, SmolLM2, Qwen, and Granite.
  swa         Run the GPT-2 sliding-window benchmark.
  sophia      Run Sophia-H (routing tests in minimal; training in full).
  all         Run check, attention, models, and SWA.

Full Sophia-H and FlashBack runs are explicit targets because they need extra
dependencies or data.

Environment: PYTHON, DEVICE, OUT_DIR, WARMUP, ITERS
EOF
}

run_attention_case() {
  local name="$1"
  shift
  "${PYTHON}" -m benchmarks.attention "$@" \
    --device "${DEVICE}" \
    --warmup "${WARMUP:-5}" --iters "${ITERS:-20}" \
    --csv "${OUT_DIR}/${name}.csv"
}

run_check() {
  "${PYTHON}" -m pytest -q tests/unit tests/benchmarks
  if [[ "${DEVICE}" == cuda* ]]; then
    "${PYTHON}" -m pytest -q tests/cuda
  fi
  run_attention_case check-square --mode square --n 256 --d 64 --check
  run_attention_case check-gqa --mode gqa --h 8 --h-kv 2 --n 256 --d 64 --check
  run_attention_case check-mqa --mode gqa --h 8 --h-kv 1 --n 256 --d 64 --check
  run_attention_case check-rect --mode rect --m 128 --n-kv 256 --window 128 --d 64 --check
}

run_attention() {
  local n d
  if [[ "${PROFILE}" == "minimal" ]]; then
    for d in 32 64 128; do
      run_attention_case "mha-d${d}-n512" --mode square --n 512 --d "${d}" \
        --backends math bob --check
    done
    run_attention_case gqa-n512 --mode gqa --h 24 --h-kv 8 --n 512 --d 128 \
      --backends math bob --check
    run_attention_case mqa-n512 --mode gqa --h 8 --h-kv 1 --n 512 --d 64 \
      --backends math bob --check
    run_attention_case rectangular --mode rect --m 256 --n-kv 512 \
      --window 512 --d 64 --backends math bob --check
    return
  fi

  for n in 256 512 1024 2048 4096 8192 16384; do
    run_attention_case "mha-d64-n${n}" --mode square --n "${n}" --d 64 \
      --backends math hvp-manual hvp-semi-manual bob --check
  done
  for d in 32 128; do
    run_attention_case "mha-d${d}-n4096" --mode square --n 4096 --d "${d}" \
      --backends math hvp-manual hvp-semi-manual bob --check
  done
  for n in 256 512 1024 2048; do
    run_attention_case "mha-d128-n${n}" --mode square --n "${n}" --d 128 \
      --backends math bob --check
  done
  for n in 32768 65536 131072 262144; do
    run_attention_case "mha-d64-n${n}" --mode square --n "${n}" --d 64 \
      --backends bob --no-check
  done
  for n in 8192 16384 32768 65536; do
    run_attention_case "mha-d128-n${n}" --mode square --n "${n}" --d 128 \
      --backends bob --no-check
  done
  for n in 256 512 1024 2048 4096; do
    run_attention_case "gqa-n${n}" --mode gqa --h 24 --h-kv 8 --n "${n}" --d 128 \
      --backends math bob --check
    run_attention_case "mqa-n${n}" --mode gqa --h 8 --h-kv 1 --n "${n}" --d 64 \
      --backends math bob --check
  done
  run_attention_case rectangular --mode rect --m 256 --n-kv 4096 \
    --window 512 --d 64 --backends math bob --check
}

run_front_end() {
  local lengths=(512 2048)
  [[ "${PROFILE}" == "full" ]] && lengths=(512 1024 2048 4096 8192 16384)
  "${PYTHON}" -m benchmarks.front_end \
    --seq-lens "${lengths[@]}" \
    --warmup "${WARMUP:-10}" --iters "${ITERS:-30}" \
    --csv "${OUT_DIR}/front-end.csv"
}

run_model() {
  local preset="$1"
  local lengths="$2"
  local batch_size="$3"
  local refs="$4"
  shift 4
  "${PYTHON}" -m benchmarks.models \
    --preset "${preset}" --seq-lens "${lengths}" --batch-size "${batch_size}" \
    --candidate-backend bob --refs "${refs}" --order second \
    --dtype bfloat16 --device "${DEVICE}" \
    --warmup-ms 25 --rep-ms 100 --csv "${OUT_DIR}/${preset}.csv" "$@"
}

run_models() {
  if [[ "${PROFILE}" == "minimal" ]]; then
    run_model mini 128 1 math
    run_model pythia-160m-d128 128 1 math
    run_model granite-3.1-1b-a400m 128 1 math
    return
  fi
  run_model gpt2 512,1024,2048,4096,8192,16384,32768 4 math,hvp_manual,hvp_semi_manual \
    --ref-max-seq-len math=4096 --ref-max-seq-len hvp_manual=8192 \
    --ref-max-seq-len hvp_semi_manual=8192
  for model in gpt2-medium gpt2-large; do
    run_model "${model}" 512,1024,2048,4096,8192,16384 4 math,hvp_manual,hvp_semi_manual \
      --ref-max-seq-len math=2048 --ref-max-seq-len hvp_manual=4096 \
      --ref-max-seq-len hvp_semi_manual=4096
  done
  for model in pythia-160m pythia-1.4b pythia-160m-d128 pythia-410m-d128; do
    run_model "${model}" 512,1024,2048,4096,8192,16384,32768 1 math
  done
  for model in llama-3.2-1b llama-3.2-3b; do
    run_model "${model}" 256,512,1024,2048 1 math
  done
  for model in smollm2-135m smollm2-360m; do
    run_model "${model}" 128,256,512,1024,2048,4096,8192,16384 1 math
  done
  for model in qwen2.5-1.5b granite-3.1-1b-a400m; do
    run_model "${model}" 256,512,1024,2048 1 math
  done
}

run_swa() {
  local preset=mini
  local lengths=(128 256 512)
  local batch_size=1
  if [[ "${PROFILE}" == "full" ]]; then
    preset=gpt2
    lengths=(2048 4096 8192 16384)
    batch_size=4
  fi
  "${PYTHON}" -m benchmarks.models.swa \
    --preset "${preset}" --seq-lens "${lengths[@]}" --batch-size "${batch_size}" \
    --candidate-backend bob --refs math --device "${DEVICE}" \
    --csv "${OUT_DIR}/swa.csv"
}

run_sophia() {
  if [[ "${PROFILE}" == "minimal" ]]; then
    "${PYTHON}" -m pytest -q tests/benchmarks/test_sophia_routing.py tests/cuda/test_sophia_cuda_routing.py
  else
    SOPHIA_SUITE="${SOPHIA_SUITE:-all}" SOPHIA_OUT_DIR="${OUT_DIR}/sophia" \
      benchmarks/models/sophia/run.sh
  fi
}

run_flashback() {
  local lengths=(256 512)
  [[ "${PROFILE}" == "full" ]] && lengths=(256 512 1024 2048 4096 16384 32768 65536)
  "${PYTHON}" benchmarks/baselines/flashback/bench.py \
    --preset gpt2 --seq-lens "${lengths[@]}" --head-dim 64 \
    --warmup "${WARMUP:-5}" --iters "${ITERS:-20}" \
    --csv "${OUT_DIR}/flashback-d64.csv"
  if [[ "${PROFILE}" == "full" ]]; then
    "${PYTHON}" benchmarks/baselines/flashback/bench.py \
      --preset gpt2 --seq-lens "${lengths[@]}" --head-dim 128 \
      --warmup "${WARMUP:-5}" --iters "${ITERS:-20}" \
      --csv "${OUT_DIR}/flashback-d128.csv"
  fi
}

run_target() {
  case "$1" in
    check) run_check ;;
    attention) run_attention ;;
    flashback) run_flashback ;;
    front-end) run_front_end ;;
    models) run_models ;;
    swa) run_swa ;;
    sophia) run_sophia ;;
    all) run_check; run_attention; run_models; run_swa ;;
    -h|--help|help) usage ;;
    *) echo "unknown target: $1" >&2; usage >&2; exit 2 ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  set -- all
fi
for target in "$@"; do
  run_target "${target}"
done
