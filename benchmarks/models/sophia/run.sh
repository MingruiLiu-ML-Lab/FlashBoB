#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TORCHRUN="${TORCHRUN:-torchrun}"
SOPHIA_OUT_DIR="${SOPHIA_OUT_DIR:-runs/sophia}"
SOPHIA_SUITE="${SOPHIA_SUITE:-fineweb}"
WANDB_ARGS=()
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT:-flashbob}")
fi
EXTRA_ARGS=("$@")

DATA_ARGS=(
  --dataset custom
  --dataset-id VisionTheta/fineweb-1B
  --dataset-name default
  --dataset-revision cb8cee93f1edeeead9529bbfd561c2321431c938
  --no-streaming
)
TRAIN_ARGS=(
  --grad-clip 1.0
  --num-workers 0
  --eval-every 250
  --eval-batches 32
  --log-every 1
  --save-every 1000
  --dtype bfloat16
  --device cuda
  --seed 17
  --out-dir "${SOPHIA_OUT_DIR}"
)
SOPHIA_ARGS=(
  --optimizer sophiah
  --hutch-batch-size 32
  --hutch-samples 1
  --hutch-mode microbatch
  --lr 0.0006
  --min-lr 0.00003
  --warmup-steps 2000
  --weight-decay 0.2
  --beta1 0.96
  --beta2 0.99
  --gamma 0.01
  --eps 1e-12
)

run_job() {
  local processes="$1"
  shift
  "${TORCHRUN}" --standalone --nproc_per_node="${processes}" \
    -m benchmarks.models.sophia.train \
    "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${WANDB_ARGS[@]}" \
    "$@" "${EXTRA_ARGS[@]}"
}

run_fineweb() {
  local lr min_lr tag
  for tag in 3em4 4p5em4 6em4 8em4 1em3; do
    case "${tag}" in
      3em4) lr=0.0003; min_lr=0.00003 ;;
      4p5em4) lr=0.00045; min_lr=0.000045 ;;
      6em4) lr=0.0006; min_lr=0.00006 ;;
      8em4) lr=0.0008; min_lr=0.00008 ;;
      1em3) lr=0.001; min_lr=0.0001 ;;
    esac
    run_job 8 \
      --run-name "fineweb-adamw-lr-${tag}" \
      --preset gpt2 --block-size 2048 --tokenizer-name gpt2 \
      --train-token-budget 1000000000 --batch-size 2 --grad-accum-steps 16 \
      --eval-batch-size 2 --optimizer adamw --attn-backend flash \
      --lr "${lr}" --min-lr "${min_lr}" --warmup-steps 150 \
      --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8
  done

  run_sophia_pair gpt2 2048 10 8 2 16 1000000000 \
    tiktoken gpt2 "" fineweb-gpt2
}

run_sophia_pair() {
  local preset="$1" block_size="$2" interval="$3" processes="$4"
  local batch_size="$5" accum="$6" token_budget="$7"
  local tokenizer_backend="$8" tokenizer_name="$9" tokenizer_revision="${10}"
  local prefix="${11}"
  local tokenizer_args=(
    --tokenizer-backend "${tokenizer_backend}"
    --tokenizer-name "${tokenizer_name}"
  )
  if [[ -n "${tokenizer_revision}" ]]; then
    tokenizer_args+=(--tokenizer-revision "${tokenizer_revision}")
  fi

  run_job "${processes}" \
    --run-name "${prefix}-math-hess${interval}" \
    --preset "${preset}" --block-size "${block_size}" \
    "${tokenizer_args[@]}" --train-token-budget "${token_budget}" \
    --batch-size "${batch_size}" --grad-accum-steps "${accum}" \
    --eval-batch-size "${batch_size}" --attn-backend math \
    --non-bob-attn-backend flash --hess-interval "${interval}" \
    "${SOPHIA_ARGS[@]}"
  run_job "${processes}" \
    --run-name "${prefix}-bob-hess${interval}" \
    --preset "${preset}" --block-size "${block_size}" \
    "${tokenizer_args[@]}" --train-token-budget "${token_budget}" \
    --batch-size "${batch_size}" --grad-accum-steps "${accum}" \
    --eval-batch-size "${batch_size}" --attn-backend bob \
    --hess-interval "${interval}" "${SOPHIA_ARGS[@]}"
}

run_pythia_intervals() {
  local preset="$1" prefix="$2"
  local budget=$((1908 * 8 * 2 * 16 * 2048))
  local interval
  for interval in 10 5 2; do
    run_sophia_pair "${preset}" 2048 "${interval}" 8 2 16 "${budget}" \
      huggingface EleutherAI/pythia-160m \
      50f5173d932e8e61f858120bcb800b97af589f46 "${prefix}"
  done
}

run_llama() {
  local budget=$((6 * 4096))
  run_sophia_pair llama-3.2-1b 4096 1 1 1 1 "${budget}" \
    huggingface NousResearch/Meta-Llama-3-8B \
    315b20096dc791d381d514deb5f8bd9c8d6d3061 llama-3.2-1b-n4096
}

case "${SOPHIA_SUITE}" in
  fineweb) run_fineweb ;;
  pythia) run_pythia_intervals pythia-160m pythia-160m ;;
  pythia-d128) run_pythia_intervals pythia-160m-d128 pythia-160m-d128 ;;
  llama) run_llama ;;
  all)
    run_fineweb
    run_pythia_intervals pythia-160m pythia-160m
    run_pythia_intervals pythia-160m-d128 pythia-160m-d128
    run_llama
    ;;
  *)
    echo "SOPHIA_SUITE must be fineweb, pythia, pythia-d128, llama, or all" >&2
    exit 2
    ;;
esac
