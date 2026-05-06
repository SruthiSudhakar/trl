#!/usr/bin/env bash
set -euo pipefail
# """
# usage:
# ./examples/scripts/myscripts/evaluate_expert.sh --model_name_or_path outputs/jan29/PnPAll_20260129_222205/checkpoint-8000
# ./examples/scripts/myscripts/evaluate_expert.sh --model_name_or_path outputs/feb5/PnPAll_20260205_193405/checkpoint-20000
# """
# ==========================
# Defaults (can be overridden)
# ==========================
DEFAULT_MODEL_NAME_OR_PATH="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct"
DEFAULT_CUDA_DEVICES="0,1,2,3,4,5,6,7"   # comma-separated, e.g. "0,1,2,3"
DEFAULT_PAIR_MODE="failure"   # both | success | failure

# ==========================
# CLI args
# ==========================
MODEL_NAME_OR_PATH="${DEFAULT_MODEL_NAME_OR_PATH}"
CUDA_DEVICES_CSV="${DEFAULT_CUDA_DEVICES}"
PAIR_MODE="${DEFAULT_PAIR_MODE}"

usage() {
  cat <<EOF
Usage:
  $0 [--model_name_or_path PATH] [--cuda_devices CSV] [--pair_mode MODE]

Options:
  --pair_mode   both | success | failure  (default: failure)

Examples:
  $0
  $0 --model_name_or_path outputs/.../checkpoint-20000
  $0 --cuda_devices 0,1,2,3
  $0 --pair_mode failure --cuda_devices 6,7
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --cuda_devices)
      CUDA_DEVICES_CSV="$2"
      shift 2
      ;;
    --pair_mode)
      PAIR_MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# ==========================
# Dataset paths (parallel arrays — expert[i] pairs with nonexpert[i])
# ==========================

# ----- Expert: RoboCasa original datasets -----
EXPERT_DATASETS=(
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27"
)

# ----- Non-expert: Clip-based expert full-task datasets -----
NONEXPERT_DATASETS=(
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCoffeeServeMug"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPSinkToCounter"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToSink"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToCab"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPMicrowaveToCounter"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave"
)

NUM_PAIRS="${#EXPERT_DATASETS[@]}"
if [[ "${NUM_PAIRS}" -ne "${#NONEXPERT_DATASETS[@]}" ]]; then
  echo "Error: EXPERT_DATASETS (${NUM_PAIRS}) and NONEXPERT_DATASETS (${#NONEXPERT_DATASETS[@]}) must have the same length"
  exit 1
fi

# ==========================
# Parallel runner (1 job per GPU)
# ==========================
IFS=',' read -r -a GPUS <<< "${CUDA_DEVICES_CSV}"
NUM_GPUS="${#GPUS[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Error: --cuda_devices is empty"
  exit 1
fi

echo "Model:     ${MODEL_NAME_OR_PATH}"
echo "GPUs:      ${GPUS[*]} (parallelism = ${NUM_GPUS})"
echo "Pair mode: ${PAIR_MODE}"
echo "Jobs:      ${NUM_PAIRS}"

run_one() {
  local gpu="$1"
  local expert="$2"
  local nonexpert="$3"

  echo "=================================================="
  echo "GPU ${gpu} | expert: ${expert}"
  echo "         | nonexpert: ${nonexpert}"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES="${gpu}" python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_expert.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --expert_datasets "${expert}" \
    --nonexpert_datasets "${nonexpert}" \
    --split val \
    --compare_interval 4,8,12,16 \
    --batch_size 300 \
    --num_samples 1199 \
    --train_val_split_index 5 \
    --pair_mode "${PAIR_MODE}" \
    --visualize
}

# Requires bash 4.3+ for `wait -n` (common on modern Linux).
active=0
gpu_idx=0

for (( i=0; i<NUM_PAIRS; i++ )); do
  gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

  run_one "${gpu}" "${EXPERT_DATASETS[$i]}" "${NONEXPERT_DATASETS[$i]}" &
  active=$((active + 1))

  if [[ "${active}" -ge "${NUM_GPUS}" ]]; then
    # wait for any job to finish before launching more
    wait -n
    active=$((active - 1))
  fi
done

# wait for remaining jobs
wait
echo "All evaluations complete."
