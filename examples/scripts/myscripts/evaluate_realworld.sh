#!/usr/bin/env bash
set -euo pipefail
# usage: ./examples/scripts/myscripts/evaluate_realworld.sh --model_name_or_path outputs/realworld_all_20260223_162917/checkpoint-3800
# ==========================
# Defaults (can be overridden)
# ==========================
DEFAULT_MODEL_NAME_OR_PATH="outputs/realworld_all_20260223_162917/checkpoint-3800"
DEFAULT_CUDA_DEVICES="6,7"   # comma-separated, e.g. "0,1,2,3"

# ==========================
# CLI args
# ==========================
MODEL_NAME_OR_PATH="${DEFAULT_MODEL_NAME_OR_PATH}"
CUDA_DEVICES_CSV="${DEFAULT_CUDA_DEVICES}"

usage() {
  cat <<EOF
Usage:
  $0 [--model_name_or_path PATH] [--cuda_devices CSV]

Examples:
  $0
  $0 --model_name_or_path outputs/.../checkpoint-20000
  $0 --cuda_devices 0,1,2,3
  $0 --model_name_or_path outputs/.../checkpoint-20000 --cuda_devices 6,7
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
# Dataset paths
# ==========================
BASE_DATASET_PATHS=(
    "realworld_dataset/BimanualBikeRotorInstall-Nominal-real"
    "realworld_dataset/BimanualClearKitchenCounter-Nominal-real"
    "realworld_dataset/BimanualSetUpBreakfastTable-Nominal-real"
    "realworld_dataset/CleanLitterBox-Nominal-real"
    "realworld_dataset/CutAppleIntoSlices-Nominal-real"
    "realworld_dataset/PushCoasterToMug-Nominal-real"
    "realworld_dataset/PushCoasterToMug-ObjectCentricDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTable-StationDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTableSeenTasks_backfill-salem--video"
    "realworld_dataset/TurnMugRightsideUp-Nominal-real"
    "realworld_dataset/TurnMugRightsideUp-ObjectCentricDistributionShift-real"
    "realworld_dataset/TurnMugRightsideUp-StationDistributionShift-real"
)

# ==========================
# Parallel runner (1 job per GPU)
# ==========================
IFS=',' read -r -a GPUS <<< "${CUDA_DEVICES_CSV}"
NUM_GPUS="${#GPUS[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Error: --cuda_devices is empty"
  exit 1
fi

echo "Model: ${MODEL_NAME_OR_PATH}"
echo "GPUs:  ${GPUS[*]} (parallelism = ${NUM_GPUS})"
echo "Jobs:  ${#BASE_DATASET_PATHS[@]}"

run_one() {
  local gpu="$1"
  local dataset="$2"

  echo "=================================================="
  echo "GPU ${gpu} | dataset: ${dataset}"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES="${gpu}" python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_realworld.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --base_dataset_path "${dataset}" \
    --split val \
    --batch_size 100 \
    --num_samples 1199 \
    --compare_interval 32,100 \
    --visualize
}

# Requires bash 4.3+ for `wait -n` (common on modern Linux). If your shell is older, tell me and I’ll give a portable fallback.
active=0
gpu_idx=0

for dataset in "${BASE_DATASET_PATHS[@]}"; do
  gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

  run_one "${gpu}" "${dataset}" &
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


# CUDA_VISIBLE_DEVICES=6 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_realworld.py \
#   --model_name_or_path outputs/jan29/PnPAll_20260129_222205/checkpoint-9500 \
#   --base_dataset_path realworld_dataset/BimanualBikeRotorInstall-Nominal-real \
#   --split val \
#   --batch_size 100 \
#   --num_samples 1199 \
#   --compare_interval 32,36 \
#   --visualize

