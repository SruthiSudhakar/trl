#!/usr/bin/env bash
set -euo pipefail
# usage: ./examples/scripts/myscripts/evaluate.sh --model_name_or_path ProgressLM/qwen25vl_7b_nothink_multitask_merged_148
# ==========================
# Defaults (can be overridden)
# ==========================
DEFAULT_MODEL_NAME_OR_PATH="outputs/jan29/PnPAll_20260129_222205/checkpoint-9500"
DEFAULT_CUDA_DEVICES="0,2,3,4,5,6,7"   # comma-separated, e.g. "0,1,2,3"

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
# #   # ----- RoboCasa original datasets -----
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26"
#   "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27"

  # ----- Clip-based expert full-task datasets -----
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPCoffeeServeMug_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPSinkToCounter_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPCounterToSink_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPStoveToCounter_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCounterToStove_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPCabToCounter_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPCounterToCab_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPMicrowaveToCounter_mg_fixed_224"
  # "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_fulltask_PnPCounterToMicrowave_mg_fixed_224"

  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCounterToSink_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPStoveToCounter_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCounterToStove_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPMicrowaveToCounter_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCounterToMicrowave_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCabToCounter_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCounterToCab_mg_fixed_224"
  "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPCoffeeServeMug_mg_fixed_224"

  # Untrained dataset
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24"
  # "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29"

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

  CUDA_VISIBLE_DEVICES="${gpu}" python3 examples/scripts/myscripts/evaluate_progressLM.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --base_dataset_path "${dataset}" \
    --split val \
    --compare_interval 16,30,50,74,90,100 \
    --batch_size 100 \
    --num_samples 200 \
    --train_val_split_index 5 \
    --visualize --nothink
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
