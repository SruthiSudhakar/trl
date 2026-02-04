#!/usr/bin/env bash
set -euo pipefail

dirs=(
  # "PnPCabToCounter"
  # "PnPCounterToCab"
  # "PnPCoffeeServeMug"
  # "PnPCounterToSink"
  # "PnPSinkToCounter"
  # "PnPMicrowaveToCounter"
  # "PnPCounterToMicrowave"
  # "PnPCounterToStove"
  "PnPStoveToCounter"
)

MODEL="outputs/jan29/PnPAll_20260129_222205/checkpoint-9500"
BASE_ROOT="/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897"

export MODEL BASE_ROOT

# Print "index dir" pairs, then run up to 8 in parallel.
for i in "${!dirs[@]}"; do
  echo "$i ${dirs[$i]}"
done | xargs -n 2 -P 8 bash -lc '
  i="$0"; dir="$1"
  gpu=$(( i % 8 ))
  
  # Sleep to avoid race condition on output dir creation (timestamp based)
  sleep $((i * 5))

  echo "[gpu=${gpu}] starting ${dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_v2.py \
    --model_name_or_path "${MODEL}" \
    --base_dataset_path "${BASE_ROOT}/na_na_16_expert_fulltask_${dir}" \
    --split val \
    --compare_interval 4,8 \
    --batch_size 300 \
    --num_samples 1199 \
    --train_val_split_index 5 \
    --visualize
'
