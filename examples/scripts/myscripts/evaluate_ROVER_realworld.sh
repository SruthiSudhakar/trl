#!/usr/bin/env bash
# ./examples/scripts/myscripts/evaluate_ROVER_realworld.sh --model_name gemini-3-flash-preview --num_samples 100

MODEL_NAME="gemini-3-flash-preview"
NUM_SAMPLES=1000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

EVAL="/workspace/hf_trl/trl/examples/scripts/myscripts/evaluate_ROVER_realworld.py"

PNP_DIRS=(
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


for dir in "${PNP_DIRS[@]}"; do
  echo "Running $dir..."

  python3 "$EVAL" \
    --model_name_or_path "$MODEL_NAME" \
    --base_dataset_path "$dir" \
    --split val \
    --num_samples "$NUM_SAMPLES" \
    --compare_interval 32,100 \
    --batch_size 100 \
    --visualize

  echo "Finished $dir"
  echo "-----------------------------"
done

echo "All jobs finished."