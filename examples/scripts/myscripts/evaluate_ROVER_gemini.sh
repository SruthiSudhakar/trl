#!/usr/bin/env bash
# ./examples/scripts/myscripts/evaluate_ROVER_gemini.sh --model_name gemini-3-flash-preview --num_samples 100

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

EVAL="/workspace/hf_trl/trl/examples/scripts/myscripts/evaluate_ROVER.py"
# ROOT="/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897"

PNP_DIRS=(
  # "feb7_na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCounterToSink_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPStoveToCounter_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCounterToStove_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPMicrowaveToCounter_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCounterToMicrowave_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCabToCounter_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCounterToCab_mg_fixed_224"
  # "feb7_na_na_16_mg_place_PnPCoffeeServeMug_mg_fixed_224"
    
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29"

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
    --visualize \
    --prefix ood

  echo "Finished $dir"
  echo "-----------------------------"
done

echo "All jobs finished."