#!/usr/bin/env bash
set -e

dirs=(
  "PnPCabToCounter"
  "PnPCounterToCab"
  "PnPCoffeeServeMug"
  "PnPCounterToSink"
  "PnPSinkToCounter"
  "PnPMicrowaveToCounter"
  "PnPCounterToMicrowave"
  "PnPCounterToStove"
  "PnPStoveToCounter"
)

for dir in "${dirs[@]}"; do
  ls /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_${dir}
  # CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --gpu_ids=0 \
  #     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
  #     examples/scripts/myscripts/sft_vlm_overlay_regression_dp_compare_across_sf.py \
  #     --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
  #     --output_dir "outputs/TEST_$(date +%Y%m%d_%H%M%S)" \
  #     --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_${dir}" \
  #     --eval_strategy steps \
  #     --logging_steps 1 \
  #     --eval_steps 1 \
  #     --save_steps 1 \
  #     --gradient_accumulation_steps 1 \
  #     --num_train_epochs 500 \
  #     --learning_rate 1e-5 \
  #     --per_device_train_batch_size 8 \
  #     --per_device_eval_batch_size 8 \
  #     --report_to wandb \
  #     --split val \
  #     --train_sample_interval 1 \
  #     --compare_interval 4,12 \
  #     --max_exact_per_demo 50 \
  #     --binary_or_exact_gt binary \
  #     --just_prepare_data
    
    # CUDA_VISIBLE_DEVICES=1 python3 examples/scripts/myscripts/evaluate_sft_vlm_overlay_regression_dp.py \
    #   --model_name_or_path outputs/expert_allPnP_20260122_193654/checkpoint-40000 \
    #   --base_model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    #   --dataset_cache_file /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_${dir}/overlay_images_binary/dataset_cache_val_1__4_8_12_16__True_True.pkl \
    #   --batch_size 300 \
    #   --visualize \
    #   --num_visualize 10 \
    #   --seed 42 \
    #   --num_samples 500



done

# /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltaskPnPSinkToCounter