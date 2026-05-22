#!/usr/bin/env bash
# Launch SFT VLM training on BagPlate (8x A6000, DeepSpeed ZeRO-2).
# Same as launch_bag_plate.sh, but only fail-vs-succ pairs (no succ-vs-succ).
#
# Activate the env first:
# conda activate vlmoverlay
#
# Then:
# bash examples/scripts/myscripts/launch_bag_plate_fail_only.sh

set -e
conda activate vlmoverlay

cd /proj/vondrick3/sruthi/Appaji/trl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/myscripts/sft_vlm_bag_plate.py \
    --model_name_or_path /proj/vondrick3/sruthi/robots/sruthi_dreamitate/pretrained_vlms/Qwen2.5-VL-3B-Instruct/ \
    --attn_implementation flash_attention_2 \
    --dataset_root /proj/vondrick3/datasets/expert_data_jgd_bag_plate \
    --task_name BagPlate \
    --box_indices 1-20 \
    --glass_indices 1-20 \
    --remote_indices 1-20 \
    --eval_box_indices 18-20 \
    --eval_glass_indices 18-20 \
    --eval_remote_indices 18-20 \
    --output_dir "outputs/BagPlate_failonly_$(date +%Y%m%d_%H%M%S)" \
    --eval_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 50 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --compare_interval 3,4,8,12,16 \
    --train_sample_interval 2 \
    --failure_last_frac 0.95 \
    --failure_min_frames 8 \
    --eval_max_pairs 200 \
    --report_to wandb \
    --warmup_ratio 0.05 \
    --max_pixels 960x540 \
    --balance_fail_vs_succ True \
    --include_succ_vs_succ False
