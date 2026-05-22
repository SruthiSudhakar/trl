#!/usr/bin/env bash
# Launch SFT VLM training on PushBowl (8x A6000, DeepSpeed ZeRO-2).
# Uses BOTH camera views: every comparison is emitted once per camera, so
# train/eval are balanced 50/50 across cam0 and cam1.
#
# Activate the env first:
# conda activate vlmoverlay
#
# Then:
# bash examples/scripts/myscripts/launch_push_bowl.sh
#
# Dataset: 30 success demos (1-30), 11 failure demos (1-11).
# Strict held-out eval: success 28-30 / failure 9-11 are NOT in train.

set -e
conda activate vlmoverlay

cd /proj/vondrick3/sruthi/Appaji/trl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
--config_file examples/accelerate_configs/deepspeed_zero2.yaml \
examples/scripts/myscripts/sft_vlm_push_bowl.py \
--model_name_or_path /proj/vondrick3/sruthi/robots/sruthi_dreamitate/pretrained_vlms/Qwen2.5-VL-3B-Instruct/ \
--attn_implementation flash_attention_2 \
--dataset_root /proj/vondrick3/datasets/expert_data_jgd_push_bowl \
--task_name PushBowl \
--failure_indices 1-11 \
--success_indices 1-30 \
--eval_failure_indices 9-11 \
--eval_success_indices 28-30 \
--output_dir "outputs/PushBowl_$(date +%Y%m%d_%H%M%S)" \
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
--balance_fail_vs_succ True
