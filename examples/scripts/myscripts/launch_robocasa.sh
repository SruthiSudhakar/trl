#!/usr/bin/env bash
# Launch SFT VLM training on robocasa policy rollouts (8x A6000, DeepSpeed ZeRO-2).
# Generalizable: set TASK to any task present in PRETRAIN_ROOT's rollout subdirs.
#
#   conda activate vlmoverlay
#   bash examples/scripts/myscripts/launch_robocasa.sh
#   # or retarget to another task:
#   TASK=CloseFridge bash examples/scripts/myscripts/launch_robocasa.sh
#
# Single camera per episode (one mp4 per rollout). Pairs that share an initial
# condition are matched by episode index across rollouts. The last
# NUM_EVAL_EPISODES initial conditions (that have successes) are held out for eval.

# Here the videos are 10 fps (see line 273: subsample: int = 1  # frame stride (videos are 10 fps)), and --subsample 1 means no skipping. The
#   pair gap is interval × subsample ÷ 10 fps:

#   ┌───────────────────────────┬───────────────┐
#   │ compare_interval (frames) │ seconds apart │
#   ├───────────────────────────┼───────────────┤
#   │ 3                         │ 0.3 s         │
#   ├───────────────────────────┼───────────────┤
#   │ 4                         │ 0.4 s         │
#   ├───────────────────────────┼───────────────┤
#   │ 8                         │ 0.8 s         │
#   ├───────────────────────────┼───────────────┤
#   │ 12                        │ 1.2 s         │
#   ├───────────────────────────┼───────────────┤
#   │ 16                        │ 1.6 s         │
#   └───────────────────────────┴───────────────┘

set -e
conda activate vlmoverlay

cd /proj/vondrick3/sruthi/Appaji/trl
PackDessert - cv13
PickPlaceCounterToDrawer - cv12
PickPlaceToasterOvenToCounter - cv11
TurnOnToaster

# --- Configure here -----------------------------------------------------------
TASK="PickPlaceToasterOvenToCounter"
PRETRAIN_ROOT="/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/released_checkpoints/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/evals/epoch=1400-test_mean_score=-1.000/pretrain"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
--config_file examples/accelerate_configs/deepspeed_zero2.yaml \
examples/scripts/myscripts/sft_vlm_robocasa.py \
--model_name_or_path /proj/vondrick3/sruthi/robots/sruthi_dreamitate/pretrained_vlms/Qwen2.5-VL-3B-Instruct/ \
--attn_implementation flash_attention_2 \
--pretrain_root "$PRETRAIN_ROOT" \
--task "$TASK" \
--num_eval_episodes 8 \
--output_dir "outputs/${TASK}_$(date +%Y%m%d_%H%M%S)_robocasa" \
--eval_strategy steps \
--logging_steps 10 \
--eval_steps 100 \
--save_steps 100 \
--gradient_accumulation_steps 2 \
--num_train_epochs 25 \
--learning_rate 1e-5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--compare_interval 4,8,12,16 \
--train_sample_interval 2 \
--failure_last_frac 0.95 \
--failure_min_frames 8 \
--max_succ_per_fail 1 \
--subsample 1 \
--eval_max_pairs 100 \
--report_to wandb \
--warmup_ratio 0.05 \
--max_pixels 848x480 \
--balance_fail_vs_succ True
