#!/usr/bin/env bash
# Launch SFT VLM training on PnPRedLegoToBrownBowl (8x A6000, DeepSpeed ZeRO-3).
#
# Activate the env first:
# conda activate vlmoverlay
#
# Then:
# bash examples/scripts/myscripts/launch_lego.sh

#   With completion_only_loss=True, only ~3 tokens per example contribute to the loss instead of ~700. Per-step gradient is
#    now much sparser, so a bigger effective batch reduces noise and stabilizes training (you saw grad_norm=171 early in   
#   run 2 — that's the kind of variance bigger batches damp out).                                                          
                  
#   Concrete suggestion:
#   - per_device_train_batch_size 4, gradient_accumulation_steps 2 on 8 GPUs → effective batch 64. ✓ Good.
#   - Going higher (accum=4 → batch 128) is fine if memory permits, but with ~1500 train pairs you'd be down to ~12        
#   optimizer steps per epoch, which gets coarse for tracking with eval_steps=90.                                  
#   - Don't bother going below accum=1 (effective batch 32) unless you OOM.                                                
                  
#   Two related things to flag:                                                                                            
#   1. Drop epochs. 100 epochs at this batch size = ~2300 steps. With completion_only_loss, the model converges much faster
#    on the answer signal — most likely you'll hit best eval sign accuracy in the first 5–15 epochs, then start            
#   overfitting. Your load_best_model_at_end + metric_for_best_model="eval_sign_acc_overall" config will save the best
#   checkpoint regardless, so nothing breaks, but you're burning compute. Consider dropping to ~20 epochs unless you see   
#   late gains.                                                                                                         
#   2. Watch grad_norm. If it stays >50 even after the first few hundred steps with completion-only loss, drop LR to 5e-6
#   or add --warmup_ratio 0.03. The previous huge spikes were partly an artifact of the broken loss landscape.

set -e
conda activate vlmoverlay

cd /proj/vondrick3/sruthi/Appaji/trl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
--config_file examples/accelerate_configs/deepspeed_zero2.yaml \
examples/scripts/myscripts/sft_vlm_lego.py \
--model_name_or_path /proj/vondrick3/sruthi/robots/sruthi_dreamitate/pretrained_vlms/Qwen2.5-VL-3B-Instruct/ \
--attn_implementation flash_attention_2 \
--dataset_root /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl \
--task_name PnPRedLegoToBrownBowl \
--failure_indices 1-20 \
--success_indices 1-50 \
--eval_failure_indices 18-20 \
--eval_success_indices 18-20 \
--output_dir "outputs/PnPRedLegoToBrownBowl_$(date +%Y%m%d_%H%M%S)_wr" \
--eval_strategy steps \
--logging_steps 10 \
--eval_steps 50 \
--save_steps 50 \
--gradient_accumulation_steps 2 \
--num_train_epochs 100 \
--learning_rate 1e-5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--compare_interval 4,8,12,16 \
--train_sample_interval 8 \
--failure_last_frac 0.50 \
--failure_min_frames 8 \
--eval_max_pairs 200 \
--report_to wandb \
--warmup_ratio 0.03 \
--max_pixels 960x540


Todo tmr:
cv12
0. DP rollouts and check its all good 
cv14: claude --resume 330fbcb4-326c-4e9b-80a7-09a97b2b9433
1. switch to 7b qwen model 
2. train vlm with success and failure (make sure mae threshold works, or just dont care)
3. train vlm with just success
