# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=4 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --output_dir "outputs/grpo-Qwen2.5-VL-7B-Instruct_$(date +%Y%m%d_%H%M%S)" \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --log_completions \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10000 \
    --per_device_train_batch_size 8 \
    --save_steps 50 \
    --eval_steps 50 \
    --report_to wandb

GRPOConfig parameters (training-specific):
  - learning_rate: 1e-6                                                                                                                                                 
  - num_train_epochs: 3 (inherited from TrainingArguments)                                                                                                              
  - per_device_train_batch_size: 8 (inherited from TrainingArguments)                                                                                                   
  - gradient_accumulation_steps: 1 (inherited from TrainingArguments)                                                                                                   
  - gradient_checkpointing: True                                                                                                                                        
  - logging_steps: 10                                                                                                                                                   
  - num_generations: 8                                                                                                                                                  
  - max_prompt_length: 512                                                                                                                                              
  - max_completion_length: 256
  - temperature: 1.0
  - top_p: 1.0
  - top_k: None (disabled)
  - beta: 0.0 (no KL penalty)
  - epsilon: 0.2
  - loss_type: "dapo"
  - scale_rewards: "group"
  - bf16: True (if fp16 not set)
  - use_vllm: False
  - vllm_mode: "server"
  - vllm_gpu_memory_utilization: 0.3
  - log_completions: False
  - mask_truncated_completions: False
  - top_entropy_quantile: 1.0
  - --output_dir - Directory to save the model                                                                                                                          
  - --save_steps - Save checkpoint every N steps                                                                                                                        
  - --eval_steps - Run evaluation every N steps                                                                                                                         
  - --warmup_steps - Number of warmup steps                                                                                                                             
  - --log_completions - Log completions during training                                                                                                                 
  - --push_to_hub - Push model to Hugging Face Hub                                                                                                                      

  ModelConfig parameters:
  - --model_name_or_path - Path to the base model
  - --dtype - Data type (bfloat16, float16, auto)
  - --use_peft - Enable LoRA/PEFT (false)
  - --lora_r - LoRA rank (default: 16)
  - --lora_alpha - LoRA alpha parameter (32)
  - --lora_dropout - LoRA dropout
  - --lora_target_modules - Which modules to apply LoRA to (e.g., "q_proj", "v_proj")
  - --attn_implementation - Attention implementation (flash_attention_2, sdpa, eager)

  vLLM-specific parameters:
  - --use_vllm - Use vLLM for generation
  - --vllm_mode - vLLM mode ("colocate" or "server")
  - --vllm_tensor_parallel_size - Tensor parallel size for vLLM

"""

import os
import json
from PIL import Image
from datasets import Dataset
from pathlib import Path
import hashlib
import pdb
import re

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5'
    cache_dir = Path('/workspace/image_cache')
    
    # Generate unique cache subdirectory based on dataset path
    dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    image_dir = cache_dir / dataset_hash
    
    # Check if images have been extracted
    cache_marker = image_dir / '.cache_complete'
    metadata_file = image_dir / 'metadata.json'
    
    if not cache_marker.exists():
        raise RuntimeError(
            f"Images not extracted yet! Please run:\n"
            f"python extract_images_from_hdf5.py --hdf5-path '{dataset_path}' --output-dir '{cache_dir}'\n"
            f"Expected cache directory: {image_dir}"
        )
    
    # Load metadata
    metadata_file="/workspace/image_cache/7b998b96/metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check if split information exists
    if 'split' not in metadata:
        raise RuntimeError(
            f"Split information not found in metadata! Please run:\n"
            f"python split_demos.py --cache-dir '{cache_dir}' --dataset-path '{dataset_path}'\n"
        )
    
    split_info = metadata['split']
    split='train'

    if split == 'val':
        selected_demos = split_info['val_demos']
    else:  # train
        selected_demos = split_info['train_demos']
    print(f"Loading pre-extracted images from {image_dir} for {split} split")
    
    if not selected_demos:
        raise ValueError(f"No demos found for {split} split!")
    
    task_description = "Pick and place an object from the sink to the plate on the counter"
    SYSTEM_PROMPT =f"""You are an expert roboticist tasked to predict task completion "
        percentage for a frame of a robot for the task of {task_description}.
        The task completion percentages are between 0 and 100, where 100
        corresponds to full task completion.
    """
    # problem = f"""Here are two frames from a robot performing a task.\n
    #             Frame 1 (initial): Shows the robot at the start state. the task completion percentage is 0.\n
    #             Frame 2 (current): Shows the robot's current state.\n
    #             For the task of {task_description}, output the task completion percentage (an integer from 0-100) for Frame 2. 
    #             Please answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>. 
    # """"
    problem = f"""Here is a frame showing the robot performing the task. The frame shows the robot's current state.\n
                For the task of {task_description}, output the task completion percentage (an integer from 0-100) for this current frame. 
                Please answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>. 
    """
    data = []
    for demo_name in selected_demos:
        demo_info = metadata['demos'][demo_name]
        num_frames = demo_info['num_frames']
        demo_dir = image_dir / demo_name
        
        first_image_path = str(demo_dir / 'frame_0000.png')
        
        for idx in range(0, num_frames, 16):
            idx_image_path = str(demo_dir / f'frame_{idx:04d}.png')
            progress = str(int(idx / num_frames * 100))  # Progress as 0-100
            data.append({
                "image": [Image.open(idx_image_path)], #Image.open(first_image_path),
                "problem": problem,
                "solution": progress
            })
    print(f"Loaded {len(data)} image pairs from {len(selected_demos)} {split} demos")

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=100, seed=42)

    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Filter have big images
    def filter_big_images(example):
        images = example["image"]
        for image in images:
            if image.size[0] >= 512 or image.size[1] >= 512:
                return False
        return True
    dataset = dataset.filter(filter_big_images)

    def convert_to_rgb(example):
        images = example["image"]
        for idx,image in enumerate(images):
            if image.mode != "RGB":
                image = image.convert("RGB")
            example["image"][idx] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    ################
    # Reward Function for Training
    ################
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
                    if not answer_match:
                        reward = float(0.0)
                    answer_text = answer_match.group(1).strip()
                    number_match = re.search(r'\b(\d{1,3})\b', answer_text)
                    if not number_match:
                        reward = float(0.0)
                    extracted_number = int(number_match.group(1))
                    if not (0 <= extracted_number <= 100):
                        reward = float(0.0)
                    if (int(sol) - 5) < extracted_number < (int(sol) + 5):
                        reward = float(1.0)
                    else:
                        reward = float(0.0)
                    # answer_parsed = parse(
                    #     content,
                    #     extraction_config=[
                    #         LatexExtractionConfig(
                    #             normalization_config=NormalizationConfig(
                    #                 nits=False,
                    #                 malformed_operators=False,
                    #                 basic_latex=True,
                    #                 boxed="all",
                    #                 units=True,
                    #             ),
                    #             boxed_match_priority=0,
                    #             try_extract_without_anchor=False,
                    #         )
                    #     ],
                    #     extraction_mode="first_match",
                    # )

                    # reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}")#, answer: {content}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(content.strip().lower() == sol.strip().lower())

            rewards.append(reward)
        return rewards

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


"""

"""