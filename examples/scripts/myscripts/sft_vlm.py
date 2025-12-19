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
#     "Pillow>=9.4.0",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install pillow

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --output_dir "outputs/sep24/sft-Qwen2.5-VL-7B-Instruct_$(date +%Y%m%d_%H%M%S)" \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 200 \
    --logging_steps 50 \
    --eval_steps 50 \
    --save_steps 50 \
    --eval_strategy steps \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --report_to wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --output_dir "outputs/sep24/sft-Qwen2.5-VL-7B-Instruct_$(date +%Y%m%d_%H%M%S)" \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1000 \
    --logging_steps 50 \
    --eval_steps 50 \
    --save_steps 50 \
    --eval_strategy steps \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --report_to wandb
    
  Training Parameters                                                                                                                                              [32/283]
                                                                                                                                                                           
  - --per_device_train_batch_size (default: 8) - Batch size per GPU                                                                                                        
  - --per_device_eval_batch_size (default: 8) - Eval batch size per GPU                                                                                                    
  - --gradient_accumulation_steps (default: 1) - Accumulate gradients over N steps                                                                                         
  - --num_train_epochs (default: 3) - Number of training epochs                                                                                                            
  - --max_steps - Override epochs with max training steps                                                                                                                  
  - --learning_rate (default: 5e-5) - Initial learning rate                                                                                                                
  - --warmup_steps or --warmup_ratio - Learning rate warmup                                                                                                                
  - --weight_decay (default: 0) - L2 regularization                                                                                                                        
  - --gradient_checkpointing - Trade compute for memory                                                                                                                    
                                                                                                                                                                           
  Optimizer & Scheduler                                                                                                                                                    
                                                                                                                                                                           
  - --optim (default: "adamw_torch") - Options: adamw_torch, adamw_hf, sgd, adafactor                                                                                      
  - --lr_scheduler_type (default: "linear") - Options: linear, cosine, polynomial                                                                                          
  - --adam_beta1 (default: 0.9) - Adam beta1                                                                                                                               
  - --adam_beta2 (default: 0.999) - Adam beta2                                                                                                                             
  - --adam_epsilon (default: 1e-8) - Adam epsilon                                                                                                                          
                                                                                                                                                                           
  Mixed Precision                                                                                                                                                          
                                                                                                                                                                           
  - --bf16 - Use bfloat16 precision (recommended for A100/H100)                                                                                                            
  - --fp16 - Use float16 precision                                                                                                                                         
  - --tf32 - Enable TensorFloat-32 on Ampere GPUs                                                                                                                          
                                                                                                                                                                           
  SFT-Specific                                                                                                                                                             
                                                                                                                                                                           
  - --max_length (default: 1024) - Max sequence length
  - --completion_only_loss - Only compute loss on completions
  - --dataset_text_field (default: "text") - Dataset text column name
  - --dataset_num_proc - Parallel dataset processing

  Saving & Logging

  - --output_dir - Where to save model
  - --save_steps - Save checkpoint every N steps
  - --save_total_limit - Max checkpoints to keep
  - --logging_steps (default: 500) - Log metrics every N steps
  - --eval_steps - Run evaluation every N steps
  - --eval_strategy - Options: no, steps, epoch
  - --report_to - Options: wandb, tensorboard, none
  - --push_to_hub - Push model to HuggingFace Hub

  PEFT/LoRA

  - --use_peft - Enable LoRA fine-tuning
  - --lora_r (default: 16) - LoRA rank
  - --lora_alpha (default: 32) - LoRA alpha
  - --lora_dropout (default: 0.1) - LoRA dropout
  - --lora_target_modules - Which modules to apply LoRA to

    
"""

import os
import pdb
import json
import hashlib
import gc
import logging
from pathlib import Path
import time

import torch
from datasets import load_dataset, Dataset
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import TrainerCallback

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OutputLoggingCallback(TrainerCallback):
    """Callback to log model inputs/outputs occasionally during training."""

    def __init__(self, log_every=100):
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        # Log sample outputs every N steps
        if state.global_step % self.log_every == 0 and state.global_step > 0:
            if 'inputs' in kwargs:
                # Try to decode and log a sample input/output
                try:
                    inputs = kwargs.get('inputs')
                    outputs = kwargs.get('outputs')

                    logger.info(f"\n{'='*60}")
                    logger.info(f"[Step {state.global_step}] Sample Input/Output:")

                    # Log the messages if available
                    if hasattr(inputs, 'get') and 'messages' in inputs:
                        messages = inputs['messages'][0] if isinstance(inputs['messages'], list) else inputs['messages']
                        logger.info(f"Input messages: {messages}")

                    # Log the model's prediction if available
                    if outputs is not None:
                        if hasattr(outputs, 'logits'):
                            # Get the predicted token
                            predicted_tokens = torch.argmax(outputs.logits, dim=-1)
                            logger.info(f"Model prediction shape: {predicted_tokens.shape}")
                            # Show first prediction sample
                            if len(predicted_tokens.shape) > 0:
                                logger.info(f"Sample prediction tokens: {predicted_tokens[0][:10].tolist()}...")

                    logger.info(f"Loss at this step: {kwargs.get('logs', {}).get('loss', 'N/A')}")
                    logger.info(f"{'='*60}\n")
                except Exception as e:
                    logger.debug(f"Could not log outputs: {e}")

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Basic loss logging
        if logs and 'loss' in logs and state.global_step % 10 == 0:
            logger.info(f"[Step {state.global_step}] Loss: {logs['loss']:.4f}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.gradient_checkpointing = False
    training_args.max_length = None

    logger.info("="*80)
    logger.info("Starting VLM SFT Training Script")
    logger.info("="*80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Training parameters:")
    logger.info(f"  - Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    logger.info(f"  - Logging steps: {training_args.logging_steps}")
    logger.info(f"  - Eval steps: {training_args.eval_steps if training_args.eval_strategy != 'no' else 'Disabled'}")
    logger.info(f"  - Save steps: {training_args.save_steps}")
    logger.info(f"  - Mixed precision dtype: {model_args.dtype}")
    logger.info(f"  - Use PEFT: {model_args.use_peft}")

    ################
    # Model, Tokenizer & Processor
    ################
    logger.info("\nLoading model and tokenizer...")
    start_time = time.time()

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'num_parameters'):
        logger.info(f"Total parameters: {model.num_parameters()/1e9:.2f}B")

    ################
    # Dataset
    ################
    logger.info("\nPreparing dataset...")
    # Comment out the original HuggingFace dataset loading
    # dataset = load_dataset('HuggingFaceH4/llava-instruct-mix-vsft', name=script_args.dataset_config)

    dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5'
    cache_dir = Path('/workspace/image_cache')

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Cache directory: {cache_dir}")

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
    logger.info(f"Loading metadata from: {metadata_file}")
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
    logger.info(f"Loading pre-extracted images from {image_dir} for {split} split")
    logger.info(f"Number of demos selected: {len(selected_demos)}")

    if not selected_demos:
        raise ValueError(f"No demos found for {split} split!")
    
    task_description = "Pick and place an object from the sink to the plate on the counter"
    SYSTEM_PROMPT =f"""You are an expert roboticist tasked to predict task completion "
        percentage for a frame of a robot for the task of {task_description}.
        The task completion percentages are between 0 and 100, where 100
        corresponds to full task completion.
    """
    problem = f"""Here is a frame showing the robot performing the task. The frame shows the robot's current state.\n
                For the task of {task_description}, output the task completion percentage (an integer from 0-100) for this current frame.
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

            # Format for SFTTrainer: conversational format with images (plural)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"{progress}"}
            ]

            data.append({
                "images": [Image.open(idx_image_path)],  # Changed to "images" (plural)
                "messages": messages  # Using messages format for conversational data
            })
    print(f"Loaded {len(data)} image pairs from {len(selected_demos)} {split} demos")

    dataset = Dataset.from_list(data)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=100, seed=42, shuffle=True)

    # Filter have big images
    def filter_big_images(example):
        images = example["images"]  # Changed to "images" (plural)
        for image in images:
            if image.size[0] >= 512 or image.size[1] >= 512:
                return False
        return True
    dataset = dataset.filter(filter_big_images)

    def convert_to_rgb(example):
        images = example["images"]  # Changed to "images" (plural)
        for idx, image in enumerate(images):
            if image.mode != "RGB":
                image = image.convert("RGB")
            example["images"][idx] = image  # Changed to "images" (plural)
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42) if training_args.eval_strategy != "no" else None
    ################
    # Training
    ################
    # # Create memory management callback
    # memory_callback = MemoryManagementCallback(gc_steps=10)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        # callbacks=[memory_callback],  # Add the memory management callback
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
