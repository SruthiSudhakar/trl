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
#     "numpy",
# ]
# ///

"""
Image Overlay SFT Script for Task Completion Progress Regression - OPTIMIZED VERSION

This script trains a VLM to analyze side-by-side images from a robot demonstration
where two frames are put side-by-side and predict the relative task completion progress between them.

Optimizations:
- Image caching to avoid repeated disk I/O
- Parallel overlay generation
- Persistent overlay cache
- Optimized numpy operations



Usage:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_dp_compare_across_sf.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --output_dir "outputs/expert_allPnP_$(date +%Y%m%d_%H%M%S)" \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove" \
    --eval_strategy steps \
    --logging_steps 500 \
    --eval_steps 500 \
    --save_steps 500 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 500 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --report_to wandb \
    --split train \
    --train_sample_interval 1 \
    --compare_interval 4,8,12,16 \
    --max_exact_per_demo 50 \
    --binary_or_exact_gt binary \
    --task_description ""

just to load data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_dp_compare_across_sf.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --output_dir "outputs/TEST_$(date +%Y%m%d_%H%M%S)" \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave" \
    --eval_strategy steps \
    --logging_steps 1 \
    --eval_steps 1 \
    --save_steps 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 500 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --report_to wandb \
    --split val \
    --train_sample_interval 1 \
    --compare_interval 4,8,12,16 \
    --max_exact_per_demo 50 \
    --binary_or_exact_gt binary \
    --task_description ""
    --just_prepare_data

"""

import os
import json
import hashlib
import random
# Set random seed for reproducibility
random.seed(42)
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pickle
import textwrap

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from tqdm import tqdm
import datasets as hf_datasets
import logging
from pathlib import Path
import pdb

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
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
from trl.data_utils import prepare_multimodal_messages
from transformers import TrainerCallback
import time
import importlib.util
import ast
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from collections import defaultdict

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=2048)
def _load_and_prepare_image(image_path):
    """Cache loaded images to avoid repeated disk I/O."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32)

@lru_cache(maxsize=1000)
def _get_frame_numbers(variant_frames_dir):
    """Cache frame number lists per variant directory to avoid repeated os.listdir calls."""
    return [int(f.split("_")[1].split(".")[0]) for f in os.listdir(variant_frames_dir)]

def create_overlay_image(image1_path, image2_path, method='side_by_side', use_cache=True):
    """
    Creates an overlayed/combined image from two input images using various methods.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        method: Overlay method to use:
            - 'side_by_side': Places images horizontally next to each other
            - 'difference': Shows the absolute difference between images
            - 'blend': Simple 50/50 alpha blend
            - 'checkerboard': Alternating checkerboard pattern from both images
            - 'color_channels': Red/green channel overlay (original method)
            - 'edge_overlay': Shows edges from both images in different colors
            - 'temporal_fade': Fades from image1 on left to image2 on right
        use_cache: Whether to use the image loading cache

    Returns:
        PIL Image: The combined/overlayed image
    """
    # Load images using cached function
    if use_cache:
        arr1 = _load_and_prepare_image(image1_path)
        arr2 = _load_and_prepare_image(image2_path)
    else:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

    # Ensure both images have the same size
    if arr1.shape != arr2.shape:
        # Convert back to PIL for resizing
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)
            arr2 = np.array(img2, dtype=np.float32)

    if method == 'side_by_side':
        # Place images side by side with a separator line
        height, width = arr1.shape[:2]
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))
        combined = Image.new('RGB', (width * 2 + 2, height))
        combined.paste(img1, (0, 0))
        combined.paste(Image.new('RGB', (2, height), (255, 255, 0)), (width, 0))  # Yellow separator
        combined.paste(img2, (width + 2, 0))
        return combined

    else:
        raise ValueError(f"Unknown overlay method: {method}")


if __name__ == "__main__":
    @dataclass
    class OverlayArguments:
        """Custom script arguments for overlay image configuration."""
        overlay_method: str = "side_by_side"  # one of: side_by_side, difference, blend, checkerboard, color_channels, edge_overlay, temporal_fade
        split: str = "train"  # one of: train, val
        include_successes: bool = True
        include_failures: bool = True
        train_sample_interval: int = 5
        compare_interval: str = "16"  # can be a single int (e.g., "16") or comma-separated list (e.g., "4,8,12,16")
        train_val_split_index: int = 5  # index to split job directories into train/val sets
        base_dataset_path: str = ''#/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01'  # base path to dataset directory
        max_exact_per_demo: int = 2  # maximum number of demo_id_exact per demo_id to keep in the dataset
        binary_or_exact_gt: str = "exact"
        debug_samples: int = -1
        task_description: str = ""
        just_prepare_data: bool = False

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, OverlayArguments))
    script_args, training_args, model_args, overlay_args = parser.parse_args_and_config()
    training_args.max_length = None
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # training_args.gradient_checkpointing = False 

    # Configuration for overlay method
    ALLOWED_OVERLAY_METHODS = {
        'side_by_side',
        # 'difference',
        # 'blend',
        # 'checkerboard',
        # 'color_channels',
        # 'edge_overlay',
        # 'temporal_fade',
    }
    OVERLAY_METHOD = overlay_args.overlay_method
    if OVERLAY_METHOD not in ALLOWED_OVERLAY_METHODS:
        raise ValueError(
            f"Invalid --overlay_method '{OVERLAY_METHOD}'. Choose one of: {sorted(ALLOWED_OVERLAY_METHODS)}"
        )

    logger.info("="*80)
    logger.info("Starting VLM Overlay Regression Training Script - OPTIMIZED")
    logger.info(f"Using overlay method: {OVERLAY_METHOD}")
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
    logger.info(f"  - overlay method: {OVERLAY_METHOD}")
    logger.info(f"  - split: {overlay_args.split}")
    logger.info(f"  - success and failures: {overlay_args.include_successes} and {overlay_args.include_failures}")
    logger.info(f"  - debug_samples: {overlay_args.debug_samples}")
    logger.info(f"  - train_sample_interval: {overlay_args.train_sample_interval}")
    compare_intervals = [int(x.strip()) for x in overlay_args.compare_interval.split(',')] if ',' in overlay_args.compare_interval else [int(overlay_args.compare_interval)]
    logger.info(f"  - compare_interval: {compare_intervals}")
    logger.info(f"  - train_val_split_index: {overlay_args.train_val_split_index}")
    logger.info(f"  - base_dataset_path: {overlay_args.base_dataset_path}")
    logger.info(f"  - task_description: {overlay_args.task_description}")


    logger.info("\nPreparing datasets...")

    # Get distributed training info (for multi-GPU support)
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            logger.info(f"Distributed training: rank {local_rank} of {world_size}")
        else:
            local_rank = 0
            world_size = 1
    except (ImportError, RuntimeError):
        local_rank = 0
        world_size = 1

    ################################
    # Dataset of all successes AND failures
    # IMPORTANT: Only rank 0 processes data to avoid redundant work and race conditions
    ################################
    TASK_DESC_TO_SYSTEM_PROMPT = {
        "PnPCounterToCab": "Pick the object from the counter and place it in the cabinet",
        "PnPCabToCounter": "Pick the object from the cabinet and place it on the counter",
        "PnPCounterToMicrowave": "Pick the object from the plate on the counter and place it in the microwave",
        "PnPMicrowaveToCounter": "Pick the object from the microwave and place it on the plate on the counter",
        "PnPStoveToCounter": "Pick the object from the stove and place it on the plate on the counter",  
        "PnPCounterToStove": "Pick the object from the plate on the counter and place it on the stove",  
        "PnPCounterToSink": "Pick the object from the plate on the counter and place it in the sink",  
        "PnPSinkToCounter": "Pick the object from the sink and place it on the plate on the counter",
        "PnPCoffeeServeMug": "Pick the mug from under the coffee machine dispenser and place it on the counter",
        "PnPCloseDrawer": "Close the drawer",
    }
    SYSTEM_PROMPT = f"""You are an expert roboticist tasked to compare a side-by-side of 2 images from a robot demonstration and determine which side shows more progress toward completing the task.
    The robot task is: INSERT_TASK_DESC_HERE.
    You will be given a side-by-side of 2 images from the same demonstration, and you need to identify how much closer or behind in task completion is the right image compared to the left."""

    problem = f"""Look at these two side-by-side images of a robot performing the task. \

    Left side image: Shows the robot at one point during the task. \
    Right side image: Shows the robot at another point during the task. \

    Task: Compare the two images and determine the relative progress difference. \
    - If the right image shows more progress toward task completion, respond with a positive number of how much farther (1 to 100) \
    - If the right image shows less progress toward task completion, respond with a negative number (-1 to -100) \

    The number should represent how much more or less progress the right image shows compared to the left."""

    # Load failure dataset from all job directories

    # Find all sep29_job* directories
    import glob
    import os
    all_dirs = sorted(glob.glob(f'{overlay_args.base_dataset_path}/*'))
    # Filter to only include directories with eval_log.json
    job_dirs = [d for d in all_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'eval_log.json'))]
    split = overlay_args.split

    if split == 'val':
        job_dirs = job_dirs[-overlay_args.train_val_split_index:]
    elif split=='train':
        job_dirs = job_dirs[:-overlay_args.train_val_split_index]
    else:
        job_dirs = job_dirs[:int(split)]
    logger.info(f"Found {len(job_dirs)} job directories.")
    print(f"Loading pre-extracted images from {len(job_dirs)} for {split} split")
    
    debug_suffix = '_debug' if overlay_args.debug_samples > -1 else ''
    binary_or_exact = '_binary' if overlay_args.binary_or_exact_gt == 'binary' else '_exact'

    # Support multiple dataset paths (comma-separated)
    dataset_paths = [p.strip() for p in overlay_args.base_dataset_path.split(',')]
    is_multi_path = len(dataset_paths) > 1
    logger.info(f"Loading from {len(dataset_paths)} dataset path(s): {dataset_paths}")

    # For single path, use the original overlay_dir; for multi-path, we'll use a combined output
    if "nov21_generate_dp_data_all_demos" in dataset_paths[0]:
        overlay_dir = Path(f'{dataset_paths[0]}/overlay_images{debug_suffix}')
    else:
        overlay_dir = Path(f'{dataset_paths[0]}/overlay_images{debug_suffix}{binary_or_exact}')
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_images_cache_file = overlay_dir / f'overlay_images_cache.pkl'
    # Create a string representation of compare_intervals for cache filename
    intervals_str = '_'.join(map(str, compare_intervals))
    dataset_cache_file = overlay_dir / f'dataset_cache_{split}_{overlay_args.train_sample_interval}__{intervals_str}__{overlay_args.include_successes}_{overlay_args.include_failures}.pkl'
    stats_cache_file = overlay_dir / f'dataset_stats_{split}_{overlay_args.train_sample_interval}__{intervals_str}__{overlay_args.include_successes}_{overlay_args.include_failures}.json'
    # Only rank 0 processes data; other ranks will wait and load the result
    # if local_rank == 0:
    if True:
        logger.info(f"Rank {local_rank}: Processing dataset...")
       
        # Load or create persistent overlay cache
        overlay_images_cache = {}
        if overlay_images_cache_file.exists():
            try:
                with open(overlay_images_cache_file, 'rb') as f:
                    overlay_images_cache = pickle.load(f)
                logger.info(f"Loaded {len(overlay_images_cache)} cached overlays")
            except Exception as e:
                logger.warning(f"Failed to load overlay cache: {e}")
                overlay_images_cache = {}

        combined_data = []
        total_successful_trajectories = 0
        job_dir_trajectory_counts = {}

        # Try to load cached datasets from all paths
        all_cached = True
        for dataset_path in dataset_paths:
            if "nov21_generate_dp_data_all_demos" in dataset_path:
                path_overlay_dir = Path(f'{dataset_path}/overlay_images{debug_suffix}')
            else:
                path_overlay_dir = Path(f'{dataset_path}/overlay_images{debug_suffix}{binary_or_exact}')
            path_cache_file = path_overlay_dir / f'dataset_cache_{split}_{overlay_args.train_sample_interval}__{intervals_str}__{overlay_args.include_successes}_{overlay_args.include_failures}.pkl'

            if path_cache_file.exists():
                logger.info(f"Loading pre-computed dataset from {path_cache_file}")
                try:
                    with open(path_cache_file, 'rb') as f:
                        path_data = pickle.load(f)
                    logger.info(f"  Loaded {len(path_data)} pairs from {dataset_path}")
                    combined_data.extend(path_data)

                    # Try to load trajectory counts from stats cache if available
                    stats_path_cache = path_overlay_dir / 'combined_data_analysis' / 'combined_data_stats.json'
                    if stats_path_cache.exists():
                        try:
                            with open(stats_path_cache, 'r') as f:
                                cached_stats = json.load(f)
                                if 'trajectory_statistics' in cached_stats:
                                    total_successful_trajectories += cached_stats['trajectory_statistics'].get('total_successful_trajectories', 0)
                                    path_job_counts = cached_stats['trajectory_statistics'].get('job_dir_breakdown', {})
                                    job_dir_trajectory_counts.update(path_job_counts)
                        except Exception as e:
                            logger.warning(f"Failed to load trajectory statistics from {stats_path_cache}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load dataset cache from {path_cache_file}: {e}")
                    all_cached = False
            else:
                logger.warning(f"No cached dataset found at {path_cache_file}")
                all_cached = False

        if all_cached and len(combined_data) > 0:
            logger.info(f"Loaded {len(combined_data)} total pairs from {len(dataset_paths)} cached dataset(s) - skipping processing!")
        elif is_multi_path and not all_cached:
            raise ValueError(f"Multi-path mode requires all paths to have pre-cached datasets. Missing cache for one or more paths.")
        else:
            # Single path mode without cache - will process below
            combined_data = []            

        # Only process if we don't have cached data
        if len(combined_data) == 0:
            def process_demo_pairs(one_demo, dataset_path):
                """Process all pairs for a single failure demo."""
                local_data = []
                job_name = Path(dataset_path).name
                demo_dir = one_demo['video_path'][:-4]
                demo_id = demo_dir.split('/')[-1].split('_')[0]
                demo_id_exact = demo_dir.split('/')[-1]
                num_frames=200
                #1 frame is rendered every 2 environment/action steps
                for interval in compare_intervals:
                    if num_frames <= interval:
                        continue
                    if one_demo['sf']=='success':
                        # Generate ALL possible pairs for this interval
                        max_idx1 = one_demo['trajectory_index'] - interval - 1
                        for idx1 in range(0, max_idx1+1, overlay_args.train_sample_interval):
                            idx2 = idx1 + interval
                            # idx2 = min(frame_numbers, key=lambda x: abs(x-start_random_actions))

                            # Load the images
                            image1_path = demo_dir + f'/frame_{idx1:06d}.png'
                            image2_path = demo_dir + f'/frame_{idx2:06d}.png'

                            # Determine which image is closer to completion (higher index = more progress)
                            if overlay_args.binary_or_exact_gt == "binary":
                                correct_answer = 32
                            elif overlay_args.binary_or_exact_gt == "exact":
                                correct_answer = idx2 - idx1
                                correct_answer = int(correct_answer/50 * 100)
                                raise Exception("Exact not supported. calcaulation of previous line is incorrect")
                            else:
                                raise Exception("Invalid binary_or_exact_gt value")
                            # Randomly swap image order 50% of the time to avoid position bias
                            swap = random.random() < 0.5
                            if swap:
                                image1_path, image2_path = image2_path, image1_path
                                idx1, idx2 = idx2, idx1
                                correct_answer = -correct_answer
                            # Create cache key for this overlay
                            cache_key = f"success_{correct_answer}_{'/'.join(demo_dir.split('/')[11:]).replace('/','_')}_{f'frame_{idx1:06d}'}_{f'frame_{idx2:06d}'}"
                            overlay_filename = cache_key+".png"
                            overlay_path = overlay_dir / overlay_filename

                            # Check if overlay exists in cache
                            if cache_key in overlay_images_cache:
                                # Use cached overlay path
                                pass
                            else:
                                # Create new overlay
                                try:
                                    overlay_img = create_overlay_image(image1_path, image2_path, method=OVERLAY_METHOD)
                                    overlay_img.save(overlay_path)
                                    overlay_images_cache[cache_key] = str(overlay_path)
                                except Exception as e:
                                    logger.warning(f"SUCCESS CREATION: failed to create overlay for demo {demo_id} path name {overlay_filename} frames {idx1}-{idx2}: {e}")
                                    continue
                            # Store minimal data structure (optimization: defer message dict creation)
                            local_data.append((
                                str(overlay_path),
                                correct_answer,
                                image1_path,
                                image2_path,
                                demo_id,
                                demo_id_exact,
                                'success',
                                job_name,
                            ))
                interval = 4
                if one_demo['sf']=='fail':
                    # Generate ALL possible pairs for this interval
                    max_idx1 = one_demo['trajectory_index'] - 1 #- interval
                    beginning_of_failure = None
                    for idx1 in range(0, max_idx1+1, overlay_args.train_sample_interval):
                        success_dir = one_demo['success_video_path'][:-4]
                        # Load the images
                        image1_path = demo_dir + f'/frame_{idx1:06d}.png'
                        image2_path = success_dir + f'/frame_{idx1:06d}.png'

                        # Compare the two images to check if they're significantly different
                        try:
                            arr1 = np.array(Image.open(image1_path).convert('RGB'))
                            arr2 = np.array(Image.open(image2_path).convert('RGB'))
                            diff = np.abs(arr1.astype(float) - arr2.astype(float))
                            mean_diff = np.mean(diff)
                            # Check if difference is significant
                            # Check if difference is significant
                            should_skip = False
                            if len(success_mean_diffs_at_idx) > 0:
                                # Determine key type for demo_id
                                first_demo_key = list(success_mean_diffs_at_idx.keys())[0]
                                current_demo_id = type(first_demo_key)(one_demo['demo_id'])
                                
                                if current_demo_id in success_mean_diffs_at_idx:
                                    demo_stats = success_mean_diffs_at_idx[current_demo_id]
                                    if len(demo_stats) > 0:
                                        # Determine key type for idx
                                        first_idx_key = list(demo_stats.keys())[0]
                                        idx1_key = type(first_idx_key)(idx1)
                                        
                                        if idx1_key in demo_stats:
                                            stats = demo_stats[idx1_key]
                                            if mean_diff <= np.mean(stats) + np.std(stats):
                                                should_skip = True
                            
                            if should_skip:
                                continue
                            
                            idx1 = int(idx1)
                            if beginning_of_failure is None:
                                beginning_of_failure = idx1
                            if idx1 <= beginning_of_failure + 8:
                                #only consider pairs after 8 frames after the failure starts
                                continue
                        except Exception as e:
                            logger.warning(f"Failed to compare images {image1_path} vs {image2_path}: {e}")
                            # pdb.set_trace()
                            continue

                        # Determine which image is closer to completion (higher index = more progress)
                        if overlay_args.binary_or_exact_gt == "binary":
                            correct_answer = 32
                        elif overlay_args.binary_or_exact_gt == "exact":
                            correct_answer = idx1 - beginning_of_failure + 4
                            correct_answer = int(correct_answer/50 * 100)
                        else:
                            raise Exception("Invalid binary_or_exact_gt value")

                        # Randomly swap image order 50% of the time to avoid position bias
                        swap = random.random() < 0.5
                        if swap:
                            image1_path, image2_path = image2_path, image1_path
                            correct_answer = -correct_answer

                        # Create cache key for this overlay
                        cache_key = f"svsf_{correct_answer}_{'/'.join(image1_path.split('/')[11:-1]).replace('/','_')}_{'/'.join(image2_path.split('/')[11:-1]).replace('/','_')}_frame_{idx1:06d}"
                        overlay_filename = cache_key + ".png"
                        overlay_path = overlay_dir / overlay_filename

                        # Check if overlay exists in cache
                        if False and cache_key in overlay_images_cache:
                            # Use cached overlay path
                            pass
                        else:
                            # Create new overlay
                            try:
                                overlay_img = create_overlay_image(image1_path, image2_path, method=OVERLAY_METHOD)
                                overlay_img.save(overlay_path)
                                overlay_images_cache[cache_key] = str(overlay_path)
                            except Exception as e:
                                logger.warning(f"S VS F CREATION: failed to create overlay for demo {demo_id} path name {overlay_filename} frames {idx1}: {e}")
                                continue
                        # Store minimal data structure (optimization: defer message dict creation)
                        local_data.append((
                            str(overlay_path),
                            correct_answer,
                            image1_path,
                            image2_path,
                            demo_id,
                            demo_id_exact,
                            'failure',
                            job_name,
                        ))
                return local_data

            # Track successful trajectories across all job directories
            total_successful_trajectories = 0
            job_dir_trajectory_counts = {}
            success_data = []
            unfiltered_failure_data = []

            # Process all job directories
            for job_dir in job_dirs:
                job_name = Path(job_dir).name
                # logger.info(f"\nProcessing job directory: {job_name}")

                # Load metadata for this job
                metadata_path = Path(job_dir) / 'eval_log.json'
                if not metadata_path.exists():
                    logger.warning(f"Metadata file not found for {job_name}, skipping...")
                    continue

                all_metadata = json.load(open(metadata_path, 'r'))
                for key,value in all_metadata.items():
                    if key.startswith("train/sim_reward_trajectory_"):
                        trajectory = ast.literal_eval(value)
                        video_path = '/workspace/guided_diffusion_policy/'+all_metadata[key.replace('train/sim_reward_trajectory_','train/sim_video_')]
                        demo_id = int(key.split('train/sim_reward_trajectory_')[-1].split('_')[0])
                        if 1 in trajectory:
                            success_data.append({
                                'video_path': video_path,
                                'trajectory_index': int(trajectory.index(1)/2),
                                'sf': 'success',
                                'demo_id': demo_id
                            })
                        else:
                            unfiltered_failure_data.append({
                                'video_path': video_path,
                                'trajectory_index': int(len(trajectory)/2),
                                'sf': 'fail', 
                                'demo_id': demo_id
                            })

            failure_data = []
            for idx, fd in enumerate(unfiltered_failure_data):
                demo_id = fd['demo_id']
                for sd in success_data:
                    if sd['demo_id'] == demo_id:
                        failure_data.append(fd)
                        failure_data[-1]['success_video_path'] = sd['video_path']
                        failure_data[-1]['success_trajectory_index'] = sd['trajectory_index']

            # BALANCED SAMPLING BEGINS. to ensure equal representation of all demo_ids
            
            # Group success_data and failure_data by demo_id
            success_by_demo = defaultdict(list)
            failure_by_demo = defaultdict(list)
            for sd in success_data:
                success_by_demo[sd['demo_id']].append(sd)
            for fd in failure_data:
                failure_by_demo[fd['demo_id']].append(fd)

            # Find all unique demo_ids present in both success and failure data
            common_demo_ids = set(success_by_demo.keys()) & set(failure_by_demo.keys())
            logger.info(f"Found {len(common_demo_ids)} demo_ids present in both success and failure data")
            if len(common_demo_ids) == 0:
                logger.warning("No common demo_ids found between success and failure data!")
                raise Exception('No common demo_ids found between success and failure data!')
            else:
                # Calculate the minimum count per demo_id to ensure equal representation
                # For each demo_id, find the minimum between success and failure counts
                min_per_demo = {}
                for demo_id in common_demo_ids:
                    min_count = min(len(success_by_demo[demo_id]), len(failure_by_demo[demo_id]))
                    min_per_demo[demo_id] = min_count
                logger.info(f"Samples per demo_id after balancing: {dict(sorted(min_per_demo.items()))}")
                # Sample equal number from each demo_id
                balanced_success = []
                balanced_failure = []
                for demo_id in sorted(common_demo_ids):
                    count = min(min_per_demo[demo_id],overlay_args.max_exact_per_demo)
                    # Randomly sample 'count' items from each demo_id (use seed for reproducibility)
                    rng = random.Random(42 + demo_id)  # Different seed per demo_id but deterministic
                    success_samples = rng.sample(success_by_demo[demo_id], count)
                    failure_samples = rng.sample(failure_by_demo[demo_id], count)
                    balanced_success.extend(success_samples)
                    balanced_failure.extend(failure_samples)
                success_data = balanced_success
                failure_data = balanced_failure

                logger.info(f"After balanced sampling:")
                logger.info(f"  Total success samples: {len(success_data)}")
                logger.info(f"  Total failure samples: {len(failure_data)}")
                logger.info(f"  Samples per demo_id: {len(success_data) // len(common_demo_ids) if common_demo_ids else 0}")

            # BALANCED SAMPLING ENDS.
            print('number of success by demo', [len(success_by_demo[k]) for k in common_demo_ids] )
            print('number of failures by demo', [len(failure_by_demo[k]) for k in common_demo_ids] )
            job_dir_trajectory_counts[job_name] = len(success_data) + len(failure_data)
            total_successful_trajectories += len(success_data)

            logger.info(f"Found {len(all_metadata)} demos in {job_name}")
            logger.info(f"Found Total: {len(success_data)} successful and {len(failure_data)} failure trajectories in {job_name}")
            if stats_cache_file.exists():
                with open(stats_cache_file, 'r') as f:
                    cached_stats = json.load(f)
                    success_mean_diffs_at_idx = cached_stats['success_mean_diffs_at_idx']
                    sf_mean_diffs_at_idx = cached_stats['sf_mean_diffs_at_idx']
            else:
                success_mean_diffs_at_idx = {}
                sf_mean_diffs_at_idx = {}
                
                # Parallelize stats generation
                all_demo_ids = sorted(list(success_by_demo.keys()))
                my_demo_ids = all_demo_ids[local_rank::world_size]
                
                for sk in tqdm(my_demo_ids, desc=f"Rank {local_rank} Stats"):
                    # success_mean_diffs_at_idx[sk]={}
                    # Use a local dict to avoid key errors if we were to update the main one directly (though here it's fine)
                    # But we need to reconstruct the logic carefully.
                    # The original code iterated over success_by_demo.items().
                    # We iterate over keys.
                    sv = success_by_demo[sk] # Get the value
                    
                    success_mean_diffs_at_idx[sk]={}
                    for x in range(1,min(10,len(sv)-1)):
                        one_sd = sv[0]
                        two_sd = sv[x]
                        for idx in range(min(one_sd['trajectory_index'],two_sd['trajectory_index'])):
                            image1_path = one_sd['video_path'][:-4] + f'/frame_{idx:06d}.png'
                            image2_path = two_sd['video_path'][:-4] + f'/frame_{idx:06d}.png'

                            img1 = Image.open(image1_path).convert('RGB')
                            img2 = Image.open(image2_path).convert('RGB')

                            # Convert to numpy arrays
                            arr1 = np.array(img1)
                            arr2 = np.array(img2)

                            # Calculate difference metrics
                            diff = np.abs(arr1.astype(float) - arr2.astype(float))
                            mean_diff = np.mean(diff)
                            if idx not in success_mean_diffs_at_idx[sk]:
                                success_mean_diffs_at_idx[sk][idx]=[]
                            success_mean_diffs_at_idx[sk][idx].append(mean_diff)
                            
                # sf_mean_diffs_at_idx logic
                for sk in tqdm(my_demo_ids, desc=f"Rank {local_rank} SF Stats"):
                    if sk not in failure_by_demo:
                        continue # changed from pass to continue for clarity
                    
                    sf_mean_diffs_at_idx[sk]={}
                    for x in range(0,min(10,len(failure_by_demo[sk])-1)):
                        one_sd = success_by_demo[sk][0]
                        two_sd = failure_by_demo[sk][x]
                        for idx in range(min(one_sd['trajectory_index'],two_sd['trajectory_index'])):
                            image1_path = one_sd['video_path'][:-4] + f'/frame_{idx:06d}.png'
                            image2_path = two_sd['video_path'][:-4] + f'/frame_{idx:06d}.png'

                            img1 = Image.open(image1_path).convert('RGB')
                            img2 = Image.open(image2_path).convert('RGB')

                            # Convert to numpy arrays
                            arr1 = np.array(img1)
                            arr2 = np.array(img2)

                            # Calculate difference metrics
                            diff = np.abs(arr1.astype(float) - arr2.astype(float))
                            mean_diff = np.mean(diff)
                            if idx not in sf_mean_diffs_at_idx[sk]:
                                sf_mean_diffs_at_idx[sk][idx]=[]
                            sf_mean_diffs_at_idx[sk][idx].append(mean_diff)

                # Gather stats from all ranks
                if world_size > 1:
                    logger.info(f"Rank {local_rank}: Gathering stats from all ranks...")
                    all_success_stats = [None for _ in range(world_size)]
                    all_sf_stats = [None for _ in range(world_size)]
                    dist.all_gather_object(all_success_stats, success_mean_diffs_at_idx)
                    dist.all_gather_object(all_sf_stats, sf_mean_diffs_at_idx)
                    
                    # Merge
                    success_mean_diffs_at_idx = {}
                    sf_mean_diffs_at_idx = {}
                    for rank_stats in all_success_stats:
                        success_mean_diffs_at_idx.update(rank_stats)
                    for rank_stats in all_sf_stats:
                        sf_mean_diffs_at_idx.update(rank_stats)
                    logger.info(f"Rank {local_rank}: Gathered stats")

                if local_rank == 0:
                    with open(stats_cache_file, 'w') as f:
                        json.dump({"sf_mean_diffs_at_idx": sf_mean_diffs_at_idx, "success_mean_diffs_at_idx": success_mean_diffs_at_idx}, f)
            
            # Process failure demos in parallel for this job
            if overlay_args.debug_samples > -1:
                success_samples = random.sample(success_data, min(len(success_data), overlay_args.debug_samples))
                failure_samples = random.sample(failure_data, min(len(failure_data), overlay_args.debug_samples))
            else:
                success_samples = success_data
                failure_samples = failure_data
            
            # Split samples across ranks
            success_samples = success_samples[local_rank::world_size]
            failure_samples = failure_samples[local_rank::world_size]
            logger.info(f"Rank {local_rank}: Found {len(success_samples)} successful and {len(failure_samples)} failure trajectories in {job_name}")

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_demo_pairs, one_demo, job_dir): one_demo
                            for one_demo in failure_samples + success_samples}

                for future in tqdm(as_completed(futures), total=len(futures),
                                    desc=f"Generating {job_name} overlay pairs", unit="demo"):
                    try:
                        demo_data = future.result()
                        combined_data.extend(demo_data)
                    except Exception as e:
                        demo_name = futures[future]
                        logger.warning(f"Failed to process demo {demo_name} in {job_name}: {e}")

            logger.info(f"Rank {local_rank}: Generated {len(combined_data)} pairs locally")
            
            # Gather data from all ranks
            if world_size > 1:
                logger.info(f"Rank {local_rank}: Gathering data from all ranks...")
                all_data = [None for _ in range(world_size)]
                dist.all_gather_object(all_data, combined_data)
                
                # Flatten the list of lists
                combined_data = []
                for rank_data in all_data:
                    combined_data.extend(rank_data)
                logger.info(f"Rank {local_rank}: Gathered {len(combined_data)} total pairs from all ranks")

            logger.info(f"Total pairs so far: {len(combined_data)}")

            logger.info(f"\nGenerated {len(combined_data)} total overlayed comparisons from {len(job_dirs)} job directories")
            logger.info(f"Total successful trajectories used for training: {total_successful_trajectories}")
            # logger.info(f"Breakdown by job directory:")
            # for job_name, count in job_dir_trajectory_counts.items():
            #     logger.info(f"  {job_name}: {count} successful trajectories")
            # Convert tuples to final message format (optimization: done once after all processing)
            logger.info("Converting data to final message format...")
            final_combined_data = []
            for item in combined_data:
                # item format: (overlay_path, correct_answer, orig_img1, orig_img2, demo_id, demo_id_exact, demo_type, job_name)
                job_name = item[7] if len(item) > 7 else ""

                # Find matching task description based on job_name
                task_desc = None
                for task_key, task_description in TASK_DESC_TO_SYSTEM_PROMPT.items():
                    if task_key in job_name:
                        task_desc = task_description
                        break

                # Create system prompt with task-specific description
                if task_desc:
                    sample_system_prompt = SYSTEM_PROMPT.replace("INSERT_TASK_DESC_HERE", task_desc)
                else:
                    sample_system_prompt = SYSTEM_PROMPT

                messages = [
                    {"role": "system", "content": sample_system_prompt},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": str(item[1])}
                ]
                final_combined_data.append({
                    "images": [item[0]],
                    "correct_answer": item[1],
                    "messages": messages,
                    "original_images": [item[2], item[3]],
                    "demo_id": item[4],
                    "demo_id_exact": item[5],
                    "demo_success": item[6],
                    "job_name": job_name
                })
            combined_data = final_combined_data
            logger.info(f"Converted {len(combined_data)} pairs to message format")

            # Save dataset cache (biggest speedup for subsequent runs)
            if local_rank == 0:
                with open(dataset_cache_file, 'wb') as f:
                    pickle.dump(combined_data, f)
                logger.info(f"Saved {len(combined_data)} pairs to dataset cache at {dataset_cache_file}")

                with open(overlay_images_cache_file, 'wb') as f:
                    pickle.dump(overlay_images_cache, f)
                logger.info(f"Saved {len(overlay_images_cache)} overlays to cache (including failures)")

    # Synchronize all processes: wait for rank 0 to finish processing
    if world_size > 1:
        try:
            logger.info(f"Rank {local_rank}: Synchronizing with other processes...")
            dist.barrier()
            logger.info(f"Rank {local_rank}: Synchronization complete")
        except Exception as e:
            logger.warning(f"Failed to synchronize processes: {e}")

    # Non-rank-0 processes load the cached dataset created by rank 0
    # if local_rank != 0:
    #     logger.info(f"Rank {local_rank}: Loading dataset created by rank 0...")
    #     if dataset_cache_file.exists():
    #         try:
    #             with open(dataset_cache_file, 'rb') as f:
    #                 combined_data = pickle.load(f)
    #             logger.info(f"Rank {local_rank}: Loaded {len(combined_data)} pairs from dataset cache")
    #         except Exception as e:
    #             logger.error(f"Rank {local_rank}: Failed to load dataset cache: {e}")
    #             raise
    #     else:
    #         logger.error(f"Rank {local_rank}: Dataset cache not found at {dataset_cache_file}")
    #         raise FileNotFoundError(f"Rank 0 should have created {dataset_cache_file}")
    # ============================================================================================
    # VISUALIZATION: Show examples and statistics of combined_data (BEFORE SHARDING)
    # ============================================================================================
    # Only rank 0 needs to generate visualizations to save memory
    # IMPORTANT: Do this BEFORE sharding to get accurate full dataset statistics
    if local_rank == 0 and len(combined_data) > 0:
        logger.info("\n" + "="*80)
        logger.info("DATASET VISUALIZATION AND STATISTICS")
        logger.info("="*80)

        # Create visualization directories in both locations
        # 1. Training output directory
        viz_dir = Path(training_args.output_dir) / 'combined_data_analysis'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 2. Dataset cache directory (where overlay images are stored)
        viz_dir_cache = overlay_dir / 'combined_data_analysis'
        viz_dir_cache.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving visualizations to:")
        logger.info(f"  - Training output: {viz_dir}")
        logger.info(f"  - Dataset cache: {viz_dir_cache}")

        # 1. Basic Statistics
        logger.info(f"\n📊 Dataset Statistics:")
        logger.info(f"  Total number of samples: {len(combined_data)}")

        # Extract answer distribution
        answers = [item['messages'][2]['content'] for item in combined_data]
        answers_numeric = [int(a) for a in answers]

        logger.info(f"\n📈 Answer Distribution:")
        logger.info(f"  Mean answer: {np.mean(answers_numeric):.2f}")
        logger.info(f"  Median answer: {np.median(answers_numeric):.2f}")
        logger.info(f"  Std dev: {np.std(answers_numeric):.2f}")
        logger.info(f"  Min answer: {min(answers_numeric)}")
        logger.info(f"  Max answer: {max(answers_numeric)}")

        # Count positive vs negative answers
        positive_count = sum(1 for a in answers_numeric if a > 0)
        negative_count = sum(1 for a in answers_numeric if a < 0)
        zero_count = sum(1 for a in answers_numeric if a == 0)

        logger.info(f"\n📊 Answer Polarity:")
        logger.info(f"  Positive answers (right image shows more progress): {positive_count} ({positive_count/len(answers_numeric)*100:.1f}%)")
        logger.info(f"  Negative answers (right image shows less progress): {negative_count} ({negative_count/len(answers_numeric)*100:.1f}%)")
        logger.info(f"  Zero answers (same progress): {zero_count} ({zero_count/len(answers_numeric)*100:.1f}%)")

        # Demo type distribution
        demo_types = [item.get('demo_success', 'unknown') for item in combined_data]
        from collections import Counter, defaultdict
        demo_type_counts = Counter(demo_types)
        logger.info(f"\n📁 Demo Type Distribution:")
        for demo_type, count in demo_type_counts.items():
            logger.info(f"  {demo_type}: {count} ({count/len(demo_types)*100:.1f}%)")

        # 2. Show some random examples
        logger.info(f"\n🖼️  Sample Examples (5 random samples):")
        logger.info("-"*80)

        import random
        random.seed(42)
        sample_indices = random.sample(range(len(combined_data)), min(5, len(combined_data)))

        for i, idx in enumerate(sample_indices, 1):
            item = combined_data[idx]
            demo_success = item.get('demo_success', 'unknown')
            success_label = 'SUCCESS' if demo_success == 'success' else 'FAILURE' if demo_success == 'failure' else 'UNKNOWN'
            logger.info(f"\nExample {i} (Index {idx}) - {success_label}:")
            logger.info(f"  Overlay Image: {item['images'][0]}")
            logger.info(f"  Original Image 1: {item['original_images'][0]}")
            logger.info(f"  Original Image 2: {item['original_images'][1]}")
            logger.info(f"  Demo Type: {item.get('demo_success', 'unknown')}")
            logger.info(f"  Demo Success: {demo_success}")
            logger.info(f"  Answer (progress difference): {item['messages'][2]['content']}")
            logger.info(f"  System Prompt: {item['messages'][0]['content'][:100]}...")
            logger.info(f"  User Question: {item['messages'][1]['content'][:100]}...")

        # 3. Create visualizations
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Create comprehensive statistics plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Dataset Analysis - {split.upper()} Split', fontsize=16, fontweight='bold')

            # Plot 1: Answer distribution histogram
            ax = axes[0, 0]
            ax.hist(answers_numeric, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (Equal Progress)')
            ax.set_xlabel('Progress Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of Progress Values\n(n={len(answers_numeric)})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Answer polarity pie chart
            ax = axes[0, 1]
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            labels = [f'Positive\n({positive_count})', f'Negative\n({negative_count})', f'Zero\n({zero_count})']
            sizes = [positive_count, negative_count, zero_count]
            # Filter out zero sizes
            sizes_filtered = [s for s in sizes if s > 0]
            labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
            colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]
            if sizes_filtered:
                ax.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                      autopct='%1.1f%%', startangle=90)
            ax.set_title('Answer Polarity Distribution')

            # Plot 3: Progress value box plot
            ax = axes[0, 2]
            ax.boxplot(answers_numeric, vert=True)
            ax.set_ylabel('Progress Value')
            ax.set_title('Progress Value Distribution (Box Plot)')
            ax.grid(True, alpha=0.3)

            # Plot 4: Positive vs Negative distribution bar chart
            ax = axes[1, 0]
            ax.bar(['Negative\n(L > R)', 'Zero\n(L = R)', 'Positive\n(R > L)'],
                  [negative_count, zero_count, positive_count],
                  color=['red', 'gray', 'green'], alpha=0.7, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title('Progress Direction Distribution')
            ax.grid(True, alpha=0.3, axis='y')

            # Plot 5: Demo type breakdown
            ax = axes[1, 1]
            demo_type_names = list(demo_type_counts.keys())
            demo_type_values = list(demo_type_counts.values())
            ax.bar(demo_type_names, demo_type_values, alpha=0.7, edgecolor='black', color='purple')
            ax.set_xlabel('Demo Type')
            ax.set_ylabel('Count')
            ax.set_title('Pairs per Demo Type')
            ax.grid(True, alpha=0.3, axis='y')

            # Hide the last subplot
            axes[1, 2].axis('off')

            plt.tight_layout()
            # Save to both locations
            stats_plot_path = viz_dir / 'combined_data_statistics.png'
            stats_plot_path_cache = viz_dir_cache / 'combined_data_statistics.png'
            plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
            plt.savefig(stats_plot_path_cache, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"\n📊 Saved statistics plot to:")
            logger.info(f"  - {stats_plot_path}")
            logger.info(f"  - {stats_plot_path_cache}")

            # Plot example images grid
            logger.info(f"\n🖼️  Creating visual grids of example images (Success & Failure)...")
            
            # Split data
            success_items_viz = [item for item in combined_data if item.get('demo_success') == 'success']
            failure_items_viz = [item for item in combined_data if item.get('demo_success') == 'failure']

            for viz_name, items_list in [('success', success_items_viz), ('failure', failure_items_viz)]:
                num_random = min(12, len(items_list))  # Show up to 12 random examples
                
                if num_random > 0:
                    random_indices = random.sample(range(len(items_list)), num_random)
                    rows = (num_random + 2) // 3  # 3 columns
                    cols = min(3, num_random)

                    fig, axes_grid = plt.subplots(rows, cols, figsize=(18, 6 * rows))
                    fig.suptitle(f'Random {viz_name.capitalize()} Examples', fontsize=16, fontweight='bold')

                    if rows == 1 and cols == 1:
                        axes_grid = [[axes_grid]]
                    elif rows == 1:
                        axes_grid = [axes_grid]
                    elif cols == 1:
                        axes_grid = [[ax] for ax in axes_grid]

                    for idx, data_idx in enumerate(random_indices):
                        row = idx // 3
                        col = idx % 3

                        if row >= len(axes_grid) or col >= len(axes_grid[row]):
                            continue

                        ax = axes_grid[row][col]
                        item = items_list[data_idx]
                        overlay_path = item['images'][0]
                        answer = item['messages'][2]['content']
                        demo_success = item.get('demo_success', 'unknown')

                        try:
                            img = Image.open(overlay_path)
                            ax.imshow(img)
                            ax.axis('off')

                            progress_val = int(answer)
                            color = 'green' if progress_val > 0 else 'red' if progress_val < 0 else 'gray'
                            # Add success/failure label to title
                            success_label = '✓ Success' if demo_success == 'success' else '✗ Failure' if demo_success == 'failure' else 'Unknown'
                            
                            # Add task description
                            job_name = item.get('job_name', '')
                            task_desc = ""
                            for task_key, task_description in TASK_DESC_TO_SYSTEM_PROMPT.items():
                                if task_key in job_name:
                                    task_desc = task_description
                                    break
                            
                            # Wrap task description for display
                            if task_desc:
                                task_desc_short = textwrap.shorten(task_desc, width=40, placeholder="...")
                                title_text = f'Progress: {answer} ({success_label})\n{task_desc_short}'
                            else:
                                title_text = f'Progress: {answer} ({success_label})'

                            ax.set_title(title_text,
                                       fontsize=8, fontweight='bold', color=color)
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error: {str(e)}',
                                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
                            ax.axis('off')

                    # Hide any unused subplots
                    for idx in range(num_random, rows * cols):
                        row = idx // 3
                        col = idx % 3
                        if row < len(axes_grid) and col < len(axes_grid[row]):
                            axes_grid[row][col].axis('off')

                    plt.tight_layout()
                    # Save to both locations
                    grid_path = viz_dir / f'{viz_name}_examples_grid.png'
                    grid_path_cache = viz_dir_cache / f'{viz_name}_examples_grid.png'
                    plt.savefig(grid_path, dpi=100, bbox_inches='tight')
                    plt.savefig(grid_path_cache, dpi=100, bbox_inches='tight')
                    plt.close()
                    logger.info(f"🖼️  Saved {viz_name} example images grid to:")
                    logger.info(f"  - {grid_path}")
                    logger.info(f"  - {grid_path_cache}")
                else:
                    logger.info(f"No {viz_name} examples found to visualize.")

        except ImportError:
            logger.warning("matplotlib not available, skipping visualizations")
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")

        # Analyze demo_id and demo_id_exact statistics
        logger.info(f"\n📋 Demo ID Analysis:")

        # Extract demo_id and demo_id_exact from original_images paths
        demo_id_to_exact = defaultdict(set)
        demo_id_to_pairs = defaultdict(int)
        demo_id_exact_to_pairs = defaultdict(int)

        # NEW: Track success vs failure breakdown per demo_id
        demo_id_to_success_pairs = defaultdict(int)
        demo_id_to_failure_pairs = defaultdict(int)
        demo_id_exact_to_demo_type = {}

        for item in combined_data:
            # Extract demo_id and demo_id_exact from the original image paths
            # Format: .../demo_id_exact/frame_XXXXXX.png
            orig_img1 = item['original_images'][0]
            # Get the demo_id_exact from path (second to last part)
            demo_id_exact = Path(orig_img1).parent.name
            # Get the demo_id (first part before underscore)
            demo_id = demo_id_exact.split('_')[0]

            # Get demo type (success/failure)
            demo_type = item.get('demo_success', 'unknown')

            demo_id_to_exact[demo_id].add(demo_id_exact)
            demo_id_to_pairs[demo_id] += 1
            demo_id_exact_to_pairs[demo_id_exact] += 1
            demo_id_exact_to_demo_type[demo_id_exact] = demo_type

            # Track success vs failure per demo_id
            if demo_type == 'success':
                demo_id_to_success_pairs[demo_id] += 1
            elif demo_type == 'failure':
                demo_id_to_failure_pairs[demo_id] += 1

        # Log statistics
        logger.info(f"  Total unique demo_id: {len(demo_id_to_exact)}")
        logger.info(f"  Total unique demo_id_exact: {sum(len(exacts) for exacts in demo_id_to_exact.values())}")

        logger.info(f"\n📊 Demo ID breakdown (top 20 by pair count):")
        sorted_demo_ids = sorted(demo_id_to_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        for demo_id, pair_count in sorted_demo_ids:
            num_exact = len(demo_id_to_exact[demo_id])
            success_count = demo_id_to_success_pairs[demo_id]
            failure_count = demo_id_to_failure_pairs[demo_id]
            logger.info(f"  demo_id '{demo_id}': {num_exact} unique demo_id_exact, {pair_count} image pairs (Success: {success_count}, Failure: {failure_count})")

        logger.info(f"\n📊 Demo ID Exact breakdown (top 20 by pair count):")
        sorted_demo_id_exact = sorted(demo_id_exact_to_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        for demo_id_exact, pair_count in sorted_demo_id_exact:
            logger.info(f"  demo_id_exact '{demo_id_exact}': {pair_count} image pairs")

        # Create visualization for demo_id statistics
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Create demo_id analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Demo ID Analysis - {split.upper()} Split', fontsize=16, fontweight='bold')

            # Plot 1: Number of demo_id_exact per demo_id (bar graph)
            ax = axes[0, 0]
            # Sort demo_ids by number of demo_id_exact (descending)
            demo_id_sorted = sorted(demo_id_to_exact.items(), key=lambda x: len(x[1]), reverse=True)
            demo_ids_plot1 = [item[0] for item in demo_id_sorted]
            num_exact_counts = [len(item[1]) for item in demo_id_sorted]

            # Create bar graph
            x_pos = range(len(demo_ids_plot1))
            ax.bar(x_pos, num_exact_counts, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('demo_id')
            ax.set_ylabel('Number of demo_id_exact')
            ax.set_title(f'Number of demo_id_exact per demo_id\n(Total demo_id: {len(demo_id_to_exact)})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(demo_ids_plot1, rotation=90, fontsize=6)
            ax.grid(True, alpha=0.3, axis='y')

            # Add mean line
            mean_val = np.mean(num_exact_counts)
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.legend()

            # Plot 2: Number of pairs per demo_id (top 20)
            ax = axes[0, 1]
            top_20_demo_ids = sorted_demo_ids[:20]
            demo_ids = [item[0] for item in top_20_demo_ids]
            pair_counts = [item[1] for item in top_20_demo_ids]
            ax.barh(range(len(demo_ids)), pair_counts, color='coral', edgecolor='black')
            ax.set_yticks(range(len(demo_ids)))
            ax.set_yticklabels(demo_ids, fontsize=8)
            ax.set_xlabel('Number of image pairs')
            ax.set_title('Top 20 demo_id by pair count')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            # Plot 3: Number of pairs per demo_id_exact (top 20)
            ax = axes[1, 0]
            top_20_demo_id_exact = sorted_demo_id_exact[:20]
            demo_id_exacts = [item[0] for item in top_20_demo_id_exact]
            pair_counts_exact = [item[1] for item in top_20_demo_id_exact]
            ax.barh(range(len(demo_id_exacts)), pair_counts_exact, color='mediumseagreen', edgecolor='black')
            ax.set_yticks(range(len(demo_id_exacts)))
            ax.set_yticklabels(demo_id_exacts, fontsize=7)
            ax.set_xlabel('Number of image pairs')
            ax.set_title('Top 20 demo_id_exact by pair count')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            # Plot 4: Scatter plot - demo_id_exact count vs pair count per demo_id
            ax = axes[1, 1]
            demo_ids_all = list(demo_id_to_exact.keys())
            x_vals = [len(demo_id_to_exact[did]) for did in demo_ids_all]
            y_vals = [demo_id_to_pairs[did] for did in demo_ids_all]
            ax.scatter(x_vals, y_vals, alpha=0.6, s=50, color='purple', edgecolor='black')
            ax.set_xlabel('Number of demo_id_exact per demo_id')
            ax.set_ylabel('Number of image pairs per demo_id')
            ax.set_title('Relationship: demo_id_exact count vs pair count')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            if len(x_vals) > 1:
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save to both locations
            demo_analysis_path = viz_dir / 'demo_id_analysis.png'
            demo_analysis_path_cache = viz_dir_cache / 'demo_id_analysis.png'
            plt.savefig(demo_analysis_path, dpi=150, bbox_inches='tight')
            plt.savefig(demo_analysis_path_cache, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"\n📊 Saved demo_id analysis plot to:")
            logger.info(f"  - {demo_analysis_path}")
            logger.info(f"  - {demo_analysis_path_cache}")

        except Exception as e:
            logger.warning(f"Failed to create demo_id analysis visualizations: {e}")

        # Create SUCCESS vs FAILURE comparison visualization
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            logger.info(f"\n📊 Creating Success vs Failure comparison visualization...")

            # Create figure with 3 subplots
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'Success vs Failure Data Comparison - {split.upper()} Split', fontsize=16, fontweight='bold')

            # Plot 1: Stacked bar chart - Success vs Failure per demo_id (top 20)
            ax = axes[0, 0]
            top_20_demo_ids = sorted_demo_ids[:20]
            demo_ids = [item[0] for item in top_20_demo_ids]
            success_counts = [demo_id_to_success_pairs[did] for did in demo_ids]
            failure_counts = [demo_id_to_failure_pairs[did] for did in demo_ids]

            x = range(len(demo_ids))
            ax.barh(x, success_counts, label='Success', color='#2ecc71', edgecolor='black')
            ax.barh(x, failure_counts, left=success_counts, label='Failure', color='#e74c3c', edgecolor='black')
            ax.set_yticks(x)
            ax.set_yticklabels(demo_ids, fontsize=8)
            ax.set_xlabel('Number of image pairs')
            ax.set_title('Success vs Failure Pairs per demo_id (Top 20)')
            ax.legend()
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            # Plot 2: Grouped bar chart - Success vs Failure comparison (top 15)
            ax = axes[0, 1]
            top_15_demo_ids = sorted_demo_ids[:15]
            demo_ids_15 = [item[0] for item in top_15_demo_ids]
            success_counts_15 = [demo_id_to_success_pairs[did] for did in demo_ids_15]
            failure_counts_15 = [demo_id_to_failure_pairs[did] for did in demo_ids_15]

            x = np.arange(len(demo_ids_15))
            width = 0.35
            ax.bar(x - width/2, success_counts_15, width, label='Success', color='#2ecc71', edgecolor='black')
            ax.bar(x + width/2, failure_counts_15, width, label='Failure', color='#e74c3c', edgecolor='black')
            ax.set_xlabel('demo_id')
            ax.set_ylabel('Number of image pairs')
            ax.set_title('Success vs Failure Comparison (Top 15)')
            ax.set_xticks(x)
            ax.set_xticklabels(demo_ids_15, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Plot 3: Scatter plot - Success count vs Failure count per demo_id
            ax = axes[1, 0]
            all_demo_ids = list(demo_id_to_exact.keys())
            success_vals = [demo_id_to_success_pairs[did] for did in all_demo_ids]
            failure_vals = [demo_id_to_failure_pairs[did] for did in all_demo_ids]

            ax.scatter(success_vals, failure_vals, alpha=0.6, s=80, color='purple', edgecolor='black')
            ax.set_xlabel('Number of Success pairs')
            ax.set_ylabel('Number of Failure pairs')
            ax.set_title('Success vs Failure Pairs Distribution')
            ax.grid(True, alpha=0.3)

            # Add diagonal reference line (equal success/failure)
            max_val = max(max(success_vals) if success_vals else 0, max(failure_vals) if failure_vals else 0)
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Equal success/failure')
            ax.legend()

            # Plot 4: Pie chart - Overall Success vs Failure distribution
            ax = axes[1, 1]
            total_success = sum(demo_id_to_success_pairs.values())
            total_failure = sum(demo_id_to_failure_pairs.values())

            sizes = [total_success, total_failure]
            labels = [f'Success\n({total_success} pairs)', f'Failure\n({total_failure} pairs)']
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)

            if sum(sizes) > 0:
                ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
            ax.set_title(f'Overall Success vs Failure Distribution\n(Total: {sum(sizes)} pairs)')

            plt.tight_layout()

            # Save to both locations
            success_failure_plot = viz_dir / 'success_vs_failure_comparison.png'
            success_failure_plot_cache = viz_dir_cache / 'success_vs_failure_comparison.png'
            plt.savefig(success_failure_plot, dpi=150, bbox_inches='tight')
            plt.savefig(success_failure_plot_cache, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"\n📊 Saved success vs failure comparison plot to:")
            logger.info(f"  - {success_failure_plot}")
            logger.info(f"  - {success_failure_plot_cache}")

            # Log summary statistics
            logger.info(f"\n📊 Success vs Failure Summary:")
            logger.info(f"  Total Success pairs: {total_success} ({total_success/(total_success+total_failure)*100:.1f}%)")
            logger.info(f"  Total Failure pairs: {total_failure} ({total_failure/(total_success+total_failure)*100:.1f}%)")
            logger.info(f"  demo_ids with only Success data: {sum(1 for did in all_demo_ids if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] == 0)}")
            logger.info(f"  demo_ids with only Failure data: {sum(1 for did in all_demo_ids if demo_id_to_failure_pairs[did] > 0 and demo_id_to_success_pairs[did] == 0)}")
            logger.info(f"  demo_ids with both Success and Failure data: {sum(1 for did in all_demo_ids if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] > 0)}")

        except Exception as e:
            logger.warning(f"Failed to create success vs failure comparison visualization: {e}")

        # Save detailed statistics to JSON
        try:
            success_count = sum(1 for dt in demo_types if dt == 'success')
            failure_count = sum(1 for dt in demo_types if dt == 'failure')
            unknown_count = sum(1 for dt in demo_types if dt not in ['success', 'failure'])

            # Prepare demo_id statistics for JSON
            demo_id_stats = {}
            for demo_id in sorted(demo_id_to_exact.keys()):
                demo_id_stats[demo_id] = {
                    'num_demo_id_exact': len(demo_id_to_exact[demo_id]),
                    'demo_id_exact_list': sorted(list(demo_id_to_exact[demo_id])),
                    'num_image_pairs': demo_id_to_pairs[demo_id],
                    'num_success_pairs': demo_id_to_success_pairs[demo_id],
                    'num_failure_pairs': demo_id_to_failure_pairs[demo_id],
                }

            # Prepare demo_id_exact statistics with demo_type info
            demo_id_exact_stats = {}
            for demo_id_exact, pair_count in sorted(demo_id_exact_to_pairs.items(),
                                                     key=lambda x: x[1], reverse=True):
                demo_id_exact_stats[demo_id_exact] = {
                    'num_pairs': pair_count,
                    'demo_type': demo_id_exact_to_demo_type.get(demo_id_exact, 'unknown')
                }

            stats_dict = {
                'total_pairs': len(combined_data),
                'demo_type_breakdown': {
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'unknown_count': unknown_count,
                    'success_percentage': float(success_count/len(demo_types)*100) if demo_types else 0.0,
                    'failure_percentage': float(failure_count/len(demo_types)*100) if demo_types else 0.0,
                },
                'progress_values': {
                    'count': len(answers_numeric),
                    'mean': float(np.mean(answers_numeric)),
                    'median': float(np.median(answers_numeric)),
                    'std': float(np.std(answers_numeric)),
                    'min': int(np.min(answers_numeric)),
                    'max': int(np.max(answers_numeric)),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'zero_count': zero_count,
                },
                'trajectory_statistics': {
                    'total_successful_trajectories': total_successful_trajectories,
                    'num_job_directories': len(job_dirs),
                    'job_dir_breakdown': job_dir_trajectory_counts,
                },
                'demo_id_statistics': {
                    'total_unique_demo_id': len(demo_id_to_exact),
                    'total_unique_demo_id_exact': sum(len(exacts) for exacts in demo_id_to_exact.values()),
                    'avg_demo_id_exact_per_demo_id': float(np.mean([len(exacts) for exacts in demo_id_to_exact.values()])),
                    'avg_pairs_per_demo_id': float(np.mean(list(demo_id_to_pairs.values()))),
                    'avg_pairs_per_demo_id_exact': float(np.mean(list(demo_id_exact_to_pairs.values()))),
                    'demo_id_breakdown': demo_id_stats,
                    'demo_id_exact_pair_counts': demo_id_exact_stats,
                },
                'success_vs_failure_statistics': {
                    'total_success_pairs': sum(demo_id_to_success_pairs.values()),
                    'total_failure_pairs': sum(demo_id_to_failure_pairs.values()),
                    'success_percentage': float(sum(demo_id_to_success_pairs.values()) / (sum(demo_id_to_success_pairs.values()) + sum(demo_id_to_failure_pairs.values())) * 100) if (sum(demo_id_to_success_pairs.values()) + sum(demo_id_to_failure_pairs.values())) > 0 else 0.0,
                    'failure_percentage': float(sum(demo_id_to_failure_pairs.values()) / (sum(demo_id_to_success_pairs.values()) + sum(demo_id_to_failure_pairs.values())) * 100) if (sum(demo_id_to_success_pairs.values()) + sum(demo_id_to_failure_pairs.values())) > 0 else 0.0,
                    'demo_ids_with_only_success': sum(1 for did in demo_id_to_exact.keys() if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] == 0),
                    'demo_ids_with_only_failure': sum(1 for did in demo_id_to_exact.keys() if demo_id_to_failure_pairs[did] > 0 and demo_id_to_success_pairs[did] == 0),
                    'demo_ids_with_both': sum(1 for did in demo_id_to_exact.keys() if demo_id_to_success_pairs[did] > 0 and demo_id_to_failure_pairs[did] > 0),
                },
                'split': split,
                'train_sample_interval': overlay_args.train_sample_interval,
                'include_successes': overlay_args.include_successes,
                'include_failures': overlay_args.include_failures,
            }

            # Save to both locations
            stats_path = viz_dir / 'combined_data_stats.json'
            stats_path_cache = viz_dir_cache / 'combined_data_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            with open(stats_path_cache, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            logger.info(f"💾 Saved detailed statistics to:")
            logger.info(f"  - {stats_path}")
            logger.info(f"  - {stats_path_cache}")
        except Exception as e:
            logger.warning(f"Failed to save statistics JSON: {e}")

        logger.info("\n" + "="*80)
        logger.info("END DATASET VISUALIZATION")
        logger.info("="*80 + "\n")
    else:
        if local_rank == 0:
            logger.warning("No data generated, skipping visualizations")

    if overlay_args.just_prepare_data:
        logger.info(f"Rank {local_rank}: Data preparation and visualization complete. Exiting as requested by --just_prepare_data")
        import sys
        sys.exit(0)

    # Combine success and failure data
    logger.info(f"Rank {local_rank}: Dataset ready with {len(combined_data)} pairs")

    # IMPORTANT: Shuffle combined_data BEFORE sharding to ensure data from multiple
    # datasets is properly mixed across all GPUs. All ranks use the same seed so they
    # compute the same shuffled order before taking their respective shards.
    logger.info(f"Rank {local_rank}: Shuffling {len(combined_data)} samples before sharding (seed=42)...")
    random.seed(42)
    random.shuffle(combined_data)
    logger.info(f"Rank {local_rank}: Shuffle complete")

    # IMPORTANT: Shard dataset per process for correct distributed training
    # Without sharding, all GPUs would process the same data (wasted computation)
    # With sharding, each GPU processes different data (correct and efficient)
    if world_size > 1:
        total_samples = len(combined_data)
        logger.info(f"Rank {local_rank}: Sharding {total_samples} samples across {world_size} processes")

        # Calculate this rank's shard indices
        samples_per_rank = total_samples // world_size
        start_idx = local_rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if local_rank < world_size - 1 else total_samples

        # Keep only this rank's shard
        combined_data = combined_data[start_idx:end_idx]
        logger.info(f"Rank {local_rank}: After sharding, keeping samples [{start_idx}:{end_idx}] = {len(combined_data)} pairs")

    # Combine datasets
    logger.info(f"Rank {local_rank}: Total combined pairs after sharding: {len(combined_data)}")

    # Clear image cache to free memory before training
    _load_and_prepare_image.cache_clear()

    # Synchronize before dataset creation
    if world_size > 1:
        try:
            dist.barrier()
            logger.info(f"Rank {local_rank}: Synchronized before dataset creation")
        except Exception as e:
            logger.warning(f"Failed to synchronize: {e}")

    # Create dataset and split (with shuffling)
    # Note: Each rank now only has its shard, so memory usage is 1/world_size
    logger.info(f"Rank {local_rank}: Creating HuggingFace dataset from {len(combined_data)} samples...")
    dataset = Dataset.from_list(combined_data)

    # Free the original list to save memory
    del combined_data
    import gc
    gc.collect()
    # Cast to lazy image loading BEFORE any operations to avoid loading images into memory
    logger.info(f"Rank {local_rank}: Casting images column to lazy loading format...")
    dataset = dataset.cast_column("images", hf_datasets.Sequence(hf_datasets.Image()))

    # Smaller test split to reduce memory per GPU
    test_size = max(2, len(dataset) // 20)
    logger.info(f"Rank {local_rank}: Splitting dataset with test_size={test_size}")
    # Note: shuffle=True in train_test_split will shuffle during split
    dataset = dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)

    # Get datasets without additional shuffling to avoid memory spikes
    # The dataloader will handle shuffling during training
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None
    logger.info(f"Rank {local_rank}: Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Rank {local_rank}: Evaluation samples: {len(eval_dataset)}")

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
    # Load processor for formatting eval-time prompts with images
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    except Exception as e:
        logger.warning(f"Failed to load AutoProcessor: {e}. Example IO logging will be limited.")
    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'num_parameters'):
        logger.info(f"Total parameters: {model.num_parameters()/1e9:.2f}B")

    ################
    # Training
    ################
    # Optional visual utils (e.g., qwen_vl_utils)
    qwen_vl_utils = None
    if importlib.util.find_spec("qwen_vl_utils") is not None:
        import qwen_vl_utils  # type: ignore

    class ExampleIOMonitorCallback(TrainerCallback):
        """Logs a few example inputs/outputs from eval set at each evaluation."""

        def __init__(self, eval_dataset, processor, system_prompt: str, max_examples: int = 3):
            self.eval_dataset = eval_dataset
            self.processor = processor
            self.system_prompt = system_prompt
            self.max_examples = max_examples

        def on_evaluate(self, args, state, control, **kwargs):
            if self.eval_dataset is None or len(self.eval_dataset) == 0:
                return
            if self.processor is None:
                logger.info("Processor not available; skipping example IO logging.")
                return

            model = kwargs.get("model")
            if model is None:
                return

            # Sample a few examples deterministically per evaluation step
            rng = random.Random(42 + int(state.global_step or 0))
            indices = list(range(len(self.eval_dataset)))
            rng.shuffle(indices)
            indices = indices[: self.max_examples]

            examples = [self.eval_dataset[i] for i in indices]

            # Build messages with overlay images
            texts = []
            images_batch = []
            fallbacks = []
            for ex in examples:
                try:
                    # Recover the system/user text and target (assistant content) from this sample's messages
                    messages_in = ex.get("messages", [])
                    system_text = self.system_prompt  # fallback to global
                    user_text = ""
                    target_text = None
                    for m in messages_in:
                        if m.get("role") == "system":
                            system_text = m.get("content", self.system_prompt)
                        if m.get("role") == "user":
                            user_text = m.get("content", "")
                        if m.get("role") == "assistant":
                            target_text = m.get("content", None)

                    # Compose multi-modal chat with single overlay image
                    mm_messages = [
                        {"role": "system", "content": [{"type": "text", "text": system_text}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": ex["images"][0]},  # Single overlay image
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ]

                    # If visual utils exist (e.g., Qwen), process images list accordingly
                    image_input = None
                    if qwen_vl_utils is not None:
                        image_input, _ = qwen_vl_utils.process_vision_info(mm_messages)
                    else:
                        # Fallback: load PIL image directly
                        try:
                            image_input = [ex["images"][0].convert("RGB")]
                        except Exception:
                            image_input = None

                    text = self.processor.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True
                    )

                    texts.append(text)
                    images_batch.append(image_input)

                    # Store original images if available for logging
                    original_imgs = ex.get("original_images", [None, None])
                    fallbacks.append({
                        "user_text": user_text,
                        "target": target_text,
                        "overlay_img": ex["images"][0].filename if hasattr(ex["images"][0], 'filename') else str(ex["images"][0]),
                        "original_img1": original_imgs[0] if original_imgs else None,
                        "original_img2": original_imgs[1] if original_imgs else None,
                    })
                except Exception as e:
                    logger.warning(f"Failed to prepare example for logging: {e}")

            if not texts:
                return

            try:
                inputs = self.processor(
                    text=texts,
                    images=images_batch if any(img is not None for img in images_batch) else None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(args, "max_length", 2048) or 2048,
                )
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=0.1,
                        top_p=0.95,
                        do_sample=True,
                    )

                decoded = self.processor.batch_decode(
                    outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )

                logger.info("\n===== Example IO (eval) =====")
                for fb, resp in zip(fallbacks, decoded):
                    log_data = {
                        "overlay_image": fb["overlay_img"],
                        "target": fb["target"],
                        "prediction": resp,
                    }
                    if fb["original_img1"] and fb["original_img2"]:
                        log_data["original_image1"] = fb["original_img1"]
                        log_data["original_image2"] = fb["original_img2"]

                    logger.info(json.dumps(log_data, ensure_ascii=False))
                logger.info("===== End Example IO =====\n")
            except Exception as e:
                logger.warning(f"Failed during example IO logging: {e}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Register example IO logging callback
    try:
        trainer.add_callback(ExampleIOMonitorCallback(
            eval_dataset=eval_dataset,
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            max_examples=3,
        ))
    except Exception as e:
        logger.warning(f"Could not add ExampleIOMonitorCallback: {e}")

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)