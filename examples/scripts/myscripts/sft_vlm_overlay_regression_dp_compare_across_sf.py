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
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
#     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
#     examples/scripts/myscripts/sft_vlm_overlay_regression_dp_compare_across_sf.py \
#     --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
#     --output_dir "outputs/expert_allPnP_$(date +%Y%m%d_%H%M%S)" \
#     --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove" \
#     --eval_strategy steps \
#     --logging_steps 500 \
#     --eval_steps 500 \
#     --save_steps 500 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 500 \
#     --learning_rate 1e-5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --report_to wandb \
#     --split train \
#     --train_val_split_index 
#     --train_sample_interval 1 \
#     --compare_interval 4,8,12,16 \
#     --max_exact_per_demo 50 \
#     --binary_or_exact_gt binary

just to load data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_dp_compare_across_sf.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPCabToCounter_mg_fixed_224" \
    --output_dir "outputs/TEST_$(date +%Y%m%d_%H%M%S)" \
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
    --split train \
    --train_val_split_index 40 \
    --train_sample_interval 50 \
    --compare_interval 200 \
    --max_exact_per_demo 50 \
    --binary_or_exact_gt binary 

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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def visualize_dataset(combined_data, output_dir, split_name="train", num_examples=6):
    """
    Create visualizations of the dataset including statistics and sample images.

    Args:
        combined_data: List of data dictionaries containing images and metadata
        output_dir: Directory to save visualization outputs
        split_name: Name of the split (train/val) for labeling
        num_examples: Number of example images to show
    """
    if len(combined_data) == 0:
        logger.warning("No data to visualize")
        return

    viz_dir = Path(output_dir) / "combined_data_analysis"
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating dataset visualizations in {viz_dir}")

    # Extract metadata from combined_data
    correct_answers = []
    demo_types = []  # success/failure
    demo_ids = []
    demo_ids_exact = []
    job_names = []
    task_tokens = []

    for item in combined_data:
        correct_answers.append(item.get("correct_answer", 0))
        demo_types.append(item.get("demo_success", "unknown"))
        demo_ids.append(item.get("demo_id", "unknown"))
        demo_ids_exact.append(item.get("demo_id_exact", "unknown"))
        job_names.append(item.get("job_name", "unknown"))
        task_tokens.append(item.get("task_token", "unknown"))

    # ============ Figure 1: Dataset Statistics ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dataset Statistics ({split_name} split, n={len(combined_data)})", fontsize=14, fontweight='bold')

    # 1. Distribution of correct answers
    ax1 = axes[0, 0]
    unique_answers = sorted(set(correct_answers))
    answer_counts = [correct_answers.count(a) for a in unique_answers]
    colors = ['#2ecc71' if a > 0 else '#e74c3c' if a < 0 else '#95a5a6' for a in unique_answers]
    ax1.bar([str(a) for a in unique_answers], answer_counts, color=colors)
    ax1.set_xlabel("Correct Answer (Progress Delta)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Correct Answers")
    ax1.tick_params(axis='x', rotation=45)

    # 2. Success vs Failure distribution
    ax2 = axes[0, 1]
    type_counts = {t: demo_types.count(t) for t in set(demo_types)}
    colors_pie = ['#2ecc71' if 'success' in t.lower() else '#e74c3c' for t in type_counts.keys()]
    wedges, texts, autotexts = ax2.pie(
        type_counts.values(),
        labels=type_counts.keys(),
        autopct='%1.1f%%',
        colors=colors_pie,
        explode=[0.05] * len(type_counts)
    )
    ax2.set_title("Demo Type Distribution")

    # 3. Samples per demo_id
    ax3 = axes[1, 0]
    demo_id_counts = defaultdict(int)
    for d in demo_ids:
        demo_id_counts[d] += 1
    sorted_demo_ids = sorted(demo_id_counts.keys())
    ax3.bar(range(len(sorted_demo_ids)), [demo_id_counts[d] for d in sorted_demo_ids], color='#3498db')
    ax3.set_xlabel("Demo ID")
    ax3.set_ylabel("Sample Count")
    ax3.set_title(f"Samples per Demo ID (n={len(sorted_demo_ids)} demos)")
    if len(sorted_demo_ids) > 20:
        ax3.set_xticks([])
    else:
        ax3.set_xticks(range(len(sorted_demo_ids)))
        ax3.set_xticklabels(sorted_demo_ids, rotation=45)

    # 4. Samples per job/task
    ax4 = axes[1, 1]
    job_counts = defaultdict(int)
    for j in job_names:
        # Extract task name from job_name (e.g., PnPCounterToStove)
        task_name = j
        for key in ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToMicrowave",
                    "PnPMicrowaveToCounter", "PnPStoveToCounter", "PnPCounterToStove",
                    "PnPCounterToSink", "PnPSinkToCounter", "PnPCoffeeServeMug", "PnPCloseDrawer"]:
            if key in j:
                task_name = key
                break
        job_counts[task_name] += 1

    sorted_jobs = sorted(job_counts.keys())
    ax4.barh(range(len(sorted_jobs)), [job_counts[j] for j in sorted_jobs], color='#9b59b6')
    ax4.set_yticks(range(len(sorted_jobs)))
    ax4.set_yticklabels(sorted_jobs)
    ax4.set_xlabel("Sample Count")
    ax4.set_title("Samples per Task Type")

    plt.tight_layout()
    stats_path = viz_dir / f"dataset_statistics_{split_name}.png"
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved statistics plot to {stats_path}")

    # ============ Figure 2: Sample Images with Metadata ============
    # Sample diverse examples
    num_to_show = min(num_examples, len(combined_data))

    # Try to get diverse samples (mix of success/failure, different answers)
    indices_by_type = defaultdict(list)
    for i, item in enumerate(combined_data):
        key = (item.get("demo_success", "unknown"), item.get("correct_answer", 0) > 0)
        indices_by_type[key].append(i)

    # Sample from each category
    sampled_indices = []
    rng = random.Random(42)
    for key, indices in indices_by_type.items():
        n_sample = max(1, num_to_show // len(indices_by_type))
        sampled_indices.extend(rng.sample(indices, min(n_sample, len(indices))))

    # Truncate if needed
    sampled_indices = sampled_indices[:num_to_show]

    if len(sampled_indices) == 0:
        sampled_indices = list(range(min(num_to_show, len(combined_data))))

    # Create figure for sample images
    n_cols = min(3, num_to_show)
    n_rows = (num_to_show + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for plot_idx, data_idx in enumerate(sampled_indices):
        item = combined_data[data_idx]
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)

        # Load the overlay image
        image_path = item.get("images", [None])[0]
        if image_path:
            try:
                if isinstance(image_path, str):
                    img = Image.open(image_path)
                else:
                    img = image_path  # Already PIL Image
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed to load:\n{e}", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No image", ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')

        # Create metadata label
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "unknown")
        demo_id = item.get("demo_id", "?")
        task_token = item.get("task_token", None)
        job = item.get("job_name", "unknown")

        # Use task_token if available, otherwise extract from job_name
        if task_token:
            short_task = task_token
        else:
            short_task = job
            for key in ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToMicrowave",
                        "PnPMicrowaveToCounter", "PnPStoveToCounter", "PnPCounterToStove",
                        "PnPCounterToSink", "PnPSinkToCounter", "PnPCoffeeServeMug", "PnPCloseDrawer"]:
                if key in job:
                    short_task = key
                    break

        # Color code by answer direction
        title_color = '#2ecc71' if answer > 0 else '#e74c3c' if answer < 0 else '#333333'

        title = f"Answer: {answer} | Type: {demo_type}\nDemo: {demo_id} | {short_task}"
        ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')

    plt.suptitle(f"Sample Images ({split_name} split)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    samples_path = viz_dir / f"sample_images_{split_name}.png"
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sample images to {samples_path}")

    # ============ Save Statistics Summary as JSON ============
    # Count task tokens
    task_token_counts = defaultdict(int)
    for t in task_tokens:
        task_token_counts[t] += 1

    stats_summary = {
        "split": split_name,
        "total_samples": len(combined_data),
        "answer_distribution": {str(k): v for k, v in zip(unique_answers, answer_counts)},
        "demo_type_distribution": dict(type_counts),
        "num_unique_demo_ids": len(set(demo_ids)),
        "num_unique_demo_ids_exact": len(set(demo_ids_exact)),
        "task_distribution": dict(job_counts),
        "task_token_distribution": dict(task_token_counts),
        "samples_per_demo_id": {str(k): v for k, v in demo_id_counts.items()},
        "trajectory_statistics": {
            "total_successful_trajectories": sum(1 for t in demo_types if t == "success"),
            "total_failure_trajectories": sum(1 for t in demo_types if t == "failure"),
        }
    }

    stats_json_path = viz_dir / f"combined_data_stats.json"
    with open(stats_json_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"Saved statistics JSON to {stats_json_path}")

    # ============ Figure 3: Detailed Image Grid with Original Pairs ============
    # Show a few examples with both original images and the overlay
    num_detailed = min(4, len(sampled_indices))
    fig = plt.figure(figsize=(15, 4 * num_detailed))
    gs = gridspec.GridSpec(num_detailed, 3, width_ratios=[1, 1, 2], hspace=0.3, wspace=0.1)

    for row_idx in range(num_detailed):
        data_idx = sampled_indices[row_idx]
        item = combined_data[data_idx]

        # Original image 1
        ax1 = fig.add_subplot(gs[row_idx, 0])
        orig_imgs = item.get("original_images", [None, None])
        if orig_imgs and len(orig_imgs) > 0 and orig_imgs[0]:
            try:
                if isinstance(orig_imgs[0], str):
                    img1 = Image.open(orig_imgs[0])
                else:
                    img1 = orig_imgs[0]
                ax1.imshow(img1)
            except:
                ax1.text(0.5, 0.5, "Load failed", ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')
        ax1.set_title("Left Image", fontsize=9)

        # Original image 2
        ax2 = fig.add_subplot(gs[row_idx, 1])
        if orig_imgs and len(orig_imgs) > 1 and orig_imgs[1]:
            try:
                if isinstance(orig_imgs[1], str):
                    img2 = Image.open(orig_imgs[1])
                else:
                    img2 = orig_imgs[1]
                ax2.imshow(img2)
            except:
                ax2.text(0.5, 0.5, "Load failed", ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title("Right Image", fontsize=9)

        # Overlay image
        ax3 = fig.add_subplot(gs[row_idx, 2])
        overlay_path = item.get("images", [None])[0]
        if overlay_path:
            try:
                if isinstance(overlay_path, str):
                    overlay_img = Image.open(overlay_path)
                else:
                    overlay_img = overlay_path
                ax3.imshow(overlay_img)
            except:
                ax3.text(0.5, 0.5, "Load failed", ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')

        # Metadata
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "?")
        demo_id = item.get("demo_id", "?")
        task_token = item.get("task_token", None)
        job = item.get("job_name", "")

        # Use task_token if available, otherwise extract from job_name
        if task_token:
            short_task = task_token
        else:
            short_task = "unknown"
            for key in ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToMicrowave",
                        "PnPMicrowaveToCounter", "PnPStoveToCounter", "PnPCounterToStove",
                        "PnPCounterToSink", "PnPSinkToCounter", "PnPCoffeeServeMug", "PnPCloseDrawer"]:
                if key in job:
                    short_task = key
                    break

        title_color = '#2ecc71' if answer > 0 else '#e74c3c' if answer < 0 else '#333333'
        ax3.set_title(f"Overlay | Answer: {answer} | {demo_type} | Demo {demo_id} | {short_task}",
                      fontsize=10, color=title_color, fontweight='bold')

    plt.suptitle(f"Detailed Examples: Original Pairs + Overlay ({split_name})", fontsize=12, fontweight='bold')
    detailed_path = viz_dir / f"detailed_examples_{split_name}.png"
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved detailed examples to {detailed_path}")

    logger.info(f"Visualization complete. Files saved to {viz_dir}")
    return viz_dir


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
    # Simple task tokens - easy for the model to learn distinct task identities
    TASK_TOKENS = {
        "PnPCounterToCab": "[COUNTER_TO_CAB]",
        "PnPCabToCounter": "[CAB_TO_COUNTER]",
        "PnPCounterToMicrowave": "[COUNTER_TO_MICROWAVE]",
        "PnPMicrowaveToCounter": "[MICROWAVE_TO_COUNTER]",
        "PnPStoveToCounter": "[STOVE_TO_COUNTER]",
        "PnPCounterToStove": "[COUNTER_TO_STOVE]",
        "PnPCounterToSink": "[COUNTER_TO_SINK]",
        "PnPSinkToCounter": "[SINK_TO_COUNTER]",
        "PnPCoffeeServeMug": "[COFFEE_SERVE_MUG]",
        "PnPCloseDrawer": "[CLOSE_DRAWER]",
    }

    # Minimal system prompt - the task token does the heavy lifting
    SYSTEM_PROMPT = "Compare robot task progress. Respond with a number: positive if right image shows more progress, negative if less."

    # User prompt template - task token is prominent, right before asking for comparison
    USER_PROMPT_TEMPLATE = """Task: {task_token}
Which image shows more task progress? Respond with a number from -100 to 100."""

    # Load failure dataset from all job directories

    # Find all sep29_job* directories
    import glob
    import os
    split = overlay_args.split
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
                    # Fix absolute paths to be relative to current base_dataset_path  
                    old_base = item["images"][0].split('overlay_images_binary')[0]
                    new_base = overlay_args.base_dataset_path
                    for item in path_data:                                                                                                                                                                                          
                        # Fix overlay image path                                                                                                                                                                                    
                        item["images"] = [img.replace(old_base, new_base) for img in item["images"]]                                                                                                                                

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
            all_dirs = sorted(glob.glob(f'{overlay_args.base_dataset_path}/*'))
            # Filter to only include directories with eval_log.json
            job_dirs = [d for d in all_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'eval_log.json'))]

            if split == 'val':
                job_dirs = job_dirs[-overlay_args.train_val_split_index:]
            elif split=='train':
                job_dirs = job_dirs[:-overlay_args.train_val_split_index]
            else:
                job_dirs = job_dirs[:int(split)]
            logger.info(f"Found {len(job_dirs)} job directories.")
            print(f"Loading pre-extracted images from {len(job_dirs)} for {split} split")

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
                pdb.set_trace()
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

                # Find matching task token based on job_name
                task_token = None
                for task_key, token in TASK_TOKENS.items():
                    if task_key in job_name:
                        task_token = token
                        break

                if task_token is None:
                    raise ValueError(f"No task token found for job name: {job_name}")

                # Create user prompt with task token
                user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": str(item[1])}
                ]
                final_combined_data.append({
                    "images": [item[0]],
                    "correct_answer": item[1],
                    "messages": messages,
                    "demo_id": item[4],
                    "demo_id_exact": item[5],
                    "demo_success": item[6],
                    "job_name": job_name,
                    "task_token": task_token
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

    # ============================================================================================
    # VISUALIZATION: Show examples and statistics of combined_data (BEFORE SHARDING)
    # ============================================================================================
    # Only rank 0 needs to generate visualizations to save memory
    # IMPORTANT: Do this BEFORE sharding to get accurate full dataset statistics

    if local_rank == 0 and len(combined_data) > 0:
        logger.info("="*80)
        logger.info("Generating dataset visualizations...")
        logger.info("="*80)
        try:
            # Save to overlay cache directory
            visualize_dataset(
                combined_data=combined_data,
                output_dir=overlay_dir,
                split_name=split,
                num_examples=8
            )
            # Also save to training output directory
            visualize_dataset(
                combined_data=combined_data,
                output_dir=training_args.output_dir,
                split_name=split,
                num_examples=8
            )
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()

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