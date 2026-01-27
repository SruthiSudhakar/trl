#!/usr/bin/env python
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

"""
Evaluation script for SFT VLM Overlay Regression models trained with
examples/scripts/myscripts/sft_vlm_overlay_regression_dp.py

This script loads pre-cached datasets from the training script to ensure
train/val split consistency, and generates visualizations of model predictions.

Features:
- Computes MAE, RMSE, within-tolerance rate, and sign accuracy
- Generates visualization plots showing best/worst predictions
- Creates detailed visualizations with side-by-side image comparisons
- Plots error distribution and prediction vs ground truth scatter plots

Usage (example):
# Evaluate trained checkpoint
CUDA_VISIBLE_DEVICES=1 python3 examples/scripts/myscripts/evaluate_sft_vlm_overlay_regression_dp.py \
    --model_name_or_path outputs/jan21/StoveToCounter_20260121_153119/checkpoint-17000 \
    --base_model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --dataset_cache_file /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter/overlay_images_binary/dataset_cache_val_1__4_8_12_16__True_True.pkl \
    --batch_size 300 \
    --visualize \
    --num_visualize 10 \
    --seed 42 \
    --num_samples 10000

CUDA_VISIBLE_DEVICES=2 python3 examples/scripts/myscripts/evaluate_sft_vlm_overlay_regression_dp.py \
    --model_name_or_path outputs/expert_allPnP_20260122_193654/checkpoint-28500 \
    --base_model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --dataset_cache_file /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter/overlay_images_binary/dataset_cache_val_1__4_8_12_16__True_True.pkl \
    --batch_size 300 \
    --visualize \
    --num_visualize 10 \
    --seed 42 \
    --num_samples 10000

"""

import argparse
import json
import math
import os
import pdb
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

import importlib.util

from trl.scripts.utils import logger

# Optional qwen visual utils if present
qwen_vl_utils = None
if importlib.util.find_spec("qwen_vl_utils") is not None:
    import qwen_vl_utils  # type: ignore


# -------------------------------
# Overlay image utilities (mirrors training script)
# -------------------------------
def _load_and_prepare_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32)


def create_overlay_image(image1_path: str, image2_path: str, method: str = 'side_by_side') -> Image.Image:
    """
    Creates an overlayed/combined image from two input images.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        method: Overlay method ('side_by_side' supported)

    Returns:
        PIL Image: The combined/overlayed image
    """
    arr1 = _load_and_prepare_image(image1_path)
    arr2 = _load_and_prepare_image(image2_path)

    if arr1.shape != arr2.shape:
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)
            arr2 = np.array(img2, dtype=np.float32)

    if method == 'side_by_side':
        height, width = arr1.shape[:2]
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))
        combined = Image.new('RGB', (width * 2 + 2, height))
        combined.paste(img1, (0, 0))
        combined.paste(Image.new('RGB', (2, height), (255, 255, 0)), (width, 0))  # Yellow separator
        combined.paste(img2, (width + 2, 0))
        return combined

    raise ValueError(f"Unknown overlay method: {method}")


# -------------------------------
# Prompts (mirrors training script semantics)
# -------------------------------
TASK_DESCRIPTION = "Pick and place an object from the sink to the plate on the counter"


def build_prompts(overlay_method: str):
    if overlay_method != 'side_by_side':
        raise Exception('incorrect overlat method')
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
    system_prompt = f"""You are an expert roboticist tasked to compare a side-by-side of 2 images from a robot demonstration and determine which side shows more progress toward completing the task.
    The robot task is: INSERT_TASK_DESC_HERE
    You will be given a side-by-side of 2 images from the same demonstration, and you need to identify how much closer or behind in task completion is the right image compared to the left."""

    user_prompt = f"""Look at these two side-by-side images of a robot performing the task. \

    Left side image: Shows the robot at one point during the task. \
    Right side image: Shows the robot at another point during the task. \

    Task: Compare the two images and determine the relative progress difference. \
    - If the right image shows more progress toward task completion, respond with a positive number of how much farther (1 to 100) \
    - If the right image shows less progress toward task completion, respond with a negative number (-1 to -100) \

    The number should represent how much more or less progress the right image shows compared to the left."""
    return system_prompt, user_prompt, TASK_DESC_TO_SYSTEM_PROMPT


# -------------------------------
# Visualization utilities
# -------------------------------
def visualize_predictions(results: List[Dict[str, Any]], output_dir: str, num_examples: int = 10, sort_by: str = "worst"):
    """
    Visualize a grid of examples with their predictions and ground truth.

    Args:
        results: List of result dictionaries from evaluation
        output_dir: Directory to save visualization
        num_examples: Number of examples to visualize
        sort_by: How to select examples - "worst", "best", "random", "correct", "incorrect"
    """
    # Filter out results with None predictions
    valid_results = [r for r in results if r['prediction'] is not None and r['abs_error'] is not None]

    if not valid_results:
        print("No valid results to visualize")
        return

    # Select examples based on criteria
    if sort_by == "worst":
        selected = sorted(valid_results, key=lambda x: x['abs_error'], reverse=True)[:num_examples]
    elif sort_by == "best":
        selected = sorted(valid_results, key=lambda x: x['abs_error'])[:num_examples]
    elif sort_by == "correct":
        correct = [r for r in valid_results if r['within_tolerance']]
        selected = correct[:num_examples] if correct else []
    elif sort_by == "incorrect":
        incorrect = [r for r in valid_results if not r['within_tolerance']]
        selected = incorrect[:num_examples] if incorrect else []
    elif sort_by == "random":
        import random
        selected = random.sample(valid_results, min(num_examples, len(valid_results)))
    else:
        selected = valid_results[:num_examples]

    if not selected:
        print(f"No examples found for sort_by={sort_by}")
        return

    # Create figure with subplots
    n_cols = 2
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, result in enumerate(selected):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Load and display the overlay image
        if os.path.exists(result['overlay_image']):
            img = Image.open(result['overlay_image']).convert('RGB')
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', transform=ax.transAxes)

        # Prepare title with prediction info
        gt = result['ground_truth']
        pred = result['prediction']
        abs_err = result['abs_error']
        sign_correct = result['sign_correct']
        within_tol = result['within_tolerance']

        # Color code based on accuracy
        if within_tol:
            title_color = 'green'
            status = '✓'
        else:
            title_color = 'red'
            status = '✗'

        title = f"{status} GT: {gt:.1f} | Pred: {pred:.1f} | Error: {abs_err:.1f}\n"
        title += f"Sign: {'✓' if sign_correct else '✗'} | Demo: {result['demo_name']}\n"
        title += f"Frames: {result['frame_idx_1']} → {result['frame_idx_2']}"

        ax.set_title(title, fontsize=10, color=title_color, weight='bold')
        ax.axis('off')

        # Add raw completion text at the bottom
        raw_text = result.get('raw_completion', '')[:100]  # Truncate if too long
        ax.text(0.5, -0.05, f"Raw: {raw_text}", ha='center', va='top',
                transform=ax.transAxes, fontsize=8, style='italic', wrap=True)

    # Hide empty subplots
    for idx in range(len(selected), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, f"visualization_{sort_by}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close()


def create_detailed_visualization(result: Dict[str, Any], output_dir: str, idx: int):
    """
    Create a detailed single-example visualization showing original images separately if available.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Load overlay image
    if os.path.exists(result['overlay_image']):
        overlay_img = Image.open(result['overlay_image']).convert('RGB')
        axes[1].imshow(overlay_img)
        axes[1].set_title('Overlay Image (Side-by-Side)', fontsize=12, weight='bold')
        axes[1].axis('off')

        # Try to load original images if available
        if result.get('original_images') and len(result['original_images']) >= 2:
            img1_path, img2_path = result['original_images'][:2]

            if os.path.exists(img1_path):
                img1 = Image.open(img1_path).convert('RGB')
                axes[0].imshow(img1)
                axes[0].set_title(f"Left Image (Frame {result['frame_idx_1']})", fontsize=12)
                axes[0].axis('off')

            if os.path.exists(img2_path):
                img2 = Image.open(img2_path).convert('RGB')
                axes[2].imshow(img2)
                axes[2].set_title(f"Right Image (Frame {result['frame_idx_2']})", fontsize=12)
                axes[2].axis('off')

    # Add prediction information
    gt = result['ground_truth']
    pred = result['prediction']
    abs_err = result['abs_error']

    info_text = f"Ground Truth: {gt:.1f}\n"
    info_text += f"Prediction: {pred:.1f}\n"
    info_text += f"Absolute Error: {abs_err:.1f}\n"
    info_text += f"Within Tolerance: {result['within_tolerance']}\n"
    info_text += f"Sign Correct: {result['sign_correct']}\n\n"
    info_text += f"Demo: {result['demo_name']}\n"
    info_text += f"Raw Completion:\n{result.get('raw_completion', 'N/A')}"

    plt.suptitle(info_text, fontsize=10, ha='left', x=0.1, y=0.05, family='monospace')
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])

    save_path = os.path.join(output_dir, f"detailed_example_{idx:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_error_distribution(results: List[Dict[str, Any]], output_dir: str):
    """
    Plot error distribution histogram and scatter plot.
    """
    valid_results = [r for r in results if r['prediction'] is not None]

    if not valid_results:
        print("No valid results to plot error distribution")
        return

    errors = [r['error'] for r in valid_results]
    abs_errors = [r['abs_error'] for r in valid_results]
    ground_truths = [r['ground_truth'] for r in valid_results]
    predictions = [r['prediction'] for r in valid_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Error histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].set_xlabel('Error (Prediction - Ground Truth)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Error Distribution', fontsize=14, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Absolute error histogram
    axes[0, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(x=np.mean(abs_errors), color='r', linestyle='--', linewidth=2,
                       label=f'MAE: {np.mean(abs_errors):.2f}')
    axes[0, 1].set_xlabel('Absolute Error', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Absolute Error Distribution', fontsize=14, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Prediction vs Ground Truth scatter
    axes[1, 0].scatter(ground_truths, predictions, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(min(ground_truths), min(predictions))
    max_val = max(max(ground_truths), max(predictions))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    axes[1, 0].set_xlabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Prediction', fontsize=12)
    axes[1, 0].set_title('Prediction vs Ground Truth', fontsize=14, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal', adjustable='box')

    # 4. Error vs Ground Truth
    axes[1, 1].scatter(ground_truths, errors, alpha=0.5, s=20, c=abs_errors, cmap='coolwarm')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Ground Truth', fontsize=12)
    axes[1, 1].set_ylabel('Error', fontsize=12)
    axes[1, 1].set_title('Error vs Ground Truth', fontsize=14, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Absolute Error', fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved error distribution plot to: {save_path}")
    plt.close()


# -------------------------------
# Args
# -------------------------------
@dataclass
class EvaluationArgs:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained model checkpoint (adapter or full)"},
    )
    base_model_name_or_path: str = field(
        default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Base model path (used when loading PEFT adapters)"},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "float32 | float16 | bfloat16 | auto"},
    )
    batch_size: int = field(default=8)
    max_prompt_length: int = field(default=2048)
    max_completion_length: int = field(default=64)
    temperature: float = field(default=0.1)
    top_p: float = field(default=0.95)
    device: str = field(default="cuda")
    use_quantization: bool = field(default=False)
    trust_remote_code: bool = field(default=True)
    # Dataset parameters
    dataset_cache_file: str = field(
        default="",
        metadata={"help": "Path to cached dataset pickle file from training script"}
    )
    dataset_split: str = field(
        default="val",
        metadata={"help": "Dataset split: train | val"}
    )
    overlay_method: str = field(
        default="side_by_side",
        metadata={"help": "Overlay method used in training"}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Random subset of samples to evaluate (optional)"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for deterministic evaluation"})
    # Metrics
    tolerance: int = field(default=3, metadata={"help": "abs error tolerance for within_tolerance"})
    save_completions: bool = field(default=True)
    # Visualization
    visualize: bool = field(default=True, metadata={"help": "Generate visualization plots"})
    num_visualize: int = field(default=10, metadata={"help": "Number of examples to visualize"})


class OverlayRegressionEvaluator:
    def __init__(self, args: EvaluationArgs):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        base_dir = self.args.model_name_or_path or self.args.base_model_name_or_path
        self.output_dir = os.path.join(base_dir, f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.overlay_method = 'side_by_side'
        self.system_prompt, self.user_prompt, self.task_descriptions_dict = build_prompts(self.overlay_method)
        self._load_model()

    def _load_model(self) -> None:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        dtype = dtype_map.get(self.args.dtype, "auto")

        quantization_config = None
        if self.args.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype if dtype != "auto" else torch.bfloat16,
            )

        # Determine if model_name_or_path contains a PEFT adapter
        processor_path = self.args.base_model_name_or_path
        if self.args.model_name_or_path and os.path.exists(os.path.join(self.args.model_name_or_path, "adapter_config.json")):
            base_model_path = self.args.base_model_name_or_path
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=self.args.trust_remote_code,
                quantization_config=quantization_config,
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.args.model_name_or_path,
                is_trainable=False,
            )
        elif self.args.model_name_or_path:
            processor_path = self.args.model_name_or_path
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.args.model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=self.args.trust_remote_code,
                quantization_config=quantization_config,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.args.base_model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=self.args.trust_remote_code,
                quantization_config=quantization_config,
            )

        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.model.eval()

    def _build_eval_pairs(self) -> List[Dict[str, Any]]:
        """Load pre-cached dataset from training script's pickle file."""
        dataset_cache_file = self.args.dataset_cache_file

        if not os.path.exists(dataset_cache_file):
            raise FileNotFoundError(
                f"Dataset cache file not found: {dataset_cache_file}\n"
                f"Please run the training script first to generate the cache, or provide the correct path."
            )

        print(f"Loading pre-cached dataset from {dataset_cache_file}")
        with open(dataset_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Loaded {len(cached_data)} pairs from cache")

        # Convert cached data format to evaluation format
        data = []
        for item in tqdm(cached_data, desc="Converting cached data to evaluation format"):
            # Extract ground truth from messages
            ground_truth = None
            for msg in item.get('messages', []):
                if msg.get('role') == 'assistant':
                    try:
                        ground_truth = int(msg['content'])
                    except (ValueError, TypeError):
                        ground_truth = None
                        print(f"Warning: Could not parse ground truth from: {msg['content']}")

            if ground_truth is None:
                continue

            # Extract demo name and frame indices from overlay image path if possible
            overlay_path = item['images'][0] if isinstance(item['images'], list) else item['images']
            filename = Path(overlay_path).stem
            parts = filename.split('_')

            # Initialize defaults
            demo_name = filename
            frame_idx_1 = 0
            frame_idx_2 = 0

            # Try to parse frame indices from filename
            # Expected format: success_<path>_frame_XXXXXX_frame_YYYYYY.png
            # Try to parse frame indices from filename using regex (more robust)
            # Expected formats:
            # Old: ..._frame_XXXXXX_frame_YYYYYY.png
            # New: ..._frame_XXXXXX.png_..._frame_YYYYYY.png.png

            import re
            frame_parts = re.findall(r'frame_(\d+)', filename)
            if len(frame_parts) >= 2:
                frame_idx_1 = int(frame_parts[-2])
                frame_idx_2 = int(frame_parts[-1])
                
                # Extract demo name by removing frame parts
                # This is a best-effort cleanup
                clean_name = filename
                for part in frame_parts:
                    clean_name = clean_name.replace(f'frame_{part}', '')
                clean_name = clean_name.replace('.png', '').replace('__', '_').strip('_')
                if clean_name:
                    demo_name = clean_name


            orig_img1 = item['original_images'][0]
            demo_id_exact = Path(orig_img1).parent.name
            demo_id = demo_id_exact.split('_')[0] if demo_id_exact else None

            data.append({
                "demo_name": demo_name,
                "frame_idx_1": frame_idx_1,
                "frame_idx_2": frame_idx_2,
                "overlay_image": overlay_path,
                "ground_truth": ground_truth,
                "original_images": item.get('original_images'),
                "demo_id": demo_id,
                "demo_id_exact": demo_id_exact,
            })

        print(f"Converted {len(data)} pairs for evaluation")

        # Optionally sample a random subset
        if self.args.num_samples is not None and self.args.num_samples < len(data):
            import random
            random.seed(self.args.seed)
            data = random.sample(data, self.args.num_samples)
            print(f"Randomly sampled {len(data)} pairs (seed={self.args.seed})")

        return data

    def _prepare_inputs(self, batch: List[Dict[str, Any]]):
        texts: List[str] = []
        images_batch: List[Any] = []

        for item in batch:
            job_name = item['demo_name']
            # Find matching task description based on job_name
            task_desc = None
            for task_key, task_description in self.task_descriptions_dict.items():
                if task_key in job_name:
                    task_desc = task_description
                    break

            # Create system prompt with task-specific description
            if task_desc:
                sample_system_prompt = self.system_prompt.replace("INSERT_TASK_DESC_HERE", task_desc)
            else:
                raise Exception("Task description not found for job name: " + job_name)

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": sample_system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["overlay_image"]},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ]

            if qwen_vl_utils is not None:
                image_input, _ = qwen_vl_utils.process_vision_info(conversation)  # type: ignore
            else:
                # Fallback: PIL image list
                image_input = [Image.open(item["overlay_image"]).convert("RGB")]

            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            texts.append(text)
            images_batch.append(image_input)

        inputs = self.processor(
            text=texts,
            images=images_batch if images_batch else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs

    def _generate(self, inputs) -> List[str]:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=self.args.temperature > 0,
            )

        # Decode new tokens only
        input_ids = inputs["input_ids"]
        completions: List[str] = []
        for i in range(outputs.size(0)):
            gen_tokens = outputs[i, input_ids[i].shape[0]:]
            text = self.processor.decode(gen_tokens, skip_special_tokens=True)
            completions.append(text.strip())
        return completions

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        try:
            # Keep only the first number, allow negative and decimals
            import re
            m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            if not m:
                return None
            value = float(m.group(0))
            # Clip to plausible bounds
            if value < -100 or value > 100:
                return value  # still return; caller may clip
            return value
        except Exception:
            return None

    def run(self) -> Dict[str, Any]:
        dataset = self._build_eval_pairs()
        print(f'evaluating on {len(dataset)} samples')
        results: List[Dict[str, Any]] = []
        errors: List[float] = []
        abs_errors: List[float] = []
        within_tol: List[int] = []
        sign_correct: List[int] = []

        batch_size = max(1, int(self.args.batch_size))
        num_batches = (len(dataset) + batch_size - 1) // batch_size

        for b in tqdm(range(num_batches), desc=f"Evaluating {num_batches} batches"):
            batch = dataset[b * batch_size : (b + 1) * batch_size]
            if not batch:
                continue
            inputs = self._prepare_inputs(batch)
            completions = self._generate(inputs)

            for item, completion in zip(batch, completions):
                pred = self._extract_number(completion)
                gt = float(item["ground_truth"])
                clipped_pred = None if pred is None else max(-100.0, min(100.0, pred))
                if clipped_pred is None:
                    err = None
                    abs_err = None
                    tol_ok = 0
                    sign_ok = 0
                else:
                    err = clipped_pred - gt
                    abs_err = abs(err)
                    tol_ok = int(abs_err <= self.args.tolerance)
                    # Sign is correct if both have same sign (handling zero as a special case)
                    if gt == 0:
                        sign_ok = clipped_pred==0  # For zero GT, pred should be close to zero
                    else:
                        sign_ok = int(np.sign(gt) == np.sign(clipped_pred))
                    errors.append(err)
                    abs_errors.append(abs_err)
                    within_tol.append(tol_ok)
                    sign_correct.append(sign_ok)

                results.append({
                    "demo_name": item["demo_name"],
                    "demo_id": item.get("demo_id"),
                    "demo_id_exact": item.get("demo_id_exact"),
                    "frame_idx_1": item["frame_idx_1"],
                    "frame_idx_2": item["frame_idx_2"],
                    "frame_diff": item["frame_idx_2"] - item["frame_idx_1"],
                    "overlay_image": item["overlay_image"],
                    "ground_truth": gt,
                    "prediction": clipped_pred,
                    "raw_completion": completion,
                    "error": err,
                    "abs_error": abs_err,
                    "within_tolerance": bool(tol_ok),
                    "sign_correct": bool(sign_ok),
                })

        mae = float(np.mean(abs_errors)) if abs_errors else math.nan
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else math.nan
        tol_rate = float(np.mean(within_tol)) if within_tol else math.nan
        sign_acc = float(np.mean(sign_correct)) if sign_correct else math.nan

        # Compute per demo_id statistics
        from collections import defaultdict
        demo_id_stats = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'abs_errors': [],
            'within_tolerance': 0,
            'sign_correct': 0
        })

        demo_id_exact_stats = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'abs_errors': [],
            'within_tolerance': 0,
            'sign_correct': 0
        })

        # Compute per frame_diff bucket statistics
        frame_diff_stats = defaultdict(lambda: {
            'total': 0,
            'abs_errors': [],
            'within_tolerance': 0,
            'sign_correct': 0
        })

        for result in results:
            if result['prediction'] is None:
                continue

            demo_id = result.get('demo_id')
            demo_id_exact = result.get('demo_id_exact')

            if demo_id:
                demo_id_stats[demo_id]['total'] += 1
                demo_id_stats[demo_id]['abs_errors'].append(result['abs_error'])
                if result['within_tolerance']:
                    demo_id_stats[demo_id]['within_tolerance'] += 1
                if result['sign_correct']:
                    demo_id_stats[demo_id]['sign_correct'] += 1

            if demo_id_exact:
                demo_id_exact_stats[demo_id_exact]['total'] += 1
                demo_id_exact_stats[demo_id_exact]['abs_errors'].append(result['abs_error'])
                if result['within_tolerance']:
                    demo_id_exact_stats[demo_id_exact]['within_tolerance'] += 1
                if result['sign_correct']:
                    demo_id_exact_stats[demo_id_exact]['sign_correct'] += 1

            # Compute frame difference and bucket it
            frame_diff = result['frame_idx_2'] - result['frame_idx_1']
            frame_diff_stats[frame_diff]['total'] += 1
            frame_diff_stats[frame_diff]['abs_errors'].append(result['abs_error'])
            if result['within_tolerance']:
                frame_diff_stats[frame_diff]['within_tolerance'] += 1
            if result['sign_correct']:
                frame_diff_stats[frame_diff]['sign_correct'] += 1

        # Compute summary statistics for each demo_id
        demo_id_summary = {}
        for demo_id, stats in demo_id_stats.items():
            demo_id_summary[demo_id] = {
                'total_samples': stats['total'],
                'mae': float(np.mean(stats['abs_errors'])) if stats['abs_errors'] else math.nan,
                'within_tolerance_rate': stats['within_tolerance'] / stats['total'] if stats['total'] > 0 else 0.0,
                'sign_accuracy': stats['sign_correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            }

        demo_id_exact_summary = {}
        for demo_id_exact, stats in demo_id_exact_stats.items():
            demo_id_exact_summary[demo_id_exact] = {
                'total_samples': stats['total'],
                'mae': float(np.mean(stats['abs_errors'])) if stats['abs_errors'] else math.nan,
                'within_tolerance_rate': stats['within_tolerance'] / stats['total'] if stats['total'] > 0 else 0.0,
                'sign_accuracy': stats['sign_correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            }

        # Compute summary statistics for each frame_diff bucket
        frame_diff_summary = {}
        for frame_diff, stats in frame_diff_stats.items():
            frame_diff_summary[frame_diff] = {
                'total_samples': stats['total'],
                'mae': float(np.mean(stats['abs_errors'])) if stats['abs_errors'] else math.nan,
                'within_tolerance_rate': stats['within_tolerance'] / stats['total'] if stats['total'] > 0 else 0.0,
                'sign_accuracy': stats['sign_correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            }

        summary: Dict[str, Any] = {
            "model_path": self.args.model_name_or_path or self.args.base_model_name_or_path,
            "overlay_method": self.overlay_method,
            "dataset_split": self.args.dataset_split,
            "dataset_cache_file": self.args.dataset_cache_file,
            "num_samples": self.args.num_samples,
            "num_pairs": len(dataset),
            "batch_size": self.args.batch_size,
            "tolerance": self.args.tolerance,
            "seed": self.args.seed,
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "within_tolerance_rate": tol_rate,
                "sign_accuracy": sign_acc,
            },
            "per_demo_id_metrics": demo_id_summary,
            "per_demo_id_exact_metrics": demo_id_exact_summary,
            "per_frame_diff_metrics": frame_diff_summary,
            "timestamp": datetime.now().isoformat(),
            "args": vars(self.args),
        }

        # Save outputs
        with open(os.path.join(self.output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if self.args.save_completions:
            import pandas as pd
            with open(os.path.join(self.output_dir, "completions.json"), "w") as f:
                json.dump(results, f, indent=2)
            try:
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(self.output_dir, "completions.csv"), index=False)
            except Exception:
                pass

        print("\n" + "=" * 50)
        print("OVERLAY REGRESSION EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Model: {summary['model_path']}")
        print(f"Pairs evaluated: {summary['num_pairs']}")
        print(f"Overlay: {self.overlay_method} | Split: {self.args.dataset_split} | Seed: {self.args.seed}")
        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | Within tol: {tol_rate:.3f} | Sign Acc: {sign_acc:.3f}")
        print(f"Results saved to: {self.output_dir}")

        # Print per-demo_id statistics
        if demo_id_summary:
            print("\n" + "=" * 50)
            print("PER DEMO_ID ACCURACY STATISTICS")
            print("=" * 50)
            print(f"{'demo_id':<15} {'Samples':>8} {'MAE':>8} {'Tol Rate':>10} {'Sign Acc':>10}")
            print("-" * 50)

            # Sort by demo_id name for consistent display
            sorted_demo_ids = sorted(demo_id_summary.items(), key=lambda x: x[0])
            for demo_id, stats in sorted_demo_ids:
                print(f"{demo_id:<15} {stats['total_samples']:>8} {stats['mae']:>8.3f} "
                      f"{stats['within_tolerance_rate']:>10.3f} {stats['sign_accuracy']:>10.3f}")

            print("\n" + "=" * 50)
            print("TOP/BOTTOM 5 DEMO_IDs BY WITHIN-TOLERANCE RATE")
            print("=" * 50)

            # Sort by within_tolerance_rate
            sorted_by_tol = sorted(demo_id_summary.items(),
                                  key=lambda x: x[1]['within_tolerance_rate'],
                                  reverse=True)

            print("\nTop 5 (Best):")
            print(f"{'demo_id':<15} {'Samples':>8} {'MAE':>8} {'Tol Rate':>10} {'Sign Acc':>10}")
            print("-" * 50)
            for demo_id, stats in sorted_by_tol[:5]:
                print(f"{demo_id:<15} {stats['total_samples']:>8} {stats['mae']:>8.3f} "
                      f"{stats['within_tolerance_rate']:>10.3f} {stats['sign_accuracy']:>10.3f}")

            print("\nBottom 5 (Worst):")
            print(f"{'demo_id':<15} {'Samples':>8} {'MAE':>8} {'Tol Rate':>10} {'Sign Acc':>10}")
            print("-" * 50)
            for demo_id, stats in sorted_by_tol[-5:]:
                print(f"{demo_id:<15} {stats['total_samples']:>8} {stats['mae']:>8.3f} "
                      f"{stats['within_tolerance_rate']:>10.3f} {stats['sign_accuracy']:>10.3f}")

        # Print per-frame_diff statistics
        if frame_diff_summary:
            print("\n" + "=" * 50)
            print("PER FRAME_DIFF BUCKET ACCURACY STATISTICS")
            print("=" * 50)
            print(f"{'Frame Diff':>12} {'Samples':>8} {'MAE':>8} {'Tol Rate':>10} {'Sign Acc':>10}")
            print("-" * 50)

            # Sort by frame_diff for consistent display
            sorted_frame_diffs = sorted(frame_diff_summary.items(), key=lambda x: x[0])
            for frame_diff, stats in sorted_frame_diffs:
                print(f"{frame_diff:>12} {stats['total_samples']:>8} {stats['mae']:>8.3f} "
                      f"{stats['within_tolerance_rate']:>10.3f} {stats['sign_accuracy']:>10.3f}")

        # Generate visualizations
        if self.args.visualize and results:
            print("\n" + "=" * 50)
            print("GENERATING VISUALIZATIONS")
            print("=" * 50)

            # Plot error distribution
            plot_error_distribution(results, self.output_dir)

            # Visualize worst examples
            visualize_predictions(results, self.output_dir,
                                num_examples=self.args.num_visualize,
                                sort_by="worst")

            # Visualize best examples
            visualize_predictions(results, self.output_dir,
                                num_examples=self.args.num_visualize,
                                sort_by="best")

            # Visualize random examples
            visualize_predictions(results, self.output_dir,
                                num_examples=self.args.num_visualize,
                                sort_by="random")

            # Create detailed visualizations for a few worst cases
            print("\nCreating detailed visualizations for worst cases...")
            valid_results = [r for r in results if r['prediction'] is not None and r['abs_error'] is not None]
            worst_cases = sorted(valid_results, key=lambda x: x['abs_error'], reverse=True)[:5]
            for idx, result in enumerate(worst_cases):
                detailed_path = create_detailed_visualization(result, self.output_dir, idx)
                print(f"  Saved: {detailed_path}")

            print(f"\nAll visualizations saved to: {self.output_dir}")

        return summary


def parse_args() -> EvaluationArgs:
    p = argparse.ArgumentParser(description="Evaluate SFT VLM Overlay Regression model trained with sft_vlm_overlay_regression_dp.py")

    # Model arguments
    p.add_argument("--model_name_or_path", type=str, default=None, help="Path to trained model checkpoint")
    p.add_argument("--base_model_name_or_path", type=str,
                   default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Base model path")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--use_quantization", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true", default=True)

    # Generation arguments
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cuda")

    # Dataset arguments
    p.add_argument("--dataset_cache_file", type=str, required=True,
                   help="Path to cached dataset pickle file from training script")
    p.add_argument("--dataset_split", type=str, default="val", help="Dataset split: val | train")
    p.add_argument("--overlay_method", type=str, default="side_by_side", help="Overlay method used in training")
    p.add_argument("--num_samples", type=int, default=None, help="Random subset of samples to evaluate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic evaluation")

    # Metrics arguments
    p.add_argument("--tolerance", type=int, default=3, help="Absolute error tolerance for within_tolerance metric")
    p.add_argument("--save_completions", action="store_true", default=True)

    # Visualization arguments
    p.add_argument("--visualize", action="store_true", default=True, help="Generate visualization plots")
    p.add_argument("--no_visualize", dest="visualize", action="store_false", help="Skip visualization generation")
    p.add_argument("--num_visualize", type=int, default=10, help="Number of examples to visualize per category")

    args = p.parse_args()

    # Convert to dataclass
    return EvaluationArgs(**vars(args))


def main():
    eval_args = parse_args()
    evaluator = OverlayRegressionEvaluator(eval_args)
    evaluator.run()


if __name__ == "__main__":
    main()
