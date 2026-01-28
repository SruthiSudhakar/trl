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
#     "decord",
# ]
# ///

"""
VLM Overlay Regression Training (v2) - On-the-fly Video Frame Decoding

Trains a VLM to compare side-by-side robot frames and predict task completion progress.
Frames are decoded from MP4 videos on-the-fly during training (no pre-generated overlays).

Usage:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_v2.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove" \
    --output_dir "outputs/TEST_$(date +%Y%m%d_%H%M%S)" \
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
    --max_exact_per_demo 50
"""

import ast
import gc
import glob
import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback

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

from video_frame_utils import create_side_by_side, extract_frame, get_frame_source

# Set random seed for reproducibility
random.seed(42)

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

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

SYSTEM_PROMPT = "Compare robot task progress. Respond with a number: positive if right image shows more progress, negative if less."

USER_PROMPT_TEMPLATE = """Task: {task_token}
Which image shows more task progress? Respond with a number from -100 to 100."""


# ============================================================================
# Custom Data Collator: On-the-fly video frame decoding
# ============================================================================

@dataclass
class VideoOverlayCollator(DataCollatorForVisionLanguageModeling):
    """
    Extends the VLM data collator to decode video frames on-the-fly
    and create side-by-side overlay images.

    Expects each example dict to have:
        - video_path_1, frame_idx_1: source for left image
        - video_path_2, frame_idx_2: source for right image
        - messages: chat messages for SFTTrainer

    Injects the 'images' key with [overlay_pil_image] before
    delegating to the parent collator.
    """

    def _collate_language_modeling(self, examples):
        for example in examples:
            frame1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            frame2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
            example["images"] = [overlay]
        return super()._collate_language_modeling(examples)

    def _collate_prompt_completion(self, examples):
        for example in examples:
            frame1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            frame2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
            example["images"] = [overlay]
        return super()._collate_prompt_completion(examples)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_trajectories(job_dirs):
    """
    Parse eval_log.json files from job directories.

    Returns:
        success_data: List of dicts with video_path, trajectory_index, sf, demo_id
        unfiltered_failure_data: List of dicts (same format)
    """
    success_data = []
    unfiltered_failure_data = []

    for job_dir in job_dirs:
        job_name = Path(job_dir).name
        metadata_path = Path(job_dir) / "eval_log.json"
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found for {job_name}, skipping...")
            continue

        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)

        for key, value in all_metadata.items():
            if not key.startswith("train/sim_reward_trajectory_"):
                continue
            trajectory = ast.literal_eval(value)
            video_key = key.replace("train/sim_reward_trajectory_", "train/sim_video_")
            video_path = "/workspace/guided_diffusion_policy/" + all_metadata[video_key]
            demo_id = int(key.split("train/sim_reward_trajectory_")[-1].split("_")[0])

            if 1 in trajectory:
                success_data.append({
                    "video_path": video_path,
                    "trajectory_index": int(trajectory.index(1) / 2),
                    "sf": "success",
                    "demo_id": demo_id,
                })
            else:
                unfiltered_failure_data.append({
                    "video_path": video_path,
                    "trajectory_index": int(len(trajectory) / 2),
                    "sf": "fail",
                    "demo_id": demo_id,
                })

    return success_data, unfiltered_failure_data


def match_failures_to_successes(success_data, unfiltered_failure_data):
    """Match each failure trajectory with a corresponding success trajectory (same demo_id)."""
    success_by_demo = defaultdict(list)
    for sd in success_data:
        success_by_demo[sd["demo_id"]].append(sd)

    failure_data = []
    for fd in unfiltered_failure_data:
        demo_id = fd["demo_id"]
        if demo_id in success_by_demo:
            fd["success_video_path"] = success_by_demo[demo_id][0]["video_path"]
            fd["success_trajectory_index"] = success_by_demo[demo_id][0]["trajectory_index"]
            failure_data.append(fd)

    return failure_data


def balance_by_demo_id(success_data, failure_data, max_per_demo):
    """Ensure equal representation per demo_id between success and failure data."""
    success_by_demo = defaultdict(list)
    failure_by_demo = defaultdict(list)
    for sd in success_data:
        success_by_demo[sd["demo_id"]].append(sd)
    for fd in failure_data:
        failure_by_demo[fd["demo_id"]].append(fd)

    common_demo_ids = set(success_by_demo.keys()) & set(failure_by_demo.keys())
    logger.info(f"Found {len(common_demo_ids)} demo_ids present in both success and failure data")

    if len(common_demo_ids) == 0:
        raise ValueError("No common demo_ids found between success and failure data!")

    balanced_success = []
    balanced_failure = []
    for demo_id in sorted(common_demo_ids):
        count = min(
            len(success_by_demo[demo_id]),
            len(failure_by_demo[demo_id]),
            max_per_demo,
        )
        rng = random.Random(42 + demo_id)
        balanced_success.extend(rng.sample(success_by_demo[demo_id], count))
        balanced_failure.extend(rng.sample(failure_by_demo[demo_id], count))

    logger.info(f"After balanced sampling: {len(balanced_success)} success, {len(balanced_failure)} failure")
    return balanced_success, balanced_failure, success_by_demo, failure_by_demo


def compute_failure_filter_stats(success_by_demo, failure_by_demo, local_rank, world_size):
    """
    Compute per-frame mean pixel difference stats between success trajectory pairs.
    Used to filter failure frames that are too similar to success (i.e., haven't diverged yet).

    Returns:
        success_mean_diffs_at_idx: {demo_id: {frame_idx: [mean_diff_values]}}
    """
    success_mean_diffs_at_idx = {}
    all_demo_ids = sorted(list(success_by_demo.keys()))
    my_demo_ids = all_demo_ids[local_rank::world_size]

    for sk in tqdm(my_demo_ids, desc=f"Rank {local_rank} computing stats"):
        sv = success_by_demo[sk]
        success_mean_diffs_at_idx[sk] = {}
        for x in range(1, min(10, len(sv) - 1)):
            one_sd = sv[0]
            two_sd = sv[x]
            source1 = get_frame_source(one_sd["video_path"])
            source2 = get_frame_source(two_sd["video_path"])
            max_idx = min(one_sd["trajectory_index"], two_sd["trajectory_index"])
            for idx in range(max_idx):
                img1 = extract_frame(source1, idx)
                img2 = extract_frame(source2, idx)
                arr1 = np.array(img1)
                arr2 = np.array(img2)
                mean_diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))
                if idx not in success_mean_diffs_at_idx[sk]:
                    success_mean_diffs_at_idx[sk][idx] = []
                success_mean_diffs_at_idx[sk][idx].append(mean_diff)

    # Gather stats from all ranks
    if world_size > 1:
        import torch.distributed as dist
        logger.info(f"Rank {local_rank}: Gathering stats from all ranks...")
        all_stats = [None for _ in range(world_size)]
        dist.all_gather_object(all_stats, success_mean_diffs_at_idx)
        success_mean_diffs_at_idx = {}
        for rank_stats in all_stats:
            success_mean_diffs_at_idx.update(rank_stats)

    return success_mean_diffs_at_idx


def build_frame_pairs(
    success_data,
    failure_data,
    compare_intervals,
    train_sample_interval,
    success_mean_diffs_at_idx,
    job_name,
    local_rank,
    world_size,
):
    """
    Generate frame pair metadata (no images, just paths + indices).

    Returns list of dicts ready for HF Dataset.
    """
    NUM_FRAMES = 200
    combined_data = []

    # Split across ranks
    success_samples = success_data[local_rank::world_size]
    failure_samples = failure_data[local_rank::world_size]
    all_demos = success_samples + failure_samples
    logger.info(f"Rank {local_rank}: Processing {len(success_samples)} success + {len(failure_samples)} failure demos")

    for one_demo in tqdm(all_demos, desc=f"Rank {local_rank} building pairs"):
        source = get_frame_source(one_demo["video_path"])
        demo_dir_name = one_demo["video_path"][:-4].split("/")[-1]
        demo_id = demo_dir_name.split("_")[0]
        demo_id_exact = demo_dir_name

        # Find task token
        task_token = None
        for task_key, token in TASK_TOKENS.items():
            if task_key in job_name:
                task_token = token
                break
        if task_token is None:
            raise ValueError(f"No task token found for job name: {job_name}")

        user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)

        # === SUCCESS PAIRS: compare frames at different intervals ===
        if one_demo["sf"] == "success":
            for interval in compare_intervals:
                if NUM_FRAMES <= interval:
                    continue
                max_idx1 = one_demo["trajectory_index"] - interval - 1
                for idx1 in range(0, max_idx1 + 1, train_sample_interval):
                    idx2 = idx1 + interval
                    correct_answer = 32

                    # 50% swap to avoid position bias
                    v1, f1, v2, f2 = source, idx1, source, idx2
                    if random.random() < 0.5:
                        v1, f1, v2, f2 = v2, f2, v1, f1
                        correct_answer = -correct_answer

                    combined_data.append({
                        "video_path_1": v1,
                        "frame_idx_1": f1,
                        "video_path_2": v2,
                        "frame_idx_2": f2,
                        "correct_answer": correct_answer,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": str(correct_answer)},
                        ],
                        "demo_id": demo_id,
                        "demo_id_exact": demo_id_exact,
                        "demo_success": "success",
                        "job_name": job_name,
                        "task_token": task_token,
                    })

        # === FAILURE vs SUCCESS PAIRS: same frame index, different trajectory ===
        if one_demo["sf"] == "fail":
            success_source = get_frame_source(one_demo["success_video_path"])
            max_idx1 = one_demo["trajectory_index"] - 1
            beginning_of_failure = None

            for idx1 in range(0, max_idx1 + 1, train_sample_interval):
                # Compare failure frame vs success frame at same index
                try:
                    fail_img = extract_frame(source, idx1)
                    succ_img = extract_frame(success_source, idx1)
                    arr1 = np.array(fail_img)
                    arr2 = np.array(succ_img)
                    mean_diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))

                    # Filter: skip if frames are too similar (failure hasn't diverged yet)
                    should_skip = False
                    if len(success_mean_diffs_at_idx) > 0:
                        first_demo_key = list(success_mean_diffs_at_idx.keys())[0]
                        current_demo_id = type(first_demo_key)(one_demo["demo_id"])
                        if current_demo_id in success_mean_diffs_at_idx:
                            demo_stats = success_mean_diffs_at_idx[current_demo_id]
                            if len(demo_stats) > 0:
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
                    # Skip first 8 frames after failure starts
                    if idx1 <= beginning_of_failure + 8:
                        continue
                except Exception as e:
                    logger.warning(f"Failed to compare frames at idx {idx1}: {e}")
                    continue

                correct_answer = 32

                # 50% swap to avoid position bias
                v1, f1, v2, f2 = source, idx1, success_source, idx1
                if random.random() < 0.5:
                    v1, f1, v2, f2 = v2, f2, v1, f1
                    correct_answer = -correct_answer

                combined_data.append({
                    "video_path_1": v1,
                    "frame_idx_1": f1,
                    "video_path_2": v2,
                    "frame_idx_2": f2,
                    "correct_answer": correct_answer,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": str(correct_answer)},
                    ],
                    "demo_id": demo_id,
                    "demo_id_exact": demo_id_exact,
                    "demo_success": "failure",
                    "job_name": job_name,
                    "task_token": task_token,
                })

    # Gather data from all ranks
    if world_size > 1:
        import torch.distributed as dist
        logger.info(f"Rank {local_rank}: Gathering {len(combined_data)} pairs from all ranks...")
        all_data = [None for _ in range(world_size)]
        dist.all_gather_object(all_data, combined_data)
        combined_data = []
        for rank_data in all_data:
            combined_data.extend(rank_data)
        logger.info(f"Rank {local_rank}: Total {len(combined_data)} pairs after gathering")

    return combined_data


# ============================================================================
# Dataset Visualization
# ============================================================================

def visualize_dataset(combined_data, output_dir, split_name="train", num_examples=8):
    """
    Create statistics plots and sample overlay images for the dataset.

    Saves to {output_dir}/dataset_viz/:
      - dataset_statistics_{split}.png: answer distribution, demo type, per-demo counts, per-task counts
      - sample_overlays_{split}.png: grid of example side-by-side overlays with metadata
      - dataset_stats_{split}.json: machine-readable statistics

    Args:
        combined_data: List of metadata dicts (with video_path_1/2, frame_idx_1/2, etc.)
        output_dir: Directory to save outputs.
        split_name: Label for the split (e.g., "train", "val").
        num_examples: Number of sample overlay images to render.
    """
    if len(combined_data) == 0:
        logger.warning("No data to visualize")
        return

    viz_dir = Path(output_dir) / "dataset_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    correct_answers = [item.get("correct_answer", 0) for item in combined_data]
    demo_types = [item.get("demo_success", "unknown") for item in combined_data]
    demo_ids = [item.get("demo_id", "unknown") for item in combined_data]
    job_names = [item.get("job_name", "unknown") for item in combined_data]
    task_tokens = [item.get("task_token", "unknown") for item in combined_data]

    # ---- Figure 1: Dataset Statistics ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dataset Statistics ({split_name}, n={len(combined_data)})", fontsize=14, fontweight="bold")

    # 1a. Answer distribution
    ax = axes[0, 0]
    unique_answers = sorted(set(correct_answers))
    answer_counts = [correct_answers.count(a) for a in unique_answers]
    colors = ["#2ecc71" if a > 0 else "#e74c3c" if a < 0 else "#95a5a6" for a in unique_answers]
    ax.bar([str(a) for a in unique_answers], answer_counts, color=colors)
    ax.set_xlabel("Correct Answer")
    ax.set_ylabel("Count")
    ax.set_title("Answer Distribution")

    # 1b. Success vs Failure
    ax = axes[0, 1]
    type_counts = {t: demo_types.count(t) for t in set(demo_types)}
    colors_pie = ["#2ecc71" if "success" in t.lower() else "#e74c3c" for t in type_counts.keys()]
    ax.pie(type_counts.values(), labels=type_counts.keys(), autopct="%1.1f%%", colors=colors_pie)
    ax.set_title("Demo Type Distribution")

    # 1c. Samples per demo_id
    ax = axes[1, 0]
    demo_id_counts = defaultdict(int)
    for d in demo_ids:
        demo_id_counts[d] += 1
    sorted_ids = sorted(demo_id_counts.keys())
    ax.bar(range(len(sorted_ids)), [demo_id_counts[d] for d in sorted_ids], color="#3498db")
    ax.set_xlabel("Demo ID")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"Samples per Demo ({len(sorted_ids)} demos)")
    if len(sorted_ids) > 20:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(sorted_ids)))
        ax.set_xticklabels(sorted_ids, rotation=45)

    # 1d. Samples per task
    ax = axes[1, 1]
    task_counts = defaultdict(int)
    for t in task_tokens:
        task_counts[t] += 1
    sorted_tasks = sorted(task_counts.keys())
    ax.barh(range(len(sorted_tasks)), [task_counts[t] for t in sorted_tasks], color="#9b59b6")
    ax.set_yticks(range(len(sorted_tasks)))
    ax.set_yticklabels(sorted_tasks, fontsize=8)
    ax.set_xlabel("Sample Count")
    ax.set_title("Samples per Task")

    plt.tight_layout()
    stats_path = viz_dir / f"dataset_statistics_{split_name}.png"
    plt.savefig(stats_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved statistics plot to {stats_path}")

    # ---- Figure 2: Sample Overlay Images ----
    num_to_show = min(num_examples, len(combined_data))

    # Sample diverse examples (mix success/failure and positive/negative answers)
    indices_by_type = defaultdict(list)
    for i, item in enumerate(combined_data):
        key = (item.get("demo_success", "unknown"), item.get("correct_answer", 0) > 0)
        indices_by_type[key].append(i)

    sampled_indices = []
    rng = random.Random(42)
    for key, indices in indices_by_type.items():
        n_sample = max(1, num_to_show // len(indices_by_type))
        sampled_indices.extend(rng.sample(indices, min(n_sample, len(indices))))
    sampled_indices = sampled_indices[:num_to_show]
    if len(sampled_indices) == 0:
        sampled_indices = list(range(num_to_show))

    n_cols = min(3, num_to_show)
    n_rows = (num_to_show + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for plot_idx, data_idx in enumerate(sampled_indices):
        item = combined_data[data_idx]
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)

        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
            ax.imshow(overlay)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load:\n{e}", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "?")
        task = item.get("task_token", "?")
        title_color = "#2ecc71" if answer > 0 else "#e74c3c" if answer < 0 else "#333333"
        ax.set_title(f"Answer: {answer} | {demo_type} | {task}", fontsize=9, color=title_color, fontweight="bold")

    plt.suptitle(f"Sample Overlays ({split_name})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    samples_path = viz_dir / f"sample_overlays_{split_name}.png"
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample overlays to {samples_path}")

    # ---- Figure 3: Detailed view with original frame pairs ----
    num_detailed = min(4, len(sampled_indices))
    fig = plt.figure(figsize=(15, 4 * num_detailed))
    gs = gridspec.GridSpec(num_detailed, 3, width_ratios=[1, 1, 2], hspace=0.3, wspace=0.1)

    for row_idx in range(num_detailed):
        item = combined_data[sampled_indices[row_idx]]

        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
        except Exception:
            frame1 = frame2 = overlay = None

        ax1 = fig.add_subplot(gs[row_idx, 0])
        if frame1:
            ax1.imshow(frame1)
        ax1.axis("off")
        ax1.set_title("Left Frame", fontsize=9)

        ax2 = fig.add_subplot(gs[row_idx, 1])
        if frame2:
            ax2.imshow(frame2)
        ax2.axis("off")
        ax2.set_title("Right Frame", fontsize=9)

        ax3 = fig.add_subplot(gs[row_idx, 2])
        if overlay:
            ax3.imshow(overlay)
        ax3.axis("off")
        answer = item.get("correct_answer", "?")
        demo_type = item.get("demo_success", "?")
        task = item.get("task_token", "?")
        title_color = "#2ecc71" if answer > 0 else "#e74c3c" if answer < 0 else "#333333"
        ax3.set_title(
            f"Overlay | Answer: {answer} | {demo_type} | {task}",
            fontsize=10, color=title_color, fontweight="bold",
        )

    plt.suptitle(f"Detailed: Original Frames + Overlay ({split_name})", fontsize=12, fontweight="bold")
    detailed_path = viz_dir / f"detailed_examples_{split_name}.png"
    plt.savefig(detailed_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved detailed examples to {detailed_path}")

    # ---- Save JSON stats ----
    stats_summary = {
        "split": split_name,
        "total_samples": len(combined_data),
        "answer_distribution": {str(k): v for k, v in zip(unique_answers, answer_counts)},
        "demo_type_distribution": dict(type_counts),
        "num_unique_demo_ids": len(set(demo_ids)),
        "task_distribution": dict(task_counts),
    }
    stats_json_path = viz_dir / f"dataset_stats_{split_name}.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"Saved stats JSON to {stats_json_path}")

    return viz_dir


# ============================================================================
# Example IO Callback
# ============================================================================

class ExampleIOMonitorCallback(TrainerCallback):
    """Logs a few example inputs/outputs from eval set at each evaluation."""

    def __init__(self, eval_dataset, processor, max_examples=3):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.max_examples = max_examples

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return
        if self.processor is None:
            return

        model = kwargs.get("model")
        if model is None:
            return

        rng = random.Random(42 + int(state.global_step or 0))
        indices = list(range(len(self.eval_dataset)))
        rng.shuffle(indices)
        indices = indices[: self.max_examples]

        texts = []
        images_batch = []
        targets = []

        for i in indices:
            try:
                ex = self.eval_dataset[i]
                # Build overlay on-the-fly
                frame1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
                frame2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])
                overlay = create_side_by_side(frame1, frame2)

                # Extract user text and target from messages
                user_text = ""
                target_text = None
                for m in ex.get("messages", []):
                    if m.get("role") == "user":
                        user_text = m.get("content", "")
                    elif m.get("role") == "assistant":
                        target_text = m.get("content", None)

                mm_messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": overlay},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ]

                try:
                    import qwen_vl_utils
                    image_input, _ = qwen_vl_utils.process_vision_info(mm_messages)
                except (ImportError, Exception):
                    image_input = [overlay]

                text = self.processor.apply_chat_template(
                    mm_messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
                images_batch.append(image_input)
                targets.append(target_text)
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
                    **inputs, max_new_tokens=64, temperature=0.1, top_p=0.95, do_sample=True
                )

            decoded = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            logger.info("\n===== Example IO (eval) =====")
            for target, pred in zip(targets, decoded):
                logger.info(json.dumps({"target": target, "prediction": pred}))
            logger.info("===== End Example IO =====\n")
        except Exception as e:
            logger.warning(f"Failed during example IO logging: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    @dataclass
    class OverlayArguments:
        split: str = "train"
        include_successes: bool = True
        include_failures: bool = True
        train_sample_interval: int = 5
        compare_interval: str = "16"
        train_val_split_index: int = 5
        base_dataset_path: str = ""
        max_exact_per_demo: int = 50
        debug_samples: int = -1

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, OverlayArguments))
    script_args, training_args, model_args, overlay_args = parser.parse_args_and_config()
    training_args.max_length = None
    # Ensure metadata columns reach the custom collator
    training_args.remove_unused_columns = False

    compare_intervals = [int(x.strip()) for x in overlay_args.compare_interval.split(",")]

    logger.info("=" * 80)
    logger.info("VLM Overlay Regression Training v2 - On-the-fly Video Decoding")
    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"LR: {training_args.learning_rate}")
    logger.info(f"Split: {overlay_args.split}")
    logger.info(f"Compare intervals: {compare_intervals}")
    logger.info(f"Sample interval: {overlay_args.train_sample_interval}")
    logger.info(f"Base dataset path: {overlay_args.base_dataset_path}")

    # ============================
    # Distributed setup
    # ============================
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = 0
            world_size = 1
    except (ImportError, RuntimeError):
        local_rank = 0
        world_size = 1
    logger.info(f"Rank {local_rank} of {world_size}")

    # ============================
    # Load trajectories
    # ============================
    all_dirs = sorted(glob.glob(f"{overlay_args.base_dataset_path}/*"))
    job_dirs = [d for d in all_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_log.json"))]

    split = overlay_args.split
    if split == "val":
        job_dirs = job_dirs[-overlay_args.train_val_split_index :]
    elif split == "train":
        job_dirs = job_dirs[: -overlay_args.train_val_split_index]
    else:
        job_dirs = job_dirs[: int(split)]

    logger.info(f"Found {len(job_dirs)} job directories for {split} split")

    success_data, unfiltered_failure_data = load_trajectories(job_dirs)
    logger.info(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

    failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
    logger.info(f"Matched {len(failure_data)} failures to success trajectories")

    # ============================
    # Balanced sampling
    # ============================
    success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(
        success_data, failure_data, overlay_args.max_exact_per_demo
    )

    # Debug mode: subsample
    if overlay_args.debug_samples > -1:
        success_data = random.sample(success_data, min(len(success_data), overlay_args.debug_samples))
        failure_data = random.sample(failure_data, min(len(failure_data), overlay_args.debug_samples))

    # ============================
    # Compute failure filter stats
    # ============================
    stats_cache_file = Path(overlay_args.base_dataset_path) / "failure_filter_stats.json"
    if stats_cache_file.exists():
        logger.info(f"Loading cached failure filter stats from {stats_cache_file}")
        with open(stats_cache_file, "r") as f:
            cached = json.load(f)
            success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]
    else:
        logger.info("Computing failure filter stats...")
        success_mean_diffs_at_idx = compute_failure_filter_stats(
            success_by_demo, failure_by_demo, local_rank, world_size
        )
        if local_rank == 0:
            with open(stats_cache_file, "w") as f:
                json.dump({"success_mean_diffs_at_idx": success_mean_diffs_at_idx}, f)
            logger.info(f"Saved failure filter stats to {stats_cache_file}")

    # ============================
    # Build frame pair metadata
    # ============================
    job_name = Path(job_dirs[0]).name if job_dirs else ""
    combined_data = build_frame_pairs(
        success_data=success_data,
        failure_data=failure_data,
        compare_intervals=compare_intervals,
        train_sample_interval=overlay_args.train_sample_interval,
        success_mean_diffs_at_idx=success_mean_diffs_at_idx,
        job_name=job_name,
        local_rank=local_rank,
        world_size=world_size,
    )
    logger.info(f"Built {len(combined_data)} frame pairs")

    # ============================
    # Visualize dataset (rank 0 only, before sharding)
    # ============================
    if local_rank == 0 and len(combined_data) > 0:
        try:
            visualize_dataset(combined_data, training_args.output_dir, split_name=split, num_examples=8)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    # ============================
    # Shuffle + shard
    # ============================
    random.seed(42)
    random.shuffle(combined_data)

    if world_size > 1:
        total = len(combined_data)
        per_rank = total // world_size
        start = local_rank * per_rank
        end = start + per_rank if local_rank < world_size - 1 else total
        combined_data = combined_data[start:end]
        logger.info(f"Rank {local_rank}: Shard [{start}:{end}] = {len(combined_data)} pairs")

    # ============================
    # Create HF Dataset (metadata only, no images)
    # ============================
    logger.info(f"Creating HuggingFace dataset from {len(combined_data)} samples...")
    dataset = Dataset.from_list(combined_data)
    del combined_data
    gc.collect()

    # Train/test split
    test_size = max(2, len(dataset) // 20)
    dataset = dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    # Synchronize before model loading
    if world_size > 1:
        try:
            dist.barrier()
        except Exception:
            pass

    # ============================
    # Model + Processor
    # ============================
    logger.info("Loading model and tokenizer...")
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

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
    except Exception as e:
        logger.warning(f"Failed to load AutoProcessor: {e}")

    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
    if hasattr(model, "num_parameters"):
        logger.info(f"Parameters: {model.num_parameters() / 1e9:.2f}B")

    # ============================
    # Training
    # ============================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Replace the default collator with our video-aware collator
    if hasattr(trainer, "data_collator") and isinstance(
        trainer.data_collator, DataCollatorForVisionLanguageModeling
    ):
        old_collator = trainer.data_collator
        trainer.data_collator = VideoOverlayCollator(
            processor=old_collator.processor,
            max_length=old_collator.max_length,
            completion_only_loss=old_collator.completion_only_loss,
            pad_to_multiple_of=old_collator.pad_to_multiple_of,
        )
        logger.info("Replaced default collator with VideoOverlayCollator")

    # Register eval callback
    if processor is not None and eval_dataset is not None:
        try:
            trainer.add_callback(
                ExampleIOMonitorCallback(eval_dataset=eval_dataset, processor=processor, max_examples=3)
            )
        except Exception as e:
            logger.warning(f"Could not add ExampleIOMonitorCallback: {e}")

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
