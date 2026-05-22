#
# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPMicrowaveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToSink,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPSinkToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCoffeeServeMug,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCloseDrawer,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToCab" \
    --output_dir "outputs/feb3/PnPAll_successonly_$(date +%Y%m%d_%H%M%S)" \
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
    --compare_interval 4,8,12,16 \
    --include_successes true \
    --include_failures false

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_v2.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --base_dataset_path "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPMicrowaveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToSink,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPSinkToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCoffeeServeMug,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToCab" \
    --output_dir "outputs/feb5/PnPAll_$(date +%Y%m%d_%H%M%S)" \
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
    --compare_interval 4,8,12,16 \
    --upweight_robocasa 10

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --gpu_ids=0 \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/myscripts/sft_vlm_overlay_regression_v2.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --base_dataset_path "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01" \
    --output_dir "outputs/test_$(date +%Y%m%d_%H%M%S)" \
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
    --compare_interval 4,8,12,16 \
    --just_visualize true

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
import pdb
import sys
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
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from video_frame_utils import create_side_by_side, extract_frame, get_frame_source, find_job_dirs, read_s3_json, s3, parse_s3_uri
try:
    from trl import ScriptArguments

except ImportError:
    print('COULD NOT IMPORT SCRIPT ARGUMENTS')
    pass  # Not needed for eval, only for training
try:
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
except ImportError:
    print('COULD NOT IMPORT DataCollatorForVisionLanguageModeling')
    DataCollatorForVisionLanguageModeling = object  # dummy base class for eval

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
    "PnPRedLegoToBrownBowl": "[PNP_RED_LEGO_TO_BROWN_BOWL]",
    "CoffeeServeMug": "[COFFEE_SERVE_MUG]",
    "PnPCloseDrawer": "[CLOSE_DRAWER]",
    "CoffeeSetupMug": "[COFFEE_SETUP_MUG]",
    "CoffeePressButton": "[COFFEE_PRESS_BUTTON]",
    "CloseDrawer": "[CLOSE_DRAWER]",
    "OpenDrawer": "[OPEN_DRAWER]",
    "CloseSingleDoor": "[CLOSE_SINGLE_DOOR]",
    "CloseDoubleDoor": "[CLOSE_DOUBLE_DOOR]",
    "OpenDoubleDoor": "[OPEN_DOUBLE_DOOR]",
    "OpenSingleDoor": "[OPEN_SINGLE_DOOR]",
    "TurnSinkSpout": "[TURN_SINK_SPOUT]",
    # Realworld dataset tasks
    "PutKiwiInCenterOfTable": "[PUT_KIWI_IN_CENTER_OF_TABLE]",
    "PushCoasterToMug": "[PUSH_COASTER_TO_MUG]",
    "TurnMugRightsideUp": "[TURN_MUG_RIGHTSIDE_UP]",
    "BimanualBikeRotorInstall": "[BIKE_ROTOR_INSTALL]",
    "BimanualClearKitchenCounter": "[CLEAR_KITCHEN_COUNTER]",
    "BimanualSetUpBreakfastTable": "[SETUP_BREAKFAST_TABLE]",
    "CleanLitterBox": "[CLEAN_LITTERBOX]",
    "CutAppleIntoSlices": "[CUT_APPLE]",
    "UprightBottle": "[UPRIGHT_BOTTLE]",
    "BagPlate": "[BAG_PLATE]",
    "PushBowl": "[PUSH_BOWL]",
    "Stacking": "[STACKING]",
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
        job_name = job_dir.rstrip("/").split("/")[-1]
        if job_dir.startswith("s3://"):
            metadata_uri = job_dir.rstrip("/") + "/eval_log.json"
            try:
                all_metadata = read_s3_json(metadata_uri)
            except Exception:
                logger.warning(f"Metadata file not found for {job_name}, skipping...")
                continue
        else:
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
            video_rel = all_metadata[video_key]
            if job_name + "/" in video_rel:
                rel_after_job = video_rel.split(job_name + "/", 1)[1]
                video_path = job_dir.rstrip("/") + "/" + rel_after_job
            elif "trainmedia/" in video_rel:
                trainmedia_part = "trainmedia/" + video_rel.split("trainmedia/", 1)[1]
                video_path = job_dir.rstrip("/") + "/" + trainmedia_part
            else:
                video_path = "/workspace/guided_diffusion_policy/" + video_rel
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


def prioritized_sample(items, count, rng, priority_substr="robocasa/datasets"):
    """
    Sample `count` items from `items`, prioritizing those containing `priority_substr` in their video path.
    """
    prio = []
    others = []
    for x in items:
        if priority_substr in x.get("video_path_1", ""):
            prio.append(x)
        else:
            others.append(x)

    rng.shuffle(prio)
    rng.shuffle(others)

    if len(prio) >= count:
        return prio[:count]
    else:
        needed = count - len(prio)
        return prio + others[:needed]


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
    task_path,
):
    """
    Generate frame pair metadata (no images, just paths + indices).

    Returns list of dicts ready for HF Dataset.
    """
    NUM_FRAMES = 100000000
    combined_data = []

    # Split across ranks
    success_samples = success_data[local_rank::world_size]
    failure_samples = failure_data[local_rank::world_size]
    all_demos = success_samples + failure_samples
    logger.info(f"Rank {local_rank}: Processing {len(success_samples)} success + {len(failure_samples)} failure demos")

    for one_demo in tqdm(all_demos, desc=f"Rank {local_rank} building pairs"):
        source = get_frame_source(one_demo["video_path"])
        demo_dir_name = one_demo["video_path"][:-4].split("/")[-1]
        demo_id = demo_dir_name.split("_")[0] if demo_dir_name.split("_")[0]!= 'demo' else demo_dir_name.split("_")[1]
        demo_id_exact = demo_dir_name

        # Find task token
        task_token = None
        for task_key, token in TASK_TOKENS.items():
            if task_key in task_path:
                task_token = token
                break
        if task_token is None:
            raise ValueError(f"No task token found for job name: {task_path}")

        user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)

        # === SUCCESS PAIRS: compare frames at different intervals ===
        if one_demo["sf"] == "success":
            for interval in compare_intervals:
                if NUM_FRAMES <= interval:
                    continue
                max_idx1 = one_demo["trajectory_index"] - interval - 1
                offset = random.randint(0, train_sample_interval - 1)
                for idx1 in range(offset, max_idx1 + 1, train_sample_interval):
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

            offset = random.randint(0, train_sample_interval - 1)
            for idx1 in range(offset, max_idx1 + 1, train_sample_interval):
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
                    if idx1 <= beginning_of_failure + 4:
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

def visualize_dataset(combined_data, output_dir, split_name="train", num_examples=20, max_pixels=None):
    """
    Create statistics plots and sample overlay images for the dataset.

    Saves to {output_dir}/dataset_viz/:
      - dataset_statistics_{split}.png: answer distribution, demo type, per-demo counts, per-task counts
      - sample_overlays_{split}.png: grid of example side-by-side overlays with metadata
      - vlm_view_{split}.png: same samples rendered at the post-`smart_resize` resolution the VLM actually sees (only if `max_pixels` is provided).
      - dataset_stats_{split}.json: machine-readable statistics

    Args:
        combined_data: List of metadata dicts (with video_path_1/2, frame_idx_1/2, etc.)
        output_dir: Directory to save outputs.
        split_name: Label for the split (e.g., "train", "val").
        num_examples: Number of sample overlay images to render.
        max_pixels: Qwen2.5-VL processor pixel budget per image (H*W). If set,
            an additional `vlm_view_{split}.png` is written showing each sampled
            frame after `smart_resize` (28-aligned, aspect-ratio-preserving).
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

    # ---- Figure 4: What the VLM actually sees (post smart_resize) ----
    # Mirror Qwen2.5-VL's image processor: smart_resize snaps H,W to multiples
    # of 28 (patch_size*merge_size) while preserving aspect ratio, bounded by
    # max_pixels. The result is the exact pixel buffer the model is fed.
    if max_pixels is not None:
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        except Exception as e:
            logger.warning(f"Could not import smart_resize for VLM view: {e}")
        else:
            n_vlm = min(num_examples, len(sampled_indices))
            n_cols_v = 2  # frame1 | frame2
            fig = plt.figure(figsize=(6 * n_cols_v, 3 * n_vlm))
            for row_idx in range(n_vlm):
                item = combined_data[sampled_indices[row_idx]]
                try:
                    f1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                    f2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                    h1_new, w1_new = smart_resize(f1.height, f1.width, max_pixels=max_pixels)
                    h2_new, w2_new = smart_resize(f2.height, f2.width, max_pixels=max_pixels)
                    v1 = f1.resize((w1_new, h1_new), Image.BICUBIC)
                    v2 = f2.resize((w2_new, h2_new), Image.BICUBIC)
                except Exception as e:
                    v1 = v2 = None
                    h1_new = w1_new = h2_new = w2_new = 0
                    orig1 = orig2 = ""
                    err = str(e)
                else:
                    orig1 = f"{f1.width}x{f1.height}"
                    orig2 = f"{f2.width}x{f2.height}"
                    err = ""

                ax1 = fig.add_subplot(n_vlm, 2, row_idx * 2 + 1)
                if v1 is not None:
                    ax1.imshow(v1, interpolation="nearest")
                elif err:
                    ax1.text(0.5, 0.5, err, ha="center", va="center", transform=ax1.transAxes)
                ax1.axis("off")
                ax1.set_title(f"frame_1: {orig1} -> {w1_new}x{h1_new}", fontsize=9)

                ax2 = fig.add_subplot(n_vlm, 2, row_idx * 2 + 2)
                if v2 is not None:
                    ax2.imshow(v2, interpolation="nearest")
                ax2.axis("off")
                answer = item.get("correct_answer", "?")
                ax2.set_title(
                    f"frame_2: {orig2} -> {w2_new}x{h2_new} | ans={answer}",
                    fontsize=9,
                )

            plt.suptitle(
                f"VLM view ({split_name}) - smart_resize @ max_pixels={max_pixels}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            vlm_path = viz_dir / f"vlm_view_{split_name}.png"
            plt.savefig(vlm_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved VLM view to {vlm_path}")

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


def visualize_dataset_diagnostic(combined_data, output_dir, split_name="train", num_examples_per_demo=20):
    """
    Show example datapoints from each demo with metadata labels.

    Saves to {output_dir}/dataset_viz_diagnostic/:
      - One PNG per demo showing sample frames with success/failure labels and frame indices

    Args:
        combined_data: List of metadata dicts (with video_path_1/2, frame_idx_1/2, etc.)
        output_dir: Directory to save outputs.
        split_name: Label for the split (e.g., "train", "val").
        num_examples_per_demo: Number of example images to show per demo.
    """
    if len(combined_data) == 0:
        logger.warning("No data to visualize")
        return

    viz_dir = Path(output_dir) / "dataset_viz_diagnostic"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Helper to extract demo name from video path
    def get_demo_name_from_path(video_path):
        """Extract demo name (e.g., '10_5_abc123') from video path."""
        return video_path[:-4].split("/")[-1]  # Remove .mp4 and get filename

    # Group data by demo_id_exact (unique video/trajectory identifier)
    demos_by_id = defaultdict(list)
    for i, item in enumerate(combined_data):
        demo_key = item.get("demo_id_exact", item.get("demo_id", "unknown"))
        demos_by_id[demo_key].append((i, item))

    logger.info(f"Found {len(demos_by_id)} unique demos to visualize")

    for demo_key, demo_items in tqdm(demos_by_id.items(), desc="Generating per-demo visualizations"):
        # Sort by frame index to show progression through the video
        demo_items_sorted = sorted(demo_items, key=lambda x: x[1]["frame_idx_1"])

        # Sample evenly spaced examples if we have more than num_examples_per_demo
        if len(demo_items_sorted) > num_examples_per_demo:
            indices = np.linspace(0, len(demo_items_sorted) - 1, num_examples_per_demo, dtype=int)
            demo_items_sorted = [demo_items_sorted[i] for i in indices]

        n = len(demo_items_sorted)
        if n == 0:
            continue

        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))

        # Get metadata from first item for the title
        first_item = demo_items_sorted[0][1]
        demo_success = first_item.get("demo_success", "unknown")
        task_token = first_item.get("task_token", "unknown")
        success_label = "SUCCESS" if demo_success == "success" else "FAILURE"
        title_color = "#2ecc71" if demo_success == "success" else "#e74c3c"

        # For failure demos, find the success video being compared against
        compared_against = ""
        if demo_success == "failure":
            # The demo_key is from the failure video. Find which video_path is the success one.
            video_path_1 = first_item.get("video_path_1", "")
            video_path_2 = first_item.get("video_path_2", "")
            demo_name_1 = get_demo_name_from_path(video_path_1)
            demo_name_2 = get_demo_name_from_path(video_path_2)

            # The one that doesn't match demo_key is the success video
            if demo_name_1 == demo_key:
                success_demo_name = demo_name_2
            else:
                success_demo_name = demo_name_1
            compared_against = f"\nCompared against SUCCESS: {success_demo_name}"

        fig.suptitle(
            f"Demo: {demo_key} | {success_label} | Task: {task_token}{compared_against}",
            fontsize=14, fontweight="bold", color=title_color, y=1.02
        )

        for plot_idx, (data_idx, item) in enumerate(demo_items_sorted):
            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)

            try:
                frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                overlay = create_side_by_side(frame1, frame2)
                ax.imshow(overlay)
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed: {e}", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")

            # Label with frame indices and which side is which
            frame_idx_1 = item["frame_idx_1"]
            frame_idx_2 = item["frame_idx_2"]
            correct_answer = item.get("correct_answer", "?")
            answer_color = "#2ecc71" if correct_answer > 0 else "#e74c3c"

            # For failure demos, show which side is failure vs success
            if demo_success == "failure":
                video_path_1 = item.get("video_path_1", "")
                demo_name_1 = get_demo_name_from_path(video_path_1)
                if demo_name_1 == demo_key:
                    left_label, right_label = "FAIL", "SUCC"
                else:
                    left_label, right_label = "SUCC", "FAIL"
                info_text = (
                    f"Frame {frame_idx_1} ({left_label}) vs {frame_idx_2} ({right_label})\n"
                    f"Answer: {correct_answer}"
                )
            else:
                info_text = (
                    f"Frame {frame_idx_1} vs {frame_idx_2}\n"
                    f"Answer: {correct_answer}"
                )
            ax.set_title(info_text, fontsize=10, color=answer_color, fontweight="bold")

        plt.tight_layout()
        safe_demo_key = str(demo_key).replace("/", "_").replace("\\", "_")
        plt.savefig(viz_dir / f"demo_{safe_demo_key}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved per-demo visualizations to {viz_dir}")
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
        compare_interval: str = "4,8,12,16"
        train_val_split_index: int = 5
        base_dataset_path: str = ""
        max_exact_per_demo: int = 50
        debug_samples: int = -1
        just_visualize: bool = False
        balance_data: bool = False
        upweight_robocasa: int = 10
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
    logger.info(f"Just visualize: {overlay_args.just_visualize}")
    logger.info(f"Balance data: {overlay_args.balance_data}")
    logger.info(f"Include successes: {overlay_args.include_successes}")
    logger.info(f"Include failures: {overlay_args.include_failures}")
    logger.info(f"Upweight robocasa: {overlay_args.upweight_robocasa}")

    if not overlay_args.include_successes and not overlay_args.include_failures:
        raise ValueError("At least one of --include_successes or --include_failures must be True")

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
    # Parse multiple task dataset paths (comma-separated)
    # ============================
    task_paths = [p.strip() for p in overlay_args.base_dataset_path.split(",") if p.strip()]
    logger.info(f"Processing {len(task_paths)} task dataset path(s): {task_paths}")

    split = overlay_args.split
    combined_data = []

    for task_idx, task_path in enumerate(task_paths):
        logger.info("=" * 60)
        logger.info(f"[Task {task_idx + 1}/{len(task_paths)}] Processing: {task_path}")
        logger.info("=" * 60)

        # ============================
        # Load trajectories for this task
        # ============================
        job_dirs = find_job_dirs(task_path)
        if split == "val":
            job_dirs = job_dirs[-overlay_args.train_val_split_index :]
        elif split == "train":
            if 'robocasa/datasets' in task_path or overlay_args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[: -overlay_args.train_val_split_index]
        else:
            job_dirs = job_dirs[: int(split)]

        logger.info(f"Found {len(job_dirs)} job directories for {split} split in {task_path}")

        if len(job_dirs) == 0:
            logger.warning(f"No job directories found in {task_path}, skipping...")
            continue

        success_data, unfiltered_failure_data = load_trajectories(job_dirs)
        logger.info(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

        if overlay_args.include_failures:
            failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
            logger.info(f"Matched {len(failure_data)} failures to success trajectories")
        else:
            failure_data = []
            logger.info("Skipping failure data (--include_failures is False)")

        # ============================
        # Balanced sampling for this task
        # ============================
        if overlay_args.include_successes and overlay_args.include_failures and success_data and failure_data:
            success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(
                success_data, failure_data, overlay_args.max_exact_per_demo
            )
        else:
            # No cross-type balancing, but still cap per demo
            success_by_demo = defaultdict(list)
            for sd in success_data:
                success_by_demo[sd["demo_id"]].append(sd)
            failure_by_demo = defaultdict(list)
            for fd in failure_data:
                failure_by_demo[fd["demo_id"]].append(fd)

            rng = random.Random(42)
            capped_success = []
            for demo_id in sorted(success_by_demo.keys()):
                items = success_by_demo[demo_id]
                if len(items) > overlay_args.max_exact_per_demo:
                    items = rng.sample(items, overlay_args.max_exact_per_demo)
                    success_by_demo[demo_id] = items
                capped_success.extend(items)
            capped_failure = []
            for demo_id in sorted(failure_by_demo.keys()):
                items = failure_by_demo[demo_id]
                if len(items) > overlay_args.max_exact_per_demo:
                    items = rng.sample(items, overlay_args.max_exact_per_demo)
                    failure_by_demo[demo_id] = items
                capped_failure.extend(items)

            logger.info(f"After per-demo cap ({overlay_args.max_exact_per_demo}): "
                        f"{len(success_data)} -> {len(capped_success)} success, "
                        f"{len(failure_data)} -> {len(capped_failure)} failure")
            success_data = capped_success
            failure_data = capped_failure
        # Debug mode: subsample
        if overlay_args.debug_samples > -1:
            success_data = random.sample(success_data, min(len(success_data), overlay_args.debug_samples))
            failure_data = random.sample(failure_data, min(len(failure_data), overlay_args.debug_samples))

        # ============================
        # Compute failure filter stats for this task
        # ============================
        if overlay_args.include_failures and failure_data:
            success_mean_diffs_at_idx = None
            if task_path.startswith("s3://"):
                stats_s3_uri = task_path.rstrip("/") + "/failure_filter_stats.json"
                try:
                    cached = read_s3_json(stats_s3_uri)
                    success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]
                    logger.info(f"Loaded cached failure filter stats from {stats_s3_uri}")
                except Exception:
                    logger.info(f"No cached stats found at {stats_s3_uri}")
            else:
                stats_cache_file = Path(task_path) / "failure_filter_stats.json"
                if stats_cache_file.exists():
                    logger.info(f"Loading cached failure filter stats from {stats_cache_file}")
                    with open(stats_cache_file, "r") as f:
                        cached = json.load(f)
                        success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]

            if success_mean_diffs_at_idx is None:
                logger.info("Computing failure filter stats...")
                success_mean_diffs_at_idx = compute_failure_filter_stats(
                    success_by_demo, failure_by_demo, local_rank, world_size
                )
                if local_rank == 0:
                    if task_path.startswith("s3://"):
                        stats_s3_uri = task_path.rstrip("/") + "/failure_filter_stats.json"
                        body = json.dumps({"success_mean_diffs_at_idx": success_mean_diffs_at_idx})
                        bucket, prefix = parse_s3_uri(stats_s3_uri)
                        key = prefix.rstrip("/")
                        s3.put_object(Bucket=bucket, Key=key, Body=body.encode())
                        logger.info(f"Saved failure filter stats to {stats_s3_uri}")
                    else:
                        stats_cache_file = Path(task_path) / "failure_filter_stats.json"
                        with open(stats_cache_file, "w") as f:
                            json.dump({"success_mean_diffs_at_idx": success_mean_diffs_at_idx}, f)
                        logger.info(f"Saved failure filter stats to {stats_cache_file}")
        else:
            success_mean_diffs_at_idx = {}
            logger.info("Skipping failure filter stats (no failure data)")

        # ============================
        # Build frame pair metadata for this task
        # ============================
        job_name = Path(job_dirs[0]).name if job_dirs else ""
        build_success = success_data if overlay_args.include_successes else []
        build_failure = failure_data if overlay_args.include_failures else []
        task_combined_data = build_frame_pairs(
            success_data=build_success,
            failure_data=build_failure,
            compare_intervals=compare_intervals,
            train_sample_interval=overlay_args.train_sample_interval,
            success_mean_diffs_at_idx=success_mean_diffs_at_idx,
            job_name=job_name,
            local_rank=local_rank,
            world_size=world_size,
            task_path=task_path,
        )
        logger.info(f"Built {len(task_combined_data)} frame pairs for task: {job_name}")
        # ============================
        # Upweight robocasa samples by tripling them
        # ============================
        if "robocasa/datasets" in task_path:
            logger.info(f"Upweighting robocasa samples: {len(task_combined_data)} -> {len(task_combined_data) * overlay_args.upweight_robocasa}")
            # Add 2 more copies (original + 2 copies = 3x)
            combined_data.extend(task_combined_data * overlay_args.upweight_robocasa)
            logger.info(f"Total samples after upweighting: {len(combined_data)}")
        else:
            combined_data.extend(task_combined_data)

    logger.info(f"Total frame pairs across all tasks: {len(combined_data)}")

    if overlay_args.balance_data:
        # ============================
        # Balance success vs failure_vs_success pair types
        # ============================
        if len(combined_data) > 0:
            success_pairs = [item for item in combined_data if item["demo_success"] == "success"]
            failure_pairs = [item for item in combined_data if item["demo_success"] == "failure"]
            logger.info(f"Before pair-type balancing: {len(success_pairs)} success pairs, {len(failure_pairs)} failure_vs_success pairs")

            if len(success_pairs) > 0 and len(failure_pairs) > 0:
                if len(success_pairs) > len(failure_pairs):
                    rng = random.Random(42)
                    success_pairs = prioritized_sample(success_pairs, len(failure_pairs), rng)
                    logger.info(f"Downsampled success pairs: {len(success_pairs)} to match failure_vs_success count")
                elif len(failure_pairs) > len(success_pairs):
                    rng = random.Random(42)
                    failure_pairs = prioritized_sample(failure_pairs, len(success_pairs), rng)
                    logger.info(f"Downsampled failure_vs_success pairs: {len(failure_pairs)} to match success count")
                combined_data = success_pairs + failure_pairs
                logger.info(f"After pair-type balancing: {len(combined_data)} total pairs")

    # ============================
    # Balance samples across tasks
    # ============================
    if len(task_paths) > 1 and len(combined_data) > 0:
        samples_by_task = defaultdict(list)
        for item in combined_data:
            samples_by_task[item["task_token"]].append(item)

        task_counts = {t: len(items) for t, items in samples_by_task.items()}
        mean_count = int(sum(task_counts.values()) / len(task_counts))
        max_count = int(mean_count)

        logger.info(f"Per-task sample counts before balancing: {task_counts}")
        logger.info(f"Mean: {mean_count}, Cap (mean): {max_count}")

        balanced_data = []
        for task_token, items in samples_by_task.items():
            if len(items) > max_count:
                logger.info(f"Downsampling {task_token}: {len(items)} -> {max_count}")
                rng = random.Random(42)
                items = prioritized_sample(items, max_count, rng)
            balanced_data.extend(items)

        logger.info(f"After cross-task balancing: {len(combined_data)} -> {len(balanced_data)}")
        combined_data = balanced_data

    # ============================
    # Shuffle + shard
    # ============================
    random.seed(42)
    random.shuffle(combined_data)

    # ============================
    # Visualize dataset (rank 0 only, before sharding)
    # ============================
    if local_rank == 0 and len(combined_data) > 0:
        try:
            # visualize_dataset_diagnostic(combined_data, training_args.output_dir, split_name=split, num_examples_per_demo=6)
            visualize_dataset(combined_data, training_args.output_dir, split_name=split, num_examples=20)
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

    if overlay_args.just_visualize:
        sys.exit(0)

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
