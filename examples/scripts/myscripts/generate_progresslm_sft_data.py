#!/usr/bin/env python3
"""
Generate ProgressLM SFT training data (nothink / direct-prediction format)
from your existing RoboCasa trajectory dataset.

Outputs:
  1. A JSONL file in LLaMA-Factory ShareGPT format (ready for SFT)
  2. Extracted frames saved as PNGs under --image_output_dir
  3. A dataset_info.json snippet you can paste into LLaMA-Factory

Usage:

EXPERT_DATASETS=(
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26"
  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27"
)
python3 examples/scripts/myscripts/generate_progresslm_sft_data.py \
    --base_dataset_path /workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01 \
    --image_output_dir /workspace/hf_trl/trl/progresLM_dataset \
    --output_jsonl workspace/hf_trl/trl/progresLM_dataset/robocasa_progresslm_sft.jsonl \
    --num_demo_frames 5 \
    --samples_per_trajectory 10 \
    --train_val_split_index 0 \
    --split train

python3 examples/scripts/myscripts/generate_progresslm_sft_data.py \
    --base_dataset_path "\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27" \
    --image_output_dir /workspace/hf_trl/trl/progresLM_dataset \
    --output_jsonl /workspace/hf_trl/trl/progresLM_dataset/robocasa_progresslm_sft.jsonl \
    --num_demo_frames 5 \
    --samples_per_trajectory 10 \
    --train_val_split_index 0 \
    --split train

python3 examples/scripts/myscripts/generate_progresslm_sft_data.py \
    --base_dataset_path "\
realworld_dataset/BimanualBikeRotorInstall-Nominal-real,\
realworld_dataset/BimanualClearKitchenCounter-Nominal-real,\
realworld_dataset/BimanualSetUpBreakfastTable-Nominal-real,\
realworld_dataset/CleanLitterBox-Nominal-real,\
realworld_dataset/CutAppleIntoSlices-Nominal-real,\
realworld_dataset/PushCoasterToMug-Nominal-real,\
realworld_dataset/PushCoasterToMug-ObjectCentricDistributionShift-real,\
realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real,\
realworld_dataset/PutKiwiInCenterOfTable-StationDistributionShift-real,\
realworld_dataset/PutKiwiInCenterOfTableSeenTasks_backfill-salem--video,\
realworld_dataset/TurnMugRightsideUp-Nominal-real,\
realworld_dataset/TurnMugRightsideUp-ObjectCentricDistributionShift-real,\
realworld_dataset/TurnMugRightsideUp-StationDistributionShift-real" \
    --image_output_dir /workspace/hf_trl/trl/progresLM_dataset_realworld \
    --output_jsonl /workspace/hf_trl/trl/progresLM_dataset_realworld/robocasa_progresslm_sft.jsonl \
    --num_demo_frames 5 \
    --samples_per_trajectory 10 \
    --train_val_split_index 0 \
    --split train


The output JSONL can be directly used with LLaMA-Factory for LoRA SFT
on Qwen2.5-VL-3B (nothink mode — no CoT, just direct percentage output).
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from tqdm import tqdm

# ---- Import your existing data loading utilities ----
from video_frame_utils import extract_frame, find_job_dirs
from sft_vlm_overlay_regression_v2 import (
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
)
from sft_vlm_overlay_regression_realworld_data import (
    is_realworld_format,
    load_realworld_trajectories,
)


# =====================================================================
# ProgressLM prompt templates (nothink / direct-prediction)
# =====================================================================

VISUAL_DEMO_SYSTEM_PROMPT = (
    "You are a progress estimator that evaluates the progress of the "
    "current state during an ongoing task based on a visual demonstration. "
    "The demonstration consists of a sequence of vision-based states and "
    "their corresponding progress value (ranging from 0% to 100%), showing "
    "how the task evolves from start to completion."
)

VISUAL_DEMO_INSTRUCTION_NOTHINK = (
    "Based on the task goal, demonstration, and current image, output ONLY "
    'the estimated progress as a percentage (0%–100%), or output exactly '
    '"n/a" if the target is incorrect, unmatched, or any abnormal condition '
    "exists; output nothing else."
)


def format_visual_demo_progress_shifts(total_steps: int) -> str:
    """
    Build the '<image> 0% <image> 25% ... <image> 100%' string.
    total_steps = number of transitions (= num_demo_frames - 1).
    """
    parts = []
    for i in range(total_steps + 1):
        pct = round((i / total_steps) * 100)
        parts.append(f"<image> {pct}%")
    return " ".join(parts)


def build_user_message_nothink(task_goal: str, total_steps: int) -> str:
    """
    Build the user-turn text with <image> placeholders.

    The prompt will contain (total_steps + 1) <image> tags for demo frames
    plus 1 more <image> tag for the observation = total_steps + 2 total.
    """
    parts = [
        VISUAL_DEMO_SYSTEM_PROMPT,
        f"\n\nThe overall task goal is {task_goal}.",
        "\n\nHere is the demonstration:",
        format_visual_demo_progress_shifts(total_steps),
        "\n\nHere is the current state that you need to estimate:",
        "<image>",
        "\n\n" + VISUAL_DEMO_INSTRUCTION_NOTHINK,
    ]
    return "\n".join(parts)


# =====================================================================
# Frame extraction & saving
# =====================================================================

def extract_and_save_frame(
    video_path: str,
    frame_idx: int,
    save_path: str,
) -> bool:
    """Extract a single frame and save as PNG. Returns True on success."""
    try:
        frame = extract_frame(video_path, frame_idx).convert("RGB")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frame.save(save_path)
        return True
    except Exception as e:
        print(f"  Warning: failed to extract frame {frame_idx} from {video_path}: {e}")
        return False


def get_demo_frame_indices(last_frame: int, num_demo_frames: int, first_frame: int = 0) -> List[int]:
    """Evenly spaced indices from first_frame to last_frame (inclusive endpoints)."""
    if num_demo_frames <= 1:
        return [first_frame]
    span = last_frame - first_frame
    return [
        first_frame + int(i * span / (num_demo_frames - 1))
        for i in range(num_demo_frames)
    ]


def sample_observation_indices(
    last_frame: int,
    num_samples: int,
    rng: random.Random,
    min_frame: int = 0,
) -> List[int]:
    """
    Sample observation frame indices along the trajectory.
    Uses a mix of uniform random + evenly-spaced to ensure coverage.
    """
    all_frames = list(range(min_frame, last_frame + 1))
    if len(all_frames) <= num_samples:
        return all_frames

    # Half evenly spaced, half random — ensures we cover early/mid/late
    n_even = num_samples // 2
    n_rand = num_samples - n_even

    span = last_frame - min_frame
    even = [
        min_frame + int(i * span / max(n_even - 1, 1))
        for i in range(n_even)
    ]
    remaining = [f for f in all_frames if f not in set(even)]
    rand = rng.sample(remaining, min(n_rand, len(remaining)))

    combined = sorted(set(even + rand))
    # Trim to exactly num_samples if we got extras from set union
    if len(combined) > num_samples:
        combined = rng.sample(combined, num_samples)
        combined.sort()
    return combined


# =====================================================================
# Main generation logic
# =====================================================================

def find_closest_demo_idx(
    obs_frame: int,
    demo_frame_indices: List[int],
) -> int:
    """
    Find the 1-based index of the closest demo frame to the observation.
    Returns int in [1, len(demo_frame_indices)].
    """
    best_idx = 0
    best_dist = abs(obs_frame - demo_frame_indices[0])
    for i, df in enumerate(demo_frame_indices):
        d = abs(obs_frame - df)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx + 1  # 1-based


def generate_sft_data(
    base_dataset_path: str,
    image_output_dir: str,
    split: str = "train",
    train_val_split_index: int = 5,
    num_demo_frames: int = 5,
    samples_per_trajectory: int = 10,
    include_failures: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate ProgressLM SFT samples from your trajectory dataset.

    For each success trajectory:
      - Pick a demo trajectory (first success with same demo_id)
      - Extract num_demo_frames evenly-spaced demo frames
      - Sample observation frames along the trajectory
      - Compute progress_score = frame_idx / last_frame * 100
      - Build the ShareGPT format entry

    Returns list of ShareGPT-formatted dicts.
    """
    rng = random.Random(seed)
    _is_realworld = is_realworld_format(base_dataset_path)

    # ---- Load data ----
    if _is_realworld:
        print("Detected realworld dataset format")
        realworld_trajs = load_realworld_trajectories(base_dataset_path)
        task_name = Path(base_dataset_path).name

        # Episode-level train/val split (matching eval script)
        all_episode_ids = sorted(set(t["episode_id"] for t in realworld_trajs))
        split_rng = random.Random(42)
        split_rng.shuffle(all_episode_ids)
        n_eval = max(1, int(len(all_episode_ids) * 0.1))
        eval_ids = set(all_episode_ids[:n_eval])
        train_ids = set(all_episode_ids[n_eval:])

        if split == "train":
            keep_ids = train_ids
        elif split == "val":
            keep_ids = eval_ids
        else:
            keep_ids = set(all_episode_ids)

        realworld_trajs = [t for t in realworld_trajs if t["episode_id"] in keep_ids]
        print(f"Using {len(realworld_trajs)} trajectories for '{split}' split "
              f"({len(keep_ids)} episodes)")

        # Convert realworld trajectories to the format expected by the rest of the code.
        # Realworld frames are 1-based (frame_000001.png to frame_NNNNNN.png).
        # Use frames_dir as video_path (extract_frame handles PNG directories).
        # Assign demo_id=0 for all so they share one demo source.
        first_frame = 1  # realworld frames start at 1
        success_data = []
        failure_data_raw = []
        for t in realworld_trajs:
            entry = {
                "video_path": t["frames_dir"],
                "trajectory_index": t["num_frames"],  # last frame = num_frames (1-based)
                "sf": t["sf"],
                "demo_id": 0,  # all share same demo
                "demo_success": t["sf"] == "success",
                "task_token": task_name,
            }
            if t["sf"] == "success":
                success_data.append(entry)
            else:
                failure_data_raw.append(entry)

        print(f"Loaded {len(success_data)} success + {len(failure_data_raw)} failure trajectories")
        failure_data = failure_data_raw if include_failures else []

        # Task tag = directory name for realworld
        task_tag = task_name.replace(" ", "_")
    else:
        first_frame = 0  # simulation frames start at 0
        print("Finding job directories...")
        job_dirs = find_job_dirs(base_dataset_path)

        if split == "train" and train_val_split_index > 0:
            job_dirs = job_dirs[:-train_val_split_index]
        elif split == "val" and train_val_split_index > 0:
            job_dirs = job_dirs[-train_val_split_index:]

        print(f"Using {len(job_dirs)} job directories for '{split}' split")

        success_data, failure_data_raw = load_trajectories(job_dirs)
        print(f"Loaded {len(success_data)} success + {len(failure_data_raw)} failure trajectories")

        if include_failures:
            failure_data = match_failures_to_successes(success_data, failure_data_raw)
            print(f"Matched {len(failure_data)} failure trajectories")
        else:
            failure_data = []

        # Derive task tag from path (e.g. ".../PnPSinkToCounter/2024-04-26_2" -> "PnPSinkToCounter")
        _path_parts = Path(base_dataset_path).parts
        if len(_path_parts) >= 2:
            task_tag = _path_parts[-2]
        else:
            task_tag = _path_parts[-1] if _path_parts else "unknown_task"
        task_tag = task_tag.replace(" ", "_")

    print(f"Task tag: {task_tag}")

    # ---- Group by demo_id ----
    success_by_demo = defaultdict(list)
    for traj in success_data:
        success_by_demo[traj["demo_id"]].append(traj)

    # ---- For each demo_id, pick ONE trajectory as the visual demo source ----
    #      (the first success, deterministic)
    demo_sources = {}
    demo_frame_cache = {}  # demo_id -> (demo_frame_indices, demo_image_relpaths)

    print(f"\nExtracting demo frames for {len(success_by_demo)} demo groups...")
    for demo_id in tqdm(sorted(success_by_demo.keys()), desc="Demo frames"):
        demo_traj = success_by_demo[demo_id][0]
        last_frame = demo_traj["trajectory_index"]
        video_path = demo_traj["video_path"]
        task_token = demo_traj.get("task_token", "unknown")

        demo_indices = get_demo_frame_indices(last_frame, num_demo_frames, first_frame=first_frame)
        demo_relpaths = []
        all_ok = True

        for di, fidx in enumerate(demo_indices):
            # Save under: image_output_dir / task_tag / demo_id / demo_XXX.png
            safe_demo_id = str(demo_id).replace("/", "_").replace("\\", "_")
            relpath = os.path.join(task_tag, safe_demo_id, f"demo_{di:03d}.png")
            abspath = os.path.join(image_output_dir, relpath)

            if not os.path.exists(abspath):
                ok = extract_and_save_frame(video_path, fidx, abspath)
                if not ok:
                    all_ok = False
                    break
            demo_relpaths.append(relpath)

        if all_ok:
            demo_sources[demo_id] = demo_traj
            demo_frame_cache[demo_id] = (demo_indices, demo_relpaths)

    print(f"Successfully extracted demos for {len(demo_sources)}/{len(success_by_demo)} demo groups")

    # ---- Generate observation samples ----
    total_steps = num_demo_frames - 1  # transitions between demo frames
    samples = []

    trajectories_to_process = list(success_data)
    if include_failures:
        trajectories_to_process += failure_data

    print(f"\nGenerating observation samples from {len(trajectories_to_process)} trajectories...")
    for traj in tqdm(trajectories_to_process, desc="Observation samples"):
        demo_id = traj["demo_id"]
        if demo_id not in demo_frame_cache:
            continue

        demo_indices, demo_relpaths = demo_frame_cache[demo_id]
        task_token = traj.get("task_token", "unknown")
        # Look up task goal — try exact match first, then substring match
        task_goal = TASK_TOKENS.get(task_token)
        if task_goal is None:
            for key, val in TASK_TOKENS.items():
                if key in task_token:
                    task_goal = val
                    break
        if task_goal is None:
            task_goal = task_token
        video_path = traj["video_path"]
        last_frame = traj["trajectory_index"]
        is_success = traj.get("demo_success", True)

        if last_frame <= first_frame:
            continue

        # Sample observation frame indices
        obs_indices = sample_observation_indices(
            last_frame=last_frame,
            num_samples=samples_per_trajectory,
            rng=rng,
            min_frame=first_frame,
        )

        for obs_idx in obs_indices:
            # Compute ground truth progress (normalize relative to first_frame)
            progress_pct = round((obs_idx - first_frame) / (last_frame - first_frame) * 100, 1)
            # Snap to integer if close
            if abs(progress_pct - round(progress_pct)) < 0.05:
                progress_pct = int(round(progress_pct))

            # Find closest demo frame (1-based)
            closest_idx = find_closest_demo_idx(obs_idx, demo_indices)

            # Save observation frame
            safe_demo_id = str(demo_id).replace("/", "_").replace("\\", "_")
            if _is_realworld:
                # For realworld, video_path is .../episode_X/models/model_name/frames
                # Use episode + model name for uniqueness
                vp = Path(video_path)
                traj_tag = f"{vp.parent.parent.parent.name}_{vp.parent.name}"
            else:
                traj_tag = Path(video_path).stem  # unique-ish per trajectory
            obs_relpath = os.path.join(
                task_tag,
                safe_demo_id,
                "observations",
                f"{traj_tag}_frame_{obs_idx:06d}.png",
            )
            obs_abspath = os.path.join(image_output_dir, obs_relpath)

            if not os.path.exists(obs_abspath):
                ok = extract_and_save_frame(video_path, obs_idx, obs_abspath)
                if not ok:
                    continue

            # ---- Build ShareGPT entry ----
            user_content = build_user_message_nothink(task_goal, total_steps)
            assistant_content = f"{progress_pct}%"

            # Image list: demo frames + observation frame
            all_image_paths = list(demo_relpaths) + [obs_relpath]

            # Sanity check: number of <image> tags should match
            n_tags = user_content.count("<image>")
            assert n_tags == len(all_image_paths), (
                f"Image tag mismatch: {n_tags} tags vs {len(all_image_paths)} images"
            )

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": all_image_paths,
            }
            samples.append(sample)

    print(f"\nGenerated {len(samples)} SFT samples total")
    return samples


def write_dataset_info_snippet(output_jsonl: str, image_output_dir: str):
    """Print the dataset_info.json entry to paste into LLaMA-Factory."""
    snippet = {
        "robocasa_progresslm_nothink": {
            "file_name": os.path.abspath(output_jsonl),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }
    print("\n" + "=" * 60)
    print("DATASET_INFO.JSON SNIPPET")
    print("Paste this into LLaMA-Factory/data/dataset_info.json:")
    print("=" * 60)
    print(json.dumps(snippet, indent=2))
    print("=" * 60)
    print(f"\nAlso set media_dir in your YAML config to: {os.path.abspath(image_output_dir)}")

    # Also save it
    snippet_path = os.path.join(os.path.dirname(output_jsonl), "dataset_info_snippet.json")
    with open(snippet_path, "w") as f:
        json.dump(snippet, f, indent=2)
    print(f"Snippet also saved to: {snippet_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ProgressLM SFT training data from RoboCasa trajectories"
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        required=True,
        help="Path(s) to your dataset root (comma-separated for multiple)",
    )
    parser.add_argument(
        "--image_output_dir",
        type=str,
        required=True,
        help="Directory to save extracted frame PNGs",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL file path for LLaMA-Factory",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--train_val_split_index", type=int, default=5)
    parser.add_argument("--num_demo_frames", type=int, default=5,
                        help="Number of demo frames per trajectory (default: 5)")
    parser.add_argument("--samples_per_trajectory", type=int, default=10,
                        help="Observation frames to sample per trajectory")
    parser.add_argument("--include_failures", action="store_true",
                        help="Also generate samples from failure trajectories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total samples (randomly subsampled)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(args.image_output_dir, exist_ok=True)

    # Support comma-separated dataset paths
    dataset_paths = [p.strip() for p in args.base_dataset_path.split(",") if p.strip()]

    all_samples = []
    for dpath in dataset_paths:
        print(f"\n{'#' * 60}")
        print(f"Processing: {dpath}")
        print(f"{'#' * 60}")

        samples = generate_sft_data(
            base_dataset_path=dpath,
            image_output_dir=args.image_output_dir,
            split=args.split,
            train_val_split_index=args.train_val_split_index,
            num_demo_frames=args.num_demo_frames,
            samples_per_trajectory=args.samples_per_trajectory,
            include_failures=args.include_failures,
            seed=args.seed,
        )
        all_samples.extend(samples)

    # Optional: cap total samples
    if args.max_samples and len(all_samples) > args.max_samples:
        rng = random.Random(args.seed)
        all_samples = rng.sample(all_samples, args.max_samples)
        print(f"Subsampled to {len(all_samples)} samples")

    # Shuffle
    rng = random.Random(args.seed + 1)
    rng.shuffle(all_samples)

    # Write JSONL
    print(f"\nWriting {len(all_samples)} samples to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total samples:    {len(all_samples)}")
    print(f"Images saved to:  {os.path.abspath(args.image_output_dir)}")
    print(f"JSONL saved to:   {os.path.abspath(args.output_jsonl)}")

    # Peek at first sample
    if all_samples:
        s = all_samples[0]
        print(f"\n--- First sample preview ---")
        print(f"  Num images: {len(s['images'])}")
        print(f"  User msg length: {len(s['messages'][0]['content'])} chars")
        print(f"  Assistant response: {s['messages'][1]['content']}")
        print(f"  Image paths: {s['images'][:2]} ... {s['images'][-1]}")

    # Print LLaMA-Factory integration instructions
    write_dataset_info_snippet(args.output_jsonl, args.image_output_dir)

    print(f"""
========================================
NEXT STEPS — Fine-tune with LLaMA-Factory
========================================

1. Copy the dataset_info snippet above into:
   LLaMA-Factory/data/dataset_info.json

2. Create/edit your training YAML (or copy from
   ProgressLM/LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_small_nothink.yaml):

   model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
   dataset: robocasa_progresslm_nothink
   template: qwen2_vl
   media_dir: {os.path.abspath(args.image_output_dir)}
   cutoff_len: 30000
   stage: sft
   finetuning_type: lora
   lora_rank: 8
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 8
   learning_rate: 1.0e-4
   num_train_epochs: 3
   bf16: true

3. Run training:
   cd LLaMA-Factory
   CUDA_VISIBLE_DEVICES=0 llamafactory-cli train your_config.yaml

4. Merge LoRA:
   llamafactory-cli export your_merge_config.yaml

5. Evaluate with your fixed evaluate_progressLM.py
""")


if __name__ == "__main__":
    main()