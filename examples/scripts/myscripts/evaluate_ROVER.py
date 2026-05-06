"""
ROVER Baseline Evaluation Script

Implements ROVER's recursive subtask decomposition approach (Schroeder et al., NeurIPS 2025)
adapted for pairwise frame comparison evaluation.

ROVER's key idea: decompose tasks into ordered subtasks, then reason about
per-frame progress through those subtasks. We adapt this for pairwise comparison by:
  1. Decomposing each task into ordered subtasks (one API call per task, cached)
  2. For each pair, providing the subtask decomposition and asking the VLM to
     estimate per-frame progress via subtask-aware reasoning, then compare.

Supports both Gemini API and local models (e.g., Qwen2.5-VL).
Backend is auto-detected: local paths use transformers, otherwise Gemini API.

Usage (Gemini):
export GOOGLE_API_KEY="AIzaSyCE3XxOpsV5H2-B_hXWM37yYDdBv03l5jc"
python3 /workspace/hf_trl/trl/examples/scripts/myscripts/evaluate_ROVER.py \
    --model_name_or_path "gemini-3-flash-preview" \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224" \
    --split val \
    --num_samples 10 \
    --visualize

Usage (Local Qwen):
CUDA_VISIBLE_DEVICES=0 python3 /workspace/hf_trl/trl/examples/scripts/myscripts/evaluate_ROVER.py \
    --model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224" \
    --split val \
    --num_samples 10 \
    --batch_size 8 \
    --visualize

python3 /workspace/hf_trl/trl/examples/scripts/myscripts/evaluate_ROVER.py \
    --model_name_or_path "gemini-3-flash-preview" \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224" \
    --split val \
    --num_samples 10 \
    --camera_crop center \
    --visualize

"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from video_frame_utils import create_side_by_side, extract_frame, find_job_dirs


def _is_local_model(model_name_or_path: str) -> bool:
    """Auto-detect whether model_name_or_path is a local path or a Gemini model ID."""
    return os.path.isdir(model_name_or_path)


def crop_camera_view(img: Image.Image, camera_crop: Optional[str]) -> Image.Image:
    """Crop one of three horizontally concatenated camera views from a frame."""
    if camera_crop is None:
        return img
    w, h = img.size
    third = w // 3
    if camera_crop == "left":
        return img.crop((0, 0, third, h))
    elif camera_crop == "center":
        return img.crop((third, 0, 2 * third, h))
    elif camera_crop == "right":
        return img.crop((2 * third, 0, w, h))
    return img

# Import shared code from training script
from sft_vlm_overlay_regression_v2 import (
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
    compute_failure_filter_stats,
)

# ============================================================================
# Task token to natural language description (same as GVL eval)
# ============================================================================

TASK_TOKEN_TO_DESC = {
    "[COUNTER_TO_CAB]": "Pick up the object from the counter and place it in the cabinet",
    "[CAB_TO_COUNTER]": "Pick up the object from the cabinet and place it on the counter",
    "[COUNTER_TO_MICROWAVE]": "Pick up the object from the counter and place it in the microwave",
    "[MICROWAVE_TO_COUNTER]": "Pick up the object from the microwave and place it on the counter",
    "[STOVE_TO_COUNTER]": "Pick up the object from the stove and place it on the counter",
    "[COUNTER_TO_STOVE]": "Pick up the object from the counter and place it on the stove",
    "[COUNTER_TO_SINK]": "Pick up the object from the counter and place it in the sink",
    "[SINK_TO_COUNTER]": "Pick up the object from the sink and place it on the counter",
    "[COFFEE_SERVE_MUG]": "Pick up the mug from the coffee machine and place it on the counter",
    "[CLOSE_DRAWER]": "Close the drawer",
    "[COFFEE_SETUP_MUG]": "Pick up the mug from the counter and place it on the coffee machine",
    "[COFFEE_PRESS_BUTTON]": "Press the button on the coffee machine",
    "[OPEN_DRAWER]": "Open the drawer",
    "[CLOSE_SINGLE_DOOR]": "Close the door",
    "[CLOSE_DOUBLE_DOOR]": "Close both of the doors",
    "[OPEN_DOUBLE_DOOR]": "Open both of the doors",
    "[OPEN_SINGLE_DOOR]": "Open the door",
    "[TURN_SINK_SPOUT]": "Turn the sink spout",
}

# ============================================================================
# ROVER Prompts
# ============================================================================

SUBTASK_DECOMPOSITION_PROMPT = """You are an expert roboticist. Given the following robotic manipulation task, decompose it into a sequence of ordered subtasks that the robot must complete from start to finish.

Task: {task_description}

List the subtasks as a numbered sequence. Each subtask should describe a distinct phase of the manipulation (e.g., approaching, grasping, lifting, transporting, placing). Be specific about spatial relationships and object states.

Output ONLY a JSON object with key "subtasks" containing a list of strings. Example:
{{"subtasks": ["Approach the object on the counter", "Grasp the object", "Lift the object from the counter", "Transport the object toward the cabinet", "Place the object inside the cabinet", "Release the object"]}}
"""

ROVER_SCORING_SYSTEM_PROMPT = """You are an expert roboticist evaluating task progress using subtask-aware reasoning.

You will be given:
1. A robotic manipulation task description
2. An ordered list of subtasks that compose this task
3. Three frames from a robot trajectory in chronological order:
   - Image 1: The initial scene (task completion = 0%)
   - Image 2: A recent frame showing the robot's state shortly before the current moment
   - Image 3: The current frame to evaluate

Use the transition from the recent frame to the current frame, along with the initial scene for reference, to determine:
- Which subtask the robot is currently performing or has just completed
- The overall task completion percentage (0-100), where 0 means the task has not started and 100 means fully complete

You MUST respond with a JSON object containing:
- "current_subtask": which subtask the robot is currently in
- "progress": integer 0-100
- "reasoning": brief explanation

Output MUST be exactly one JSON object. No Markdown, no ``` fences, no extra text."""

ROVER_SCORING_USER_PROMPT = """Task: {task_description}

Subtasks (in order):
{subtask_list}

Image 1 is the initial scene (0% progress). Image 2 is a recent frame. Image 3 is the current frame to evaluate.
Using the progression from the initial scene through the recent frame to the current frame, identify which subtask the robot is currently performing and estimate the overall task completion percentage (0-100)."""

# Number of frames to look back for the "previous frame" in the sliding window
SLIDING_WINDOW_LOOKBACK = 4

# ============================================================================
# Subtask Decomposition (cached per task)
# ============================================================================

def get_subtask_decomposition(client, model_name, task_token, cache):
    """Get subtask decomposition for a task using Gemini API, with cache."""
    from google.genai import types

    if task_token in cache:
        return cache[task_token]

    task_desc = TASK_TOKEN_TO_DESC.get(task_token, task_token)
    prompt = SUBTASK_DECOMPOSITION_PROMPT.format(task_description=task_desc)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text.strip())
            subtasks = result.get("subtasks", [])
            if subtasks:
                cache[task_token] = subtasks
                print(f"  Decomposed '{task_token}' into {len(subtasks)} subtasks:")
                for i, st in enumerate(subtasks, 1):
                    print(f"    {i}. {st}")
                return subtasks
        except Exception as e:
            print(f"  Subtask decomposition attempt {attempt+1} failed: {e}")
            time.sleep(2)

    # Fallback: generic subtasks
    fallback = [
        "Approach the target object",
        "Grasp the object",
        "Lift and transport the object",
        "Place the object at the destination",
        "Release the object",
    ]
    cache[task_token] = fallback
    print(f"  Using fallback subtasks for '{task_token}'")
    return fallback


def get_subtask_decomposition_local(model, processor, task_token, cache, device):
    """Get subtask decomposition for a task using a local VLM, with cache."""
    if task_token in cache:
        return cache[task_token]

    task_desc = TASK_TOKEN_TO_DESC.get(task_token, task_token)
    prompt = SUBTASK_DECOMPOSITION_PROMPT.format(task_description=task_desc)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, top_p=0.95, do_sample=True)

    gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    completion = processor.decode(gen_tokens, skip_special_tokens=True).strip()

    try:
        # Try to extract JSON from the completion
        json_match = re.search(r'\{.*\}', completion, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            subtasks = result.get("subtasks", [])
            if subtasks:
                cache[task_token] = subtasks
                print(f"  Decomposed '{task_token}' into {len(subtasks)} subtasks:")
                for i, st in enumerate(subtasks, 1):
                    print(f"    {i}. {st}")
                return subtasks
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Failed to parse subtask decomposition: {e}")

    # Fallback
    fallback = [
        "Approach the target object",
        "Grasp the object",
        "Lift and transport the object",
        "Place the object at the destination",
        "Release the object",
    ]
    cache[task_token] = fallback
    print(f"  Using fallback subtasks for '{task_token}'")
    return fallback


# ============================================================================
# Inference
# ============================================================================

def _score_frame_gemini(async_client, model_name, initial_frame, previous_frame, current_frame, user_prompt):
    """Score a single frame using Gemini API with 3-frame sliding window. Returns (progress, raw_completion)."""
    from google.genai import types

    try:
        response = async_client.models.generate_content(
            model=model_name,
            contents=[initial_frame, previous_frame, current_frame, user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=ROVER_SCORING_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        completion = response.text.strip()
    except Exception as e:
        return None, f"Error: {e}"

    try:
        json_match = re.search(r'\{.*\}', completion, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            progress = int(result.get("progress", -1))
            if 0 <= progress <= 100:
                return progress, completion
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: extract any number
    m = re.search(r'\b(\d{1,3})\b', completion)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val, completion

    return None, completion


def run_inference_gemini(client, model_name, pairs, batch_size, device, subtask_cache, max_new_tokens=64, camera_crop=None):
    """Run ROVER-faithful inference: score each frame independently, then compare.

    Each pair requires 2 API calls (one per frame). Uses async concurrency.
    """
    import asyncio
    from google import genai as genai_sync
    from google.genai import types

    results = [None] * len(pairs)

    print(f"Starting ROVER evaluation on {len(pairs)} pairs using {model_name}...")

    # Phase 1: Pre-compute subtask decompositions for all unique tasks
    unique_tasks = set(item["task_token"] for item in pairs)
    print(f"\nPhase 1: Decomposing {len(unique_tasks)} unique tasks into subtasks...")
    for task_token in sorted(unique_tasks):
        get_subtask_decomposition(client, model_name, task_token, subtask_cache)

    # Phase 2: Score each frame independently, then compare
    concurrency = max(batch_size, 1)
    print(f"\nPhase 2: Scoring {len(pairs)} pairs (2 calls each, concurrency={concurrency})...")

    # Pre-build prompts per task (shared across pairs with same task)
    prompt_cache = {}
    for item in pairs:
        tk = item["task_token"]
        if tk not in prompt_cache:
            subtasks = subtask_cache.get(tk, [])
            task_desc = TASK_TOKEN_TO_DESC.get(tk, tk)
            subtask_list = "\n".join(f"  {j+1}. {st}" for j, st in enumerate(subtasks))
            prompt_cache[tk] = ROVER_SCORING_USER_PROMPT.format(
                task_description=task_desc,
                subtask_list=subtask_list,
            )

    async_client = genai_sync.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    async def _score_pair(idx):
        item = pairs[idx]
        user_prompt = prompt_cache[item["task_token"]]

        # Extract 3-frame sliding window for each frame:
        # (initial=frame 0, previous=frame t-lookback, current=frame t)
        try:
            idx1 = item["frame_idx_1"]
            initial1 = crop_camera_view(extract_frame(item["video_path_1"], 0), camera_crop)
            prev_idx1 = max(0, idx1 - SLIDING_WINDOW_LOOKBACK)
            previous1 = crop_camera_view(extract_frame(item["video_path_1"], prev_idx1), camera_crop)
            frame1 = crop_camera_view(extract_frame(item["video_path_1"], idx1), camera_crop)
        except Exception as e:
            print(f"\nWarning: failed to load frame1: {e}")
            dummy = Image.new("RGB", (128, 128), (128, 128, 128))
            initial1 = previous1 = frame1 = dummy

        try:
            idx2 = item["frame_idx_2"]
            initial2 = crop_camera_view(extract_frame(item["video_path_2"], 0), camera_crop)
            prev_idx2 = max(0, idx2 - SLIDING_WINDOW_LOOKBACK)
            previous2 = crop_camera_view(extract_frame(item["video_path_2"], prev_idx2), camera_crop)
            frame2 = crop_camera_view(extract_frame(item["video_path_2"], idx2), camera_crop)
        except Exception as e:
            print(f"\nWarning: failed to load frame2: {e}")
            dummy = Image.new("RGB", (128, 128), (128, 128, 128))
            initial2 = previous2 = frame2 = dummy

        # Score both frames independently (concurrent)
        score1, raw1 = await asyncio.to_thread(
            _score_frame_gemini, async_client, model_name, initial1, previous1, frame1, user_prompt
        )
        score2, raw2 = await asyncio.to_thread(
            _score_frame_gemini, async_client, model_name, initial2, previous2, frame2, user_prompt
        )

        # Compare scores to determine prediction
        raw_completion = f"frame1_score={score1} ({raw1[:80]}...) | frame2_score={score2} ({raw2[:80]}...)"
        if score1 is not None and score2 is not None:
            if score2 > score1:
                pred = 32.0  # right/frame2 shows more progress
            elif score1 > score2:
                pred = -32.0  # left/frame1 shows more progress
            else:
                pred = 32.0 if random.random() < 0.5 else -32.0  # tie-break randomly
        else:
            pred = None

        gt = float(item["correct_answer"])

        if pred is not None:
            error = pred - gt
            abs_error = abs(error)
            sign_correct = (np.sign(gt) == np.sign(pred)) if gt != 0 else (pred == 0)
        else:
            error = None
            abs_error = None
            sign_correct = None

        results[idx] = {
            "ground_truth": gt,
            "prediction": pred,
            "raw_completion": raw_completion,
            "frame1_score": score1,
            "frame2_score": score2,
            "error": error,
            "abs_error": abs_error,
            "sign_correct": bool(sign_correct) if sign_correct is not None else None,
            "demo_type": item.get("demo_success", "unknown"),
            "demo_id": item.get("demo_id", "unknown"),
            "task_token": item["task_token"],
            "frame_idx_1": item["frame_idx_1"],
            "frame_idx_2": item["frame_idx_2"],
            "video_path_1": item["video_path_1"],
            "video_path_2": item["video_path_2"],
        }

    async def _run_all():
        semaphore = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(pairs), desc="ROVER Evaluating")

        async def _bounded(idx):
            async with semaphore:
                await _score_pair(idx)
                pbar.update(1)

        await asyncio.gather(*[_bounded(i) for i in range(len(pairs))])
        pbar.close()

    asyncio.run(_run_all())

    return results


def _score_frames_local_batched(model, processor, frame_triples, user_prompt, batch_size, device, max_new_tokens=256, pbar=None):
    """Score a batch of frames using a local VLM with 3-frame sliding window.

    Each entry is (initial_frame, previous_frame, current_frame).
    Returns list of (progress_score, raw_completion) tuples.
    """
    all_scores = []
    num_batches = (len(frame_triples) + batch_size - 1) // batch_size

    for b in range(num_batches):
        batch = frame_triples[b * batch_size : (b + 1) * batch_size]
        texts = []
        images_list = []

        for initial_frame, previous_frame, current_frame in batch:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": ROVER_SCORING_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": initial_frame},
                        {"type": "image", "image": previous_frame},
                        {"type": "image", "image": current_frame},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            try:
                import qwen_vl_utils
                image_input, _ = qwen_vl_utils.process_vision_info(conversation)
            except (ImportError, Exception):
                image_input = [initial_frame, previous_frame, current_frame]

            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            images_list.append(image_input)

        inputs = processor(
            text=texts,
            images=images_list if any(img is not None for img in images_list) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.95, do_sample=True
            )

        for i in range(len(batch)):
            gen_tokens = outputs[i, inputs["input_ids"].shape[1]:]
            completion = processor.decode(gen_tokens, skip_special_tokens=True).strip()

            score = None
            try:
                json_match = re.search(r'\{.*\}', completion, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    progress = int(result.get("progress", -1))
                    if 0 <= progress <= 100:
                        score = progress
            except (json.JSONDecodeError, Exception):
                pass

            if score is None:
                m = re.search(r'\b(\d{1,3})\b', completion)
                if m:
                    val = int(m.group(1))
                    if 0 <= val <= 100:
                        score = val

            all_scores.append((score, completion))

        if pbar is not None:
            pbar.update(len(batch))

    return all_scores


def run_inference_local(model, processor, pairs, batch_size, device, subtask_cache, max_new_tokens=256, camera_crop=None):
    """Run ROVER-faithful inference with local model: score each frame independently, then compare.

    Batches all frame scoring calls for efficiency.
    """
    results = []

    print(f"Starting ROVER evaluation on {len(pairs)} pairs using local model...")

    # Phase 1: Pre-compute subtask decompositions for all unique tasks
    unique_tasks = set(item["task_token"] for item in pairs)
    print(f"\nPhase 1: Decomposing {len(unique_tasks)} unique tasks into subtasks...")
    for task_token in sorted(unique_tasks):
        get_subtask_decomposition_local(model, processor, task_token, subtask_cache, device)

    # Phase 2: Collect all frames to score, then batch score them
    print(f"\nPhase 2: Scoring {len(pairs)} pairs (2 frames each)...")

    # Build prompt cache per task
    prompt_cache = {}
    for item in pairs:
        tk = item["task_token"]
        if tk not in prompt_cache:
            subtasks = subtask_cache.get(tk, [])
            task_desc = TASK_TOKEN_TO_DESC.get(tk, tk)
            subtask_list = "\n".join(f"  {j+1}. {st}" for j, st in enumerate(subtasks))
            prompt_cache[tk] = ROVER_SCORING_USER_PROMPT.format(
                task_description=task_desc,
                subtask_list=subtask_list,
            )

    # Collect all (initial_frame, previous_frame, current_frame) triples and their prompts
    # We interleave: [pair0_frame1, pair0_frame2, pair1_frame1, pair1_frame2, ...]
    all_frames = []
    all_prompts = []
    for item in pairs:
        user_prompt = prompt_cache[item["task_token"]]

        try:
            idx1 = item["frame_idx_1"]
            initial1 = crop_camera_view(extract_frame(item["video_path_1"], 0), camera_crop)
            prev_idx1 = max(0, idx1 - SLIDING_WINDOW_LOOKBACK)
            previous1 = crop_camera_view(extract_frame(item["video_path_1"], prev_idx1), camera_crop)
            frame1 = crop_camera_view(extract_frame(item["video_path_1"], idx1), camera_crop)
        except Exception as e:
            print(f"Warning: failed to load frame1: {e}")
            dummy = Image.new("RGB", (128, 128), (128, 128, 128))
            initial1 = previous1 = frame1 = dummy

        try:
            idx2 = item["frame_idx_2"]
            initial2 = crop_camera_view(extract_frame(item["video_path_2"], 0), camera_crop)
            prev_idx2 = max(0, idx2 - SLIDING_WINDOW_LOOKBACK)
            previous2 = crop_camera_view(extract_frame(item["video_path_2"], prev_idx2), camera_crop)
            frame2 = crop_camera_view(extract_frame(item["video_path_2"], idx2), camera_crop)
        except Exception as e:
            print(f"Warning: failed to load frame2: {e}")
            dummy = Image.new("RGB", (128, 128), (128, 128, 128))
            initial2 = previous2 = frame2 = dummy

        all_frames.append((initial1, previous1, frame1))
        all_frames.append((initial2, previous2, frame2))
        all_prompts.append(user_prompt)
        all_prompts.append(user_prompt)

    # Score all frames in batches (group by prompt for efficiency)
    print(f"  Scoring {len(all_frames)} individual frames (batch_size={batch_size})...")
    all_scores = []
    i = 0
    pbar = tqdm(total=len(all_frames), desc="Scoring frames")
    while i < len(all_frames):
        # Find run of same prompt
        current_prompt = all_prompts[i]
        j = i
        while j < len(all_frames) and all_prompts[j] == current_prompt:
            j += 1
        chunk_frames = all_frames[i:j]
        chunk_scores = _score_frames_local_batched(
            model, processor, chunk_frames, current_prompt, batch_size, device, max_new_tokens, pbar=pbar
        )
        all_scores.extend(chunk_scores)
        i = j
    pbar.close()

    # Reassemble into pair results
    for pair_idx, item in enumerate(pairs):
        score1, raw1 = all_scores[pair_idx * 2]
        score2, raw2 = all_scores[pair_idx * 2 + 1]

        raw_completion = f"frame1_score={score1} ({raw1[:80]}...) | frame2_score={score2} ({raw2[:80]}...)"

        if score1 is not None and score2 is not None:
            if score2 > score1:
                pred = 32.0
            elif score1 > score2:
                pred = -32.0
            else:
                pred = 32.0 if random.random() < 0.5 else -32.0
        else:
            pred = None

        gt = float(item["correct_answer"])

        if pred is not None:
            error = pred - gt
            abs_error = abs(error)
            sign_correct = (np.sign(gt) == np.sign(pred)) if gt != 0 else (pred == 0)
        else:
            error = None
            abs_error = None
            sign_correct = None

        results.append({
            "ground_truth": gt,
            "prediction": pred,
            "raw_completion": raw_completion,
            "frame1_score": score1,
            "frame2_score": score2,
            "error": error,
            "abs_error": abs_error,
            "sign_correct": bool(sign_correct) if sign_correct is not None else None,
            "demo_type": item.get("demo_success", "unknown"),
            "demo_id": item.get("demo_id", "unknown"),
            "task_token": item["task_token"],
            "frame_idx_1": item["frame_idx_1"],
            "frame_idx_2": item["frame_idx_2"],
            "video_path_1": item["video_path_1"],
            "video_path_2": item["video_path_2"],
        })

    return results



# ============================================================================
# Metrics & Visualization
# ============================================================================

def _get_interval_label(r):
    """Classify a result as 'intra-N' (same video, N frames apart) or 'sf' (success-failure)."""
    f1 = r.get("frame_idx_1")
    f2 = r.get("frame_idx_2")
    v1 = r.get("video_path_1")
    v2 = r.get("video_path_2")
    if v1 == v2 and isinstance(f1, int) and isinstance(f2, int):
        return f"intra-{abs(f2 - f1)}"
    return "sf"


def compute_metrics(results, tolerance=3):
    valid = [r for r in results if r["prediction"] is not None]
    if not valid:
        return {"error": "no valid predictions"}

    abs_errors = [r["abs_error"] for r in valid]
    errors = [r["error"] for r in valid]
    sign_correct = [r["sign_correct"] for r in valid]
    within_tol = [ae <= tolerance for ae in abs_errors]

    metrics = {
        "total_pairs": len(results),
        "valid_predictions": len(valid),
        "parse_failures": len(results) - len(valid),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "median_abs_error": float(np.median(abs_errors)),
        "within_tolerance_rate": float(np.mean(within_tol)),
        "sign_accuracy": float(np.mean(sign_correct)),
        "tolerance": tolerance,
    }

    # Per demo_type breakdown
    for dtype in set(r["demo_type"] for r in valid):
        subset = [r for r in valid if r["demo_type"] == dtype]
        sub_abs = [r["abs_error"] for r in subset]
        sub_sign = [r["sign_correct"] for r in subset]
        metrics[f"{dtype}_count"] = len(subset)
        metrics[f"{dtype}_mae"] = float(np.mean(sub_abs))
        metrics[f"{dtype}_sign_accuracy"] = float(np.mean(sub_sign))

    # Per interval breakdown
    interval_groups = defaultdict(list)
    for r in valid:
        interval_groups[_get_interval_label(r)].append(r)

    interval_metrics = {}
    for label, group in sorted(interval_groups.items()):
        g_sign = [r["sign_correct"] for r in group]
        interval_metrics[label] = {
            "count": len(group),
            "sign_accuracy": float(np.mean(g_sign)),
            "mae": float(np.mean([r["abs_error"] for r in group])),
        }
        metrics[f"interval_{label}_count"] = len(group)
        metrics[f"interval_{label}_sign_accuracy"] = float(np.mean(g_sign))
        metrics[f"interval_{label}_mae"] = float(np.mean([r["abs_error"] for r in group]))

    metrics["interval_breakdown"] = interval_metrics

    return metrics


def visualize_results(results, output_dir, num_examples=8, camera_crop=None):
    valid = [r for r in results if r["prediction"] is not None]
    if not valid:
        print("No valid results to visualize")
        return

    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    abs_errors = [r["abs_error"] for r in valid]
    errors = [r["error"] for r in valid]
    gts = [r["ground_truth"] for r in valid]
    preds = [r["prediction"] for r in valid]

    # --- Plot 1: Error distribution and scatter ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"ROVER Evaluation Results (n={len(valid)})", fontsize=14, fontweight="bold")

    axes[0, 0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(x=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Error (pred - gt)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Error Distribution")

    axes[0, 1].hist(abs_errors, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].axvline(x=np.mean(abs_errors), color="r", linestyle="--",
                       label=f"MAE={np.mean(abs_errors):.2f}")
    axes[0, 1].set_xlabel("Absolute Error")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Absolute Error Distribution")
    axes[0, 1].legend()

    axes[1, 0].scatter(gts, preds, alpha=0.4, s=15)
    mn, mx = min(min(gts), min(preds)), max(max(gts), max(preds))
    axes[1, 0].plot([mn, mx], [mn, mx], "r--", label="Perfect")
    axes[1, 0].set_xlabel("Ground Truth")
    axes[1, 0].set_ylabel("Prediction")
    axes[1, 0].set_title("Prediction vs Ground Truth")
    axes[1, 0].legend()
    axes[1, 0].set_aspect("equal", adjustable="box")

    axes[1, 1].scatter(gts, errors, alpha=0.4, s=15, c=abs_errors, cmap="coolwarm")
    axes[1, 1].axhline(y=0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("Ground Truth")
    axes[1, 1].set_ylabel("Error")
    axes[1, 1].set_title("Error vs Ground Truth")

    plt.tight_layout()
    plt.savefig(viz_dir / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Sign accuracy breakdown by interval ---
    interval_groups = defaultdict(list)
    for r in valid:
        interval_groups[_get_interval_label(r)].append(r)

    if interval_groups:
        # Sort: "sf" first, then "intra-N" by N
        def _sort_key(label):
            if label == "sf":
                return (0, 0)
            return (1, int(label.split("-")[1]))

        sorted_labels = sorted(interval_groups.keys(), key=_sort_key)
        accuracies = [np.mean([r["sign_correct"] for r in interval_groups[l]]) * 100 for l in sorted_labels]
        counts = [len(interval_groups[l]) for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(max(6, len(sorted_labels) * 1.5), 5))
        bars = ax.bar(range(len(sorted_labels)), accuracies, color=["#e74c3c" if l == "sf" else "#3498db" for l in sorted_labels],
                      edgecolor="black", alpha=0.85)

        # Add count and accuracy labels on bars
        for i, (bar, acc, cnt) in enumerate(zip(bars, accuracies, counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, fontsize=11)
        ax.set_ylabel("Sign Accuracy (%)", fontsize=12)
        ax.set_xlabel("Pair Type", fontsize=12)
        ax.set_title("Sign Accuracy by Compare Interval", fontsize=14, fontweight="bold")
        ax.set_ylim(0, min(max(accuracies) + 15, 105))
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(viz_dir / "accuracy_by_interval.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Plot 3: Sample overlays with 3-frame scoring context ---
    worst = sorted(valid, key=lambda x: x["abs_error"], reverse=True)[:num_examples]
    best = sorted(valid, key=lambda x: x["abs_error"])[:num_examples]

    for label, samples in [("worst", worst), ("best", best)]:
        n = len(samples)
        # Each sample gets 2 rows (frame1 context, frame2 context) x 3 cols (initial, previous, current)
        fig, axes_grid = plt.subplots(n * 2, 3, figsize=(12, 3.5 * n))
        if n == 1:
            axes_grid = axes_grid.reshape(2, 3)

        for idx, r in enumerate(samples):
            f1_idx = r.get("frame_idx_1", 0)
            f2_idx = r.get("frame_idx_2", 0)
            f1_score = r.get("frame1_score", "?")
            f2_score = r.get("frame2_score", "?")

            # Determine interval label
            if r.get("video_path_1") == r.get("video_path_2") and isinstance(f1_idx, int) and isinstance(f2_idx, int):
                interval_str = f"intra-{abs(f2_idx - f1_idx)}"
            else:
                interval_str = f"sf@{f1_idx}"

            color = "green" if r["abs_error"] <= 3 else "red"

            # Row 1: Frame 1 context (initial, previous, current)
            row_base = idx * 2
            try:
                initial1 = crop_camera_view(extract_frame(r["video_path_1"], 0), camera_crop)
                prev_idx1 = max(0, f1_idx - SLIDING_WINDOW_LOOKBACK) if isinstance(f1_idx, int) else 0
                previous1 = crop_camera_view(extract_frame(r["video_path_1"], prev_idx1), camera_crop)
                current1 = crop_camera_view(extract_frame(r["video_path_1"], f1_idx), camera_crop)
                frames1 = [initial1, previous1, current1]
                col_labels1 = [f"initial (f=0)", f"previous (f={prev_idx1})", f"current (f={f1_idx})"]
            except Exception:
                frames1 = [None, None, None]
                col_labels1 = ["initial", "previous", "current"]

            for col in range(3):
                ax = axes_grid[row_base, col]
                if frames1[col] is not None:
                    ax.imshow(frames1[col])
                else:
                    ax.text(0.5, 0.5, "Failed", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                ax.set_title(col_labels1[col], fontsize=8)

            # Add row label on the left column
            axes_grid[row_base, 0].set_ylabel(f"Frame1 (score={f1_score})", fontsize=9, fontweight="bold",
                                                color=color, rotation=0, labelpad=80, va="center")

            # Row 2: Frame 2 context (initial, previous, current)
            try:
                initial2 = crop_camera_view(extract_frame(r["video_path_2"], 0), camera_crop)
                prev_idx2 = max(0, f2_idx - SLIDING_WINDOW_LOOKBACK) if isinstance(f2_idx, int) else 0
                previous2 = crop_camera_view(extract_frame(r["video_path_2"], prev_idx2), camera_crop)
                current2 = crop_camera_view(extract_frame(r["video_path_2"], f2_idx), camera_crop)
                frames2 = [initial2, previous2, current2]
                col_labels2 = [f"initial (f=0)", f"previous (f={prev_idx2})", f"current (f={f2_idx})"]
            except Exception:
                frames2 = [None, None, None]
                col_labels2 = ["initial", "previous", "current"]

            for col in range(3):
                ax = axes_grid[row_base + 1, col]
                if frames2[col] is not None:
                    ax.imshow(frames2[col])
                else:
                    ax.text(0.5, 0.5, "Failed", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                ax.set_title(col_labels2[col], fontsize=8)

            axes_grid[row_base + 1, 0].set_ylabel(f"Frame2 (score={f2_score})", fontsize=9, fontweight="bold",
                                                    color=color, rotation=0, labelpad=80, va="center")

            # Add a shared super-label for this sample pair (use text above row 1)
            fig.text(0.5, 1.0 - (idx * 2) / (n * 2) - 0.005,
                     f"GT={r['ground_truth']:.0f} | Pred={r['prediction']:.1f} | Err={r['abs_error']:.1f} | "
                     f"{r['demo_type']} | {interval_str}",
                     ha="center", fontsize=10, fontweight="bold", color=color,
                     transform=fig.transFigure)

        plt.suptitle(f"ROVER {label.capitalize()} Predictions (3-frame context)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(viz_dir / f"samples_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ROVER baseline evaluation on pairwise progress comparison")

    p.add_argument("--model_name_or_path", type=str, default="gemini-3-flash-preview",
                   help="Gemini model ID or local model path (e.g., /path/to/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--base_model_name_or_path", type=str,
                   default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Base model path (used when loading PEFT adapters for local models)")
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Dataset path OR comma-separated list of dataset paths")
    p.add_argument("--split", type=str, default="val",
                   help="train | val | integer (number of job dirs to use)")
    p.add_argument("--train_val_split_index", type=int, default=5,
                   help="Last N job dirs used for val split")
    p.add_argument("--compare_interval", type=str, default="8,16",
                   help="Comma-separated frame intervals for success pair comparisons")
    p.add_argument("--sample_interval", type=int, default=5,
                   help="Frame sampling stride when building pairs")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of eval pairs (random subset)")
    p.add_argument("--tolerance", type=int, default=3,
                   help="Absolute error tolerance for within-tolerance metric")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    p.add_argument("--num_visualize", type=int, default=8,
                   help="Number of examples to show in best/worst visualizations")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output directory")
    p.add_argument("--camera_crop", type=str, default=None,
                   choices=["left", "center", "right"],
                   help="Crop a single camera view from the 3-view concatenated frame "
                        "(left=first 1/3, center=middle 1/3, right=last 1/3). "
                        "Default: None (use full frame)")

    return p.parse_args()


def _split_dataset_paths(base_dataset_path_csv: str):
    s = base_dataset_path_csv.strip()
    if not s:
        return []
    parts = []
    cur = []
    escape = False
    for ch in s:
        if escape:
            cur.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == ",":
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
    last = "".join(cur).strip()
    if last:
        parts.append(last)
    return parts


def _dataset_output_tag(dataset_path: str) -> str:
    name = Path(dataset_path).name.strip()
    if not name:
        name = "dataset"
    return name.replace(" ", "_")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_intervals = [int(x.strip()) for x in args.compare_interval.split(",")]

    dataset_paths = _split_dataset_paths(args.base_dataset_path)
    if not dataset_paths:
        raise ValueError("--base_dataset_path is empty.")

    use_local = _is_local_model(args.model_name_or_path)

    crop_tag = f"_crop{args.camera_crop}" if args.camera_crop else ""
    root_output_dir = args.output_dir or os.path.join(
        "rover-vlm/outputs" if not use_local else args.model_name_or_path,
        f"{args.prefix}rover_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}{crop_tag}"
    )
    os.makedirs(root_output_dir, exist_ok=True)

    # ---- Initialize model ----
    if use_local:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading local model for ROVER evaluation: {args.model_name_or_path}")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
        dtype = dtype_map.get(args.dtype, "auto")

        is_peft = os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json"))
        if is_peft:
            from peft import PeftModel
            print(f"Loading base model from {args.base_model_name_or_path}")
            model = AutoModelForImageTextToText.from_pretrained(
                args.base_model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Loading PEFT adapter from {args.model_name_or_path}")
            model = PeftModel.from_pretrained(model, args.model_name_or_path, is_trainable=False)
            processor_path = args.base_model_name_or_path
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=True,
            )
            processor_path = args.model_name_or_path

        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        model.eval()
        client = None
        print("Local model loaded")
    else:
        from google import genai
        print(f"Initializing Gemini Client for ROVER evaluation with model: {args.model_name_or_path}")
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = None
        processor = None

    # Subtask decomposition cache (persists across datasets)
    subtask_cache = {}

    all_metrics = {}
    all_summaries = []

    # ---- Evaluate each dataset ----
    for base_dataset_path in dataset_paths:
        print("\n" + "#" * 80)
        print(f"Evaluating dataset: {base_dataset_path}")
        print("#" * 80)

        dataset_tag = _dataset_output_tag(base_dataset_path)
        output_dir = os.path.join(root_output_dir, dataset_tag)
        os.makedirs(output_dir, exist_ok=True)

        # ---- Load dataset ----
        print("Loading dataset...")
        job_dirs = find_job_dirs(base_dataset_path)

        if 'robocasa/datasets' in base_dataset_path:
            job_dirs = job_dirs[:]
        elif args.split == "val":
            if args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[-args.train_val_split_index:]
        elif args.split == "train":
            if args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[:-args.train_val_split_index]
        else:
            job_dirs = job_dirs[:int(args.split)]

        print(f"Using {len(job_dirs)} job directories for '{args.split}' split")

        success_data, unfiltered_failure_data = load_trajectories(job_dirs)
        print(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

        failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
        print(f"Matched {len(failure_data)} failures to success trajectories")

        # Balanced sampling
        if success_data and failure_data:
            success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(
                success_data, failure_data, 50
            )
        else:
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
                if len(items) > 50:
                    items = rng.sample(items, 50)
                    success_by_demo[demo_id] = items
                capped_success.extend(items)
            capped_failure = []
            for demo_id in sorted(failure_by_demo.keys()):
                items = failure_by_demo[demo_id]
                if len(items) > 50:
                    items = rng.sample(items, 50)
                    failure_by_demo[demo_id] = items
                capped_failure.extend(items)

            print(f"After per-demo cap (50): "
                  f"{len(success_data)} -> {len(capped_success)} success, "
                  f"{len(failure_data)} -> {len(capped_failure)} failure")
            success_data = capped_success
            failure_data = capped_failure

        # Load cached failure filter stats
        if failure_data:
            success_mean_diffs_at_idx = None
            stats_cache_file = Path(base_dataset_path) / "failure_filter_stats.json"
            if stats_cache_file.exists():
                print(f"Loading failure filter stats from {stats_cache_file}")
                with open(stats_cache_file, "r") as f:
                    cached = json.load(f)
                    success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]
                print("  Will apply same failure-frame filter as training")
            else:
                print("No cached stats found, computing failure filter stats...")
                success_mean_diffs_at_idx = compute_failure_filter_stats(
                    success_by_demo, failure_by_demo, 0, 1
                )
                with open(stats_cache_file, "w") as f:
                    json.dump({"success_mean_diffs_at_idx": success_mean_diffs_at_idx}, f)
                print(f"Saved failure filter stats to {stats_cache_file}")
        else:
            success_mean_diffs_at_idx = {}
            print("Skipping failure filter stats (no failure data)")

        job_name = Path(job_dirs[0]).name if job_dirs else ""
        pairs = build_frame_pairs(
            success_data=success_data,
            failure_data=failure_data,
            compare_intervals=compare_intervals,
            train_sample_interval=args.sample_interval,
            success_mean_diffs_at_idx=success_mean_diffs_at_idx,
            job_name=job_name,
            local_rank=0,
            world_size=1,
            task_path=base_dataset_path,
        )
        print(f"Built {len(pairs)} evaluation pairs")

        if args.num_samples and args.num_samples < len(pairs):
            subsample_rng = random.Random(args.seed)
            pairs = subsample_rng.sample(pairs, args.num_samples)
            print(f"Subsampled to {len(pairs)} pairs")

            # Stratified sampling: balance across interval types
            groups = defaultdict(list)
            for p in pairs:
                groups[_get_interval_label(p)].append(p)

            n_groups = len(groups)
            target_per_group = args.num_samples // n_groups
            sampled = []

            # First pass: take min(target_per_group, available) from each group
            leftover_budget = 0
            group_order = sorted(groups.keys(),
                                 key=lambda l: (0, 0) if l == "sf" else (1, int(l.split("-")[1])))
            first_pass = {}
            for label in group_order:
                available = len(groups[label])
                take = min(target_per_group, available)
                first_pass[label] = subsample_rng.sample(groups[label], take)
                leftover_budget += target_per_group - take

            # Second pass: distribute remaining budget to groups that have surplus
            if leftover_budget > 0:
                for label in group_order:
                    available_remaining = len(groups[label]) - len(first_pass[label])
                    if available_remaining > 0 and leftover_budget > 0:
                        extra = min(available_remaining, leftover_budget)
                        already_taken = set(id(p) for p in first_pass[label])
                        pool = [p for p in groups[label] if id(p) not in already_taken]
                        first_pass[label].extend(subsample_rng.sample(pool, extra))
                        leftover_budget -= extra

            for label in group_order:
                sampled.extend(first_pass[label])

            pairs = sampled
            # Print breakdown
            breakdown_str = ", ".join(
                f"{label}={len(first_pass[label])}" for label in group_order
            )
            print(f"Subsampled to {len(pairs)} pairs (stratified: {breakdown_str})")

        # ---- Run ROVER inference ----
        if use_local:
            results = run_inference_local(
                model, processor, pairs, args.batch_size, device, subtask_cache,
                camera_crop=args.camera_crop,
            )
        else:
            results = run_inference_gemini(
                client, args.model_name_or_path, pairs, args.batch_size, device, subtask_cache,
                camera_crop=args.camera_crop,
            )

        # ---- Compute metrics ----
        metrics = compute_metrics(results, tolerance=args.tolerance)
        all_metrics[base_dataset_path] = metrics

        print("\n" + "=" * 60)
        print("ROVER EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model:            {args.model_name_or_path}")
        print(f"Dataset:          {base_dataset_path}")
        print(f"Split:            {args.split}")
        print(f"Total pairs:      {metrics.get('total_pairs', 0)}")
        print(f"Valid predictions: {metrics.get('valid_predictions', 0)}")
        print(f"Parse failures:   {metrics.get('parse_failures', 0)}")
        print(f"MAE:              {metrics.get('mae', float('nan')):.3f}")
        print(f"RMSE:             {metrics.get('rmse', float('nan')):.3f}")
        print(f"Median AE:        {metrics.get('median_abs_error', float('nan')):.3f}")
        print(f"Within tol ({args.tolerance}):   {metrics.get('within_tolerance_rate', float('nan')):.3f}")
        print(f"Sign accuracy:    {metrics.get('sign_accuracy', float('nan')):.3f}")

        for key in sorted(metrics.keys()):
            if key.endswith("_mae") and key != "mae" and not key.startswith("interval_"):
                dtype_name = key.replace("_mae", "")
                count = metrics.get(f"{dtype_name}_count", 0)
                mae = metrics[key]
                sign_acc = metrics.get(f"{dtype_name}_sign_accuracy", float("nan"))
                print(f"  {dtype_name}: n={count}, MAE={mae:.3f}, sign_acc={sign_acc:.3f}")

        # Per-interval breakdown
        interval_breakdown = metrics.get("interval_breakdown", {})
        if interval_breakdown:
            print("-" * 60)
            print("Breakdown by compare interval:")
            for label, stats in sorted(interval_breakdown.items(),
                                        key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1]))):
                print(f"  {label:>10s}: n={stats['count']:>4d}, sign_acc={stats['sign_accuracy']:.3f}, MAE={stats['mae']:.3f}")

        print("=" * 60)

        # ---- Save results ----
        summary = {
            "args": {**vars(args), "base_dataset_path": base_dataset_path},
            "metrics": metrics,
            "subtask_decompositions": subtask_cache,
            "timestamp": datetime.now().isoformat(),
        }
        all_summaries.append(summary)

        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")

        completions_path = os.path.join(output_dir, "completions.json")
        with open(completions_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Completions saved to {completions_path}")

        # ---- Visualize ----
        if args.visualize:
            visualize_results(results, output_dir, num_examples=args.num_visualize, camera_crop=args.camera_crop)

        print(f"\nDataset outputs saved to {output_dir}")

    # ---- Save combined metrics ----
    combined_metrics_path = os.path.join(root_output_dir, "all_metrics.json")
    with open(combined_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {combined_metrics_path}")

    combined_summaries_path = os.path.join(root_output_dir, "all_summaries.json")
    with open(combined_summaries_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All summaries saved to {combined_summaries_path}")

    # Save subtask decompositions separately for reference
    subtasks_path = os.path.join(root_output_dir, "subtask_decompositions.json")
    with open(subtasks_path, "w") as f:
        json.dump(subtask_cache, f, indent=2)
    print(f"Subtask decompositions saved to {subtasks_path}")

    print(f"\nAll outputs saved to {root_output_dir}")


if __name__ == "__main__":
    main()
