"""
Evaluate ProgressLM-3B (SFT/RL) as a fine-tuned progress baseline on the
VLM Overlay Regression v2 pairwise progress-comparison benchmark.

This script reuses the *same* dataset loading + pair construction code as
`evaluate_vlm_overlay_regression_v2.py`, but swaps inference to:

  1) Score each frame independently with ProgressLM: s in [0, 100]
  2) Predict pairwise progress delta: pred = s2 - s1 (clipped to [-100, 100])

So your existing metrics (sign accuracy, MAE, interval breakdown, viz) work unchanged.

Example:

# ----- Expert: RoboCasa original datasets -----
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

CUDA_VISIBLE_DEVICES=6 python3 examples/scripts/myscripts/evaluate_progressLM.py \
  --model_name_or_path /workspace/hf_trl/trl/ProgressLM/qwen25vl_7b_nothink_multitask_merged_4180 \
  --base_dataset_path  "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24" \
  --split val \
  --compare_interval 16,30,50,74,90,100 \
  --batch_size 10 \
  --num_samples 100 \
  --train_val_split_index 0 \
  --visualize  --nothink

CUDA_VISIBLE_DEVICES=6 python3 examples/scripts/myscripts/evaluate_progressLM.py \
  --model_name_or_path /workspace/hf_trl/trl/ProgressLM/qwen25vl_7b_nothink_multitask_merged_148 \
  --base_dataset_path "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24,\
/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29" \
  --split val \
  --compare_interval 16,30,50,74,90,100 \
  --batch_size 20 \
  --num_samples 200 \
  --train_val_split_index 0 \
  --visualize  --nothink

Notes:
- If you hit `KeyError: 'qwen2_5_vl'` with transformers, upgrade transformers
  (often `pip install -U git+https://github.com/huggingface/transformers`).

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train our_scripts/qwen2_5vl_lora_sft_small_nothink.yaml
CUDA_VISIBLE_DEVICES=0,2,3,4 llamafactory-cli train our_scripts/qwen2_5vl_lora_sft_small_nothink.yaml
CUDA_VISIBLE_DEVICES=5,6,7 llamafactory-cli train our_scripts/qwen2_5vl_lora_sft_small_nothink_pretrained.yaml
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from video_frame_utils import create_side_by_side, extract_frame, find_job_dirs

# Import shared code from training script (same as evaluate_vlm_overlay_regression_v2.py)
from sft_vlm_overlay_regression_v2 import (
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
    compute_failure_filter_stats,
)


# ==========================================================================
# ProgressLM prompting / parsing
# ==========================================================================

VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""
VISUAL_DEMO_INSTRUCTION_PART1 = """Here is the demonstration:"""


VISUAL_DEMO_INSTRUCTION_PART2 = """Here is the current state that you need to estimate:"""


VISUAL_DEMO_INSTRUCTION_PART3 = """Your task:
1. Check the current state image carefully.
2. Analyze the overall task goal and visual demonstration to understand how the task progresses from start to completion.
3. Identify the reference states from the visual demonstration that are most related to the current state image.
4. Compare the current state image with the chosen reference state, determining whether the image is behind or after the reference state.
5. Estimate the progress numerically as a floating-point value between 0% and 100%.

Your response **must** strictly follow this format:
<ref_think>Reason for choosing the most related state from the demonstration as the reference or explanation of why the current state image does not match the task goal or any steps from demonstration</ref_think>
<ref>which state from the visual demonstration is most related to the current state (output only the number of the state)</ref>
<score_think>Reason for comparing the current state image with the reference state</score_think>
<score>Your final estimated progress score</score>"""
NOTHINK_INSTRUCTION = (
    "Based on the task goal, demonstration, and current image, output ONLY "
    'the estimated progress as a percentage (0%–100%), or output exactly '
    '"n/a" if the target is incorrect, unmatched, or any abnormal condition '
    "exists; output nothing else."
)
_SCORE_RE = re.compile(r"<score>\s*(n/a|N/A|(?:\d+(?:\.\d+)?)%?)\s*</score>")
# Fallback regex for nothink mode: matches a bare percentage like "45%" or "45.5%", or "n/a"
_BARE_SCORE_RE = re.compile(r"^\s*(n/a|N/A|(?:\d+(?:\.\d+)?)%?)\s*$")


def msgs_to_qwen_conversation(msgs):
    """Convert flat msg list from build_visual_demo_prompt to Qwen2-VL conversation format."""
    system_text = msgs[0]["value"] if msgs and msgs[0]["type"] == "text" else ""
    user_content = []
    for msg in (msgs[1:] if system_text else msgs):
        if msg["type"] == "text":
            user_content.append({"type": "text", "text": msg["value"]})
        elif msg["type"] == "image":
            user_content.append({"type": "image", "image": msg["value"]})
    conversation = []
    if system_text:
        conversation.append({"role": "system", "content": system_text})
    conversation.append({"role": "user", "content": user_content})
    return conversation
def format_visual_demo_progress_shifts(total_steps: int) -> str:
    """
    Format progress shifts for visual demo images based on total_steps.

    The progress shifts between images: 0% -> 25% -> 50% -> 75% -> 100% (for total_steps=4)

    Args:
        total_steps: Total number of steps (not including the initial 0% state)

    Returns:
        Formatted string with <image> tags and progress scores

    Example:
        >>> format_visual_demo_progress_shifts(4)
        '<image> 0% <image> 25% <image> 50% <image> 75% <image> 100%'
    """
    # Number of images is total_steps + 1 (0% to 100%)
    num_images = total_steps + 1
    parts = []

    for i in range(num_images):
        # Calculate progress percentage for this image
        progress_percentage = round((i / total_steps) * 100)
        parts.append(f"<image> {progress_percentage}%")

    return " ".join(parts)

def build_eval_conversation_nothink(task_goal, visual_demo_paths, total_steps, stage_to_estimate):
    """
    Build a Qwen2-VL conversation that EXACTLY matches the training prompt format.

    Training format (from generate_progresslm_sft_data.py build_user_message_nothink):
        User message = single string with <image> tags inline:
            "{SYSTEM_PROMPT}\n\nThe overall task goal is {task_goal}.\n\n
             Here is the demonstration:\n<image> 0% <image> 25% ... <image> 100%\n\n
             Here is the current state that you need to estimate:\n<image>\n\n{INSTRUCTION}"

    LLaMA-Factory converts <image> tags → Qwen image content items.
    We reproduce that here by splitting on <image> and interleaving with image objects.

    Args:
        task_goal: Task description string
        visual_demo_paths: List of demo image paths (or PIL Images)
        total_steps: Number of transitions (= len(visual_demo_paths) - 1)
        stage_to_estimate: Path (or PIL Image) of the observation frame

    Returns:
        Qwen2-VL conversation list ready for processor.apply_chat_template()
    """
    # Step 1: Build the exact same text string as training
    progress_parts = []
    for i in range(total_steps + 1):
        pct = round((i / total_steps) * 100)
        progress_parts.append(f"<image> {pct}%")
    progress_str = " ".join(progress_parts)

    user_text = (
        f"{VISUAL_DEMO_SYSTEM_PROMPT}"
        f"\n\nThe overall task goal is {task_goal}."
        f"\n\nHere is the demonstration:"
        f"\n{progress_str}"
        f"\n\nHere is the current state that you need to estimate:"
        f"\n<image>"
        f"\n\n{NOTHINK_INSTRUCTION}"
    )

    # Step 2: Split on <image> and interleave with actual image objects
    # All images in order: demo_0, demo_1, ..., demo_N, observation
    all_images = list(visual_demo_paths) + [stage_to_estimate]

    segments = user_text.split("<image>")
    # segments[0] = everything before first <image>
    # segments[1] = " 0% " (text after first image, before second)
    # ...
    # segments[-1] = text after last <image>

    assert len(segments) == len(all_images) + 1, (
        f"Mismatch: {len(segments)-1} <image> tags but {len(all_images)} images"
    )

    user_content = []
    for i, segment in enumerate(segments):
        # Add text segment (may be empty string for back-to-back images)
        if segment:
            user_content.append({"type": "text", "text": segment})
        # Add image (except after the last segment)
        if i < len(all_images):
            img = all_images[i]
            if isinstance(img, str):
                user_content.append({"type": "image", "image": img})
            else:
                # PIL Image — save to temp file for Qwen processor
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                user_content.append({"type": "image", "image": tmp.name})

    # Step 3: Build conversation — NO system role (matches training)
    conversation = [
        {"role": "user", "content": user_content}
    ]

    return conversation


def build_visual_demo_prompt(
    task_goal,
    visual_demo_paths,
    total_steps,
    stage_to_estimate_path,
    min_pixels=None,
    max_pixels=None,
    nothink=False
):
    """
    Build a multi-part prompt for Visual Demo progress estimation task (inference mode).

    Prompt structure:
    0. Text: VISUAL_DEMO_SYSTEM_PROMPT
    1. Text: "The overall task goal is ..."
    2. Text: "Here is the demonstration:"
    3. Images: visual_demo (N images, variable length)
    4. Text: Progress shift information (e.g., "<image> 0% <image> 25% <image> 50% <image> 75% <image> 100%")
    5. Text: "Here is the current state that you need to estimate:"
    6. Image: stage_to_estimate (1 image)
    7. Text: Task instructions

    Args:
        task_goal: Task goal description
        visual_demo_paths: List of paths to demonstration images (variable length)
        total_steps: Total number of steps (not including the initial 0% state)
        stage_to_estimate_path: Path to the current state image
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        nothink: If True, use simplified instruction for direct score output

    Returns:
        List of message dicts for the model
    """
    msgs = []

    # Part 0: System prompt and task goal
    # msgs.append({"type": "text", "value": "You are a helpful assistant."})
    msgs.append({"type": "text", "value": VISUAL_DEMO_SYSTEM_PROMPT})
    msgs.append({"type": "text", "value": f"The overall task goal is {task_goal}"})

    # Part 1: Demonstration introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART1})

    # Part 2: Visual demo images (variable length)
    for demo_img_path in visual_demo_paths:
        img_msg = {"type": "image", "value": demo_img_path}
        if min_pixels is not None:
            img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_msg["max_pixels"] = max_pixels
        msgs.append(img_msg)

    # Part 3: Progress shift information
    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    msgs.append({"type": "text", "value": f"The progress shifts across all given visual demos is: {progress_shifts}"})

    # Part 4: Current state introduction
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART2})

    # Part 5: Current state image (single image)
    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    # Part 6: Task instructions (use nothink version if specified)
    instruction = VISUAL_DEMO_INSTRUCTION_PART3_NOTHINK if nothink else VISUAL_DEMO_INSTRUCTION_PART3
    msgs.append({"type": "text", "value": instruction})

    return msgs


def extract_progresslm_score(text: str) -> Optional[float]:
    """Parse <score>NUMBER</score> (0..100) or bare percentage or n/a. Return None on n/a / parse fail."""
    m = _SCORE_RE.search(text)
    if m is None:
        # Fallback: try bare percentage (nothink mode outputs e.g. "45%")
        m = _BARE_SCORE_RE.search(text.strip())
    if m is None:
        return None
    s = m.group(1).strip().lower().rstrip("%")
    if s == "n/a":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return max(0.0, min(100.0, v))


# ==========================================================================
# Inference
# ==========================================================================


def run_inference_progresslm(model, processor, pairs, visual_demo_paths, batch_size, device, max_new_tokens=30000, nothink=False):
    """
    ProgressLM baseline:
      - score each frame independently: s1, s2 in [0,100]
      - prediction = (s2 - s1) clipped to [-100,100]

    Returns results dicts compatible with compute_metrics/visualize_results.
    """
    results = []
    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc="Evaluating (ProgressLM)"):
        batch = pairs[b * batch_size : (b + 1) * batch_size]

        texts = []
        images_list = []
        meta = []  # list of (item, which_frame)

        for item in batch:
            task_token = item["task_token"]
            # Prefer human-readable task string if available; fallback to token.
            task_goal = TASK_TOKENS.get(task_token, task_token)
            # user_prompt = PROGRESSLM_USER_TEMPLATE.format(task_goal=task_goal)

            # Load frames
            try:
                img1 = extract_frame(item["video_path_1"], item["frame_idx_1"]).convert("RGB")
            except Exception as e:
                print(f"Warning: failed to load frame1: {e}")
                img1 = Image.new("RGB", (256, 256), (128, 128, 128))

            try:
                img2 = extract_frame(item["video_path_2"], item["frame_idx_2"]).convert("RGB")
            except Exception as e:
                print(f"Warning: failed to load frame2: {e}")
                img2 = Image.new("RGB", (256, 256), (128, 128, 128))
            # Two independent conversations (one per frame)
            if nothink:
                conv1 = build_eval_conversation_nothink(
                    task_goal, visual_demo_paths, len(visual_demo_paths) - 1, img1
                )
                conv2 = build_eval_conversation_nothink(
                    task_goal, visual_demo_paths, len(visual_demo_paths) - 1, img2
                )
            else:
                conv1 = msgs_to_qwen_conversation(
                    build_visual_demo_prompt(task_goal, visual_demo_paths, len(visual_demo_paths) - 1, img1)
                )
                conv2 = msgs_to_qwen_conversation(
                    build_visual_demo_prompt(task_goal, visual_demo_paths, len(visual_demo_paths) - 1, img2)
                )

            for which, conv, img in [(1, conv1, img1), (2, conv2, img2)]:
                try:
                    import qwen_vl_utils
                    image_input, _ = qwen_vl_utils.process_vision_info(conv)
                except ImportError:
                    image_input = [img]
                except Exception as e:
                    print(f"Warning: process_vision_info failed: {e}, falling back to single image")
                    image_input = [img]

                text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                images_list.append(image_input)
                meta.append((item, which))
        flat_images = [img for sublist in images_list for img in sublist]
        inputs = processor(
            text=texts,
            images=flat_images if any(img is not None for img in flat_images) else None,
            return_tensors="pt",
            padding=True,
            # truncation=True,
            # max_new_tokens=2048,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        # FIX #4: decode per-sequence using actual input length, not padded batch length
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]
        completions = processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Collect per-item frame scores
        per_item_scores = {}  # id(item) -> {1: (score, raw), 2: (score, raw)}
        for i, (item, which) in enumerate(meta):
            completion = completions[i].strip()  # ← use the correctly trimmed version
            score = extract_progresslm_score(completion)

            key = id(item)
            if key not in per_item_scores:
                per_item_scores[key] = {1: (None, ""), 2: (None, "")}
            per_item_scores[key][which] = (score, completion)

        # Convert to pair-level results
        for item in batch:
            s1, raw1 = per_item_scores[id(item)][1]
            s2, raw2 = per_item_scores[id(item)][2]
            gt = float(item["correct_answer"])
            if (s1 is None) or (s2 is None):
                pred = None
                error = None
                abs_error = None
                sign_correct = None
                raw_completion = f"frame1={raw1} || frame2={raw2}"
            else:
                pred = float(s2 - s1)
                pred = max(-100.0, min(100.0, pred))
                error = pred - gt
                abs_error = abs(error)
                sign_correct = (np.sign(gt) == np.sign(pred)) if gt != 0 else (pred == 0)
                raw_completion = f"s1={s1:.2f} ({raw1}) || s2={s2:.2f} ({raw2})"

            results.append({
                "ground_truth": gt,
                "prediction": pred,
                "raw_completion": raw_completion,
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
                "progresslm_score_1": s1,
                "progresslm_score_2": s2,
            })

    return results


# ==========================================================================
# Metrics & Visualization (copied from evaluate_vlm_overlay_regression_v2.py)
# ==========================================================================


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


def visualize_results(results, output_dir, num_examples=8):
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
    fig.suptitle(f"Evaluation Results (n={len(valid)})", fontsize=14, fontweight="bold")

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
        def _sort_key(label):
            if label == "sf":
                return (0, 0)
            return (1, int(label.split("-")[1]))

        sorted_labels = sorted(interval_groups.keys(), key=_sort_key)
        accuracies = [np.mean([r["sign_correct"] for r in interval_groups[l]]) * 100 for l in sorted_labels]
        counts = [len(interval_groups[l]) for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(max(6, len(sorted_labels) * 1.5), 5))
        bars = ax.bar(range(len(sorted_labels)), accuracies,
                      color=["#e74c3c" if l == "sf" else "#3498db" for l in sorted_labels],
                      edgecolor="black", alpha=0.85)

        for bar, acc, cnt in zip(bars, accuracies, counts):
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

    # --- Plot 3: Sample overlays with predictions ---
    worst = sorted(valid, key=lambda x: x["abs_error"], reverse=True)[:num_examples]
    best = sorted(valid, key=lambda x: x["abs_error"])[:num_examples]

    for label, samples in [("worst", worst), ("best", best)]:
        n = len(samples)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes_grid = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes_grid = np.array([[axes_grid]])
        elif rows == 1:
            axes_grid = axes_grid.reshape(1, -1)
        elif cols == 1:
            axes_grid = axes_grid.reshape(-1, 1)

        for idx, r in enumerate(samples):
            row, col = idx // cols, idx % cols
            ax = axes_grid[row, col]
            try:
                f1 = extract_frame(r["video_path_1"], r["frame_idx_1"])
                f2 = extract_frame(r["video_path_2"], r["frame_idx_2"])
                overlay = create_side_by_side(f1, f2)
                ax.imshow(overlay)
            except Exception:
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")
            color = "green" if r["abs_error"] <= 3 else "red"
            rc = r["raw_completion"]
            ax.set_title(
                f"GT: {r['ground_truth']:.0f} | Pred: {r['prediction']:.1f} | Err: {r['abs_error']:.1f}\n"
                f"{r['demo_type']} | Raw: {rc[:45]}",
                fontsize=9, color=color, fontweight="bold",
            )

        for idx in range(len(samples), rows * cols):
            axes_grid[idx // cols, idx % cols].axis("off")

        plt.suptitle(f"{label.capitalize()} Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"samples_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


# ==========================================================================
# Main (largely identical to evaluate_vlm_overlay_regression_v2.py)
# ==========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ProgressLM as a fine-tuned progress baseline")

    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Raymond-Qiancx/ProgressLM-3B-SFT",
        help="ProgressLM checkpoint (e.g., Raymond-Qiancx/ProgressLM-3B-SFT or -RL)",
    )

    # Option B: CSV list
    p.add_argument(
        "--base_dataset_path",
        type=str,
        required=True,
        help="Dataset path OR comma-separated list of dataset paths",
    )
    p.add_argument("--split", type=str, default="val", help="train | val | integer (number of job dirs to use)")
    p.add_argument("--train_val_split_index", type=int, default=5, help="Last N job dirs used for val split")
    p.add_argument("--compare_interval", type=str, default="4,8,12,16", help="Comma-separated frame intervals")
    p.add_argument("--sample_interval", type=int, default=5, help="Frame sampling stride when building pairs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=None, help="Limit number of eval pairs (random subset)")
    p.add_argument("--tolerance", type=int, default=3, help="Absolute error tolerance for within-tolerance metric")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    p.add_argument("--num_visualize", type=int, default=8, help="Number of examples to show in best/worst plots")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to <model_path>/<prefix>eval_<timestamp>_<intervals>)",
    )
    p.add_argument("--prefix", type=str, default="", help="Prefix for output directory")
    p.add_argument("--nothink", action='store_true')

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
    name = Path(dataset_path).name.strip() or "dataset"
    return name.replace(" ", "_")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_intervals = [int(x.strip()) for x in args.compare_interval.split(",")]

    dataset_paths = _split_dataset_paths(args.base_dataset_path)
    if not dataset_paths:
        raise ValueError("--base_dataset_path is empty after parsing. Provide a path or CSV list.")

    root_output_dir = args.output_dir or os.path.join(
        args.model_name_or_path,
        f"{args.prefix}eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(map(str, compare_intervals))}",
    )
    os.makedirs(root_output_dir, exist_ok=True)

    print("Loading model...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    dtype = dtype_map.get(args.dtype, "auto")

    # ProgressLM is a full model (not a PEFT adapter) in typical usage.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype if dtype != "auto" else None,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    print("Model loaded")

    all_metrics = {}
    all_summaries = []

    for base_dataset_path in dataset_paths:
        print("\n" + "#" * 80)
        print(f"Evaluating dataset: {base_dataset_path}")
        print("#" * 80)

        dataset_tag = _dataset_output_tag(base_dataset_path)
        output_dir = os.path.join(root_output_dir, dataset_tag)
        os.makedirs(output_dir, exist_ok=True)

        print("Loading dataset...")
        job_dirs = find_job_dirs(base_dataset_path)

        if "robocasa/datasets" in base_dataset_path:
            job_dirs = job_dirs[:]
        elif args.split == "val":
            job_dirs = job_dirs[:] if args.train_val_split_index == 0 else job_dirs[-args.train_val_split_index :]
        elif args.split == "train":
            job_dirs = job_dirs[:] if args.train_val_split_index == 0 else job_dirs[: -args.train_val_split_index]
        else:
            job_dirs = job_dirs[: int(args.split)]

        print(f"Using {len(job_dirs)} job directories for '{args.split}' split")

        success_data, unfiltered_failure_data = load_trajectories(job_dirs)
        print(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

        failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
        print(f"Matched {len(failure_data)} failures to success trajectories")

        # Balanced sampling per demo
        if success_data and failure_data:
            success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(success_data, failure_data, 50)
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

            print(f"After per-demo cap (50): {len(success_data)} -> {len(capped_success)} success, {len(failure_data)} -> {len(capped_failure)} failure")
            success_data = capped_success
            failure_data = capped_failure

        # Load cached failure filter stats (same filter used during training)
        if failure_data:
            success_mean_diffs_at_idx = None
            stats_cache_file = Path(base_dataset_path) / "failure_filter_stats.json"
            if stats_cache_file.exists():
                print(f"Loading failure filter stats from {stats_cache_file}")
                with open(stats_cache_file, "r") as f:
                    cached = json.load(f)
                    success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]
                print("  Will apply same failure-frame filter as training to match train/eval distribution")
            else:
                print("No cached stats found, computing failure filter stats...")
                success_mean_diffs_at_idx = compute_failure_filter_stats(success_by_demo, failure_by_demo, 0, 1)
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
        # Load visual demo frames from first success trajectory
        num_demo_frames = 5
        if success_data:
            demo_traj = success_data[0]
            demo_video_path = demo_traj["video_path"]
            demo_last_frame = demo_traj["trajectory_index"]
            frame_indices = [int(i * demo_last_frame / (num_demo_frames - 1)) for i in range(num_demo_frames)]
            import tempfile
            demo_dir = tempfile.mkdtemp(prefix="visual_demo_")
            visual_demo_paths = []
            for idx, fi in enumerate(frame_indices):
                frame = extract_frame(demo_video_path, fi).convert("RGB")
                path = os.path.join(demo_dir, f"demo_{idx:03d}.png")
                frame.save(path)
                visual_demo_paths.append(path)
            print(f"Loaded {len(visual_demo_paths)} visual demo frames from {demo_video_path} (frames: {frame_indices})")
        else:
            visual_demo_paths = []
            print("Warning: No success data available for visual demo")

        # ---- Run inference ----
        results = run_inference_progresslm(model, processor, pairs, visual_demo_paths, args.batch_size, device, nothink=args.nothink)

        # ---- Compute metrics ----
        metrics = compute_metrics(results, tolerance=args.tolerance)
        all_metrics[base_dataset_path] = metrics

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
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

        # Per demo_type breakdown
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
            for label, stats in sorted(
                interval_breakdown.items(),
                key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1])),
            ):
                print(f"  {label:>10s}: n={stats['count']:>4d}, sign_acc={stats['sign_accuracy']:.3f}, MAE={stats['mae']:.3f}")

        print("=" * 60)

        summary = {
            "args": {**vars(args), "base_dataset_path": base_dataset_path},
            "metrics": metrics,
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

        if args.visualize:
            visualize_results(results, output_dir, num_examples=args.num_visualize)

        print(f"\nDataset outputs saved to {output_dir}")

    combined_metrics_path = os.path.join(root_output_dir, "all_metrics.json")
    with open(combined_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {combined_metrics_path}")

    combined_summaries_path = os.path.join(root_output_dir, "all_summaries.json")
    with open(combined_summaries_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All summaries saved to {combined_summaries_path}")

    print(f"\nAll outputs saved to {root_output_dir}")


if __name__ == "__main__":
    main()
