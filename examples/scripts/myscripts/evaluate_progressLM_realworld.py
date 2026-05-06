"""
Evaluate ProgressLM (SFT/RL) on realworld dataset pairwise progress-comparison benchmark.

Uses the same realworld data-loading and pair-construction code as
`evaluate_vlm_overlay_regression_realworld.py`, but swaps inference to ProgressLM:

  1) Score each frame independently with ProgressLM: s in [0, 100]
  2) Predict pairwise progress delta: pred = s2 - s1 (clipped to [-100, 100])

So the existing metrics (sign accuracy, MAE, interval breakdown, viz) work unchanged.

Usage:
BASE_DATASET_PATHS=(
    "realworld_dataset/BimanualBikeRotorInstall-Nominal-real"
    "realworld_dataset/BimanualClearKitchenCounter-Nominal-real"
    "realworld_dataset/BimanualSetUpBreakfastTable-Nominal-real"
    "realworld_dataset/CleanLitterBox-Nominal-real"
    "realworld_dataset/CutAppleIntoSlices-Nominal-real"
    "realworld_dataset/PushCoasterToMug-Nominal-real"
    "realworld_dataset/PushCoasterToMug-ObjectCentricDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTable-StationDistributionShift-real"
    "realworld_dataset/PutKiwiInCenterOfTableSeenTasks_backfill-salem--video"
    "realworld_dataset/TurnMugRightsideUp-Nominal-real"
    "realworld_dataset/TurnMugRightsideUp-ObjectCentricDistributionShift-real"
    "realworld_dataset/TurnMugRightsideUp-StationDistributionShift-real"
)
# Single dataset
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_progressLM_realworld.py \
  --model_name_or_path /workspace/hf_trl/trl/ProgressLM/qwen25vl_7b_nothink_multitask_merged \
  --base_dataset_path "realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real" \
  --split val \
  --compare_interval 32,100 \
  --batch_size 10 \
  --num_samples 10 \
  --visualize --nothink

# Multiple datasets
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_progressLM_realworld.py \
  --model_name_or_path /workspace/hf_trl/trl/ProgressLM/models_archive/qwen25vl_7b_nothink_multitask_realworld_merged_820 \
  --base_dataset_path "realworld_dataset/BimanualBikeRotorInstall-Nominal-real,\
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
  --split val \
  --compare_interval 32 \
  --batch_size 8 \
  --num_samples 100 \
  --visualize --nothink
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train our_scripts/qwen2_5vl_lora_sft_small_nothink_realworld.yaml

"""

import argparse
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from video_frame_utils import create_side_by_side, extract_frame

# Import realworld data loading
from sft_vlm_overlay_regression_realworld_data import (
    TASK_TOKENS,
    is_realworld_format,
    load_realworld_trajectories,
    match_realworld_failures_to_successes,
    compute_realworld_failure_filter_stats,
    build_realworld_frame_pairs,
)


# ==========================================================================
# ProgressLM prompting / parsing (from evaluate_progressLM.py)
# ==========================================================================

VISUAL_DEMO_SYSTEM_PROMPT = """You are a progress estimator that evaluates the progress of the current state during an ongoing task based on a visual demonstration. The demonstration consists of a sequence of vision-based states and their corresponding progress value (ranging from 0% to 100%), showing how the task evolves from start to completion."""

NOTHINK_INSTRUCTION = (
    "Based on the task goal, demonstration, and current image, output ONLY "
    'the estimated progress as a percentage (0%–100%), or output exactly '
    '"n/a" if the target is incorrect, unmatched, or any abnormal condition '
    "exists; output nothing else."
)

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

_SCORE_RE = re.compile(r"<score>\s*(n/a|N/A|(?:\d+(?:\.\d+)?)%?)\s*</score>")
_BARE_SCORE_RE = re.compile(r"^\s*(n/a|N/A|(?:\d+(?:\.\d+)?)%?)\s*$")


def format_visual_demo_progress_shifts(total_steps: int) -> str:
    num_images = total_steps + 1
    parts = []
    for i in range(num_images):
        progress_percentage = round((i / total_steps) * 100)
        parts.append(f"<image> {progress_percentage}%")
    return " ".join(parts)


def build_eval_conversation_nothink(task_goal, visual_demo_paths, total_steps, stage_to_estimate):
    """
    Build a Qwen2-VL conversation that EXACTLY matches the training prompt format.

    Splits on <image> tags and interleaves with actual image objects,
    matching how LLaMA-Factory converts <image> tags during training.
    """
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

    all_images = list(visual_demo_paths) + [stage_to_estimate]
    segments = user_text.split("<image>")

    assert len(segments) == len(all_images) + 1, (
        f"Mismatch: {len(segments)-1} <image> tags but {len(all_images)} images"
    )

    user_content = []
    for i, segment in enumerate(segments):
        if segment:
            user_content.append({"type": "text", "text": segment})
        if i < len(all_images):
            img = all_images[i]
            if isinstance(img, str):
                user_content.append({"type": "image", "image": img})
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                user_content.append({"type": "image", "image": tmp.name})

    conversation = [
        {"role": "user", "content": user_content}
    ]
    return conversation


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


def build_visual_demo_prompt(
    task_goal, visual_demo_paths, total_steps, stage_to_estimate_path,
    min_pixels=None, max_pixels=None, nothink=False
):
    msgs = []
    msgs.append({"type": "text", "value": VISUAL_DEMO_SYSTEM_PROMPT})
    msgs.append({"type": "text", "value": f"The overall task goal is {task_goal}"})
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART1})

    for demo_img_path in visual_demo_paths:
        img_msg = {"type": "image", "value": demo_img_path}
        if min_pixels is not None:
            img_msg["min_pixels"] = min_pixels
        if max_pixels is not None:
            img_msg["max_pixels"] = max_pixels
        msgs.append(img_msg)

    progress_shifts = format_visual_demo_progress_shifts(total_steps)
    msgs.append({"type": "text", "value": f"The progress shifts across all given visual demos is: {progress_shifts}"})
    msgs.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART2})

    stage_img_msg = {"type": "image", "value": stage_to_estimate_path}
    if min_pixels is not None:
        stage_img_msg["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_img_msg["max_pixels"] = max_pixels
    msgs.append(stage_img_msg)

    instruction = NOTHINK_INSTRUCTION if nothink else VISUAL_DEMO_INSTRUCTION_PART3
    msgs.append({"type": "text", "value": instruction})
    return msgs


def extract_progresslm_score(text: str) -> Optional[float]:
    """Parse <score>NUMBER</score> or bare percentage or n/a. Return None on failure."""
    m = _SCORE_RE.search(text)
    if m is None:
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
# Visual demo extraction for realworld data
# ==========================================================================

def extract_visual_demos_per_task(trajectories_by_task, num_demo_frames=5):
    """
    For each task, extract evenly-spaced demo frames from the first success trajectory.

    Realworld frames are 1-based (frame_000001.png to frame_NNNNNN.png).

    Args:
        trajectories_by_task: dict mapping task_token -> list of trajectories
        num_demo_frames: number of demo frames to extract per task

    Returns:
        dict mapping task_token -> list of saved demo frame file paths
    """
    visual_demos = {}
    demo_dir = tempfile.mkdtemp(prefix="progresslm_realworld_demo_")

    for task_token, trajs in trajectories_by_task.items():
        success_trajs = [t for t in trajs if t["sf"] == "success"]
        if not success_trajs:
            print(f"Warning: No success trajectories for task {task_token}, skipping demo extraction")
            continue

        demo_traj = success_trajs[0]
        frames_dir = demo_traj["frames_dir"]
        num_frames = demo_traj["num_frames"]
        first_frame = 1  # realworld frames are 1-based

        # Evenly-spaced frame indices from 1 to num_frames
        span = num_frames - first_frame
        frame_indices = [
            first_frame + int(i * span / (num_demo_frames - 1))
            for i in range(num_demo_frames)
        ]

        paths = []
        safe_token = task_token.replace("/", "_").replace("\\", "_").replace(" ", "_")
        task_demo_dir = os.path.join(demo_dir, safe_token)
        os.makedirs(task_demo_dir, exist_ok=True)

        ok = True
        for idx, fi in enumerate(frame_indices):
            try:
                frame = extract_frame(frames_dir, fi).convert("RGB")
                path = os.path.join(task_demo_dir, f"demo_{idx:03d}.png")
                frame.save(path)
                paths.append(path)
            except Exception as e:
                print(f"Warning: failed to extract demo frame {fi} from {frames_dir}: {e}")
                ok = False
                break

        if ok:
            visual_demos[task_token] = paths
            print(f"  Extracted {len(paths)} demo frames for {task_token} from {frames_dir} (frames: {frame_indices})")
        else:
            print(f"  Failed to extract demo frames for {task_token}")

    return visual_demos


# ==========================================================================
# Inference
# ==========================================================================

def run_inference_progresslm(model, processor, pairs, visual_demos_by_task, batch_size, device, max_new_tokens=30000, nothink=False):
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
            task_goal = task_token  # Already a token like "[PUT_KIWI_IN_CENTER_OF_TABLE]"

            visual_demo_paths = visual_demos_by_task.get(task_token, [])
            if not visual_demo_paths:
                print(f"Warning: No visual demo for task {task_token}, skipping pair")
                continue

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

        if not texts:
            continue

        flat_images = [img for sublist in images_list for img in sublist]
        inputs = processor(
            text=texts,
            images=flat_images if any(img is not None for img in flat_images) else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

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
        per_item_scores = {}
        for i, (item, which) in enumerate(meta):
            completion = completions[i].strip()
            score = extract_progresslm_score(completion)

            key = id(item)
            if key not in per_item_scores:
                per_item_scores[key] = {1: (None, ""), 2: (None, "")}
            per_item_scores[key][which] = (score, completion)

        # Convert to pair-level results
        for item in batch:
            key = id(item)
            if key not in per_item_scores:
                continue
            s1, raw1 = per_item_scores[key][1]
            s2, raw2 = per_item_scores[key][2]
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
                "compare_interval": item.get("compare_interval", "?"),
                "frame_idx_1": item["frame_idx_1"],
                "frame_idx_2": item["frame_idx_2"],
                "video_path_1": item["video_path_1"],
                "video_path_2": item["video_path_2"],
                "progresslm_score_1": s1,
                "progresslm_score_2": s2,
            })

    return results


# ==========================================================================
# Metrics & Visualization
# ==========================================================================

def _get_interval_label(r):
    """Classify a result as 'intra-N' or 'sf'."""
    interval = r.get("compare_interval", "?")
    if interval and interval != "?" and interval != 0:
        return f"intra-{interval}"
    v1 = r.get("video_path_1")
    v2 = r.get("video_path_2")
    f1 = r.get("frame_idx_1")
    f2 = r.get("frame_idx_2")
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
    fig.suptitle(f"ProgressLM Realworld Evaluation (n={len(valid)})", fontsize=14, fontweight="bold")

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
            try:
                return (1, int(label.split("-")[1]))
            except (IndexError, ValueError):
                return (2, 0)

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
            interval = r.get("compare_interval", "?")
            interval_str = f"int={interval}" if interval else "sf"
            ax.set_title(
                f"GT: {r['ground_truth']:.0f} | Pred: {r['prediction']:.1f} | Err: {r['abs_error']:.1f}\n"
                f"{r['demo_type']} | {interval_str} | Raw: {rc[:45]}",
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
# Main
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ProgressLM on realworld datasets")

    p.add_argument("--model_name_or_path", type=str,
                   default="Raymond-Qiancx/ProgressLM-3B-SFT",
                   help="ProgressLM checkpoint path")
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Dataset path OR comma-separated list of dataset paths")
    p.add_argument("--split", type=str, default="val",
                   help="train | val (episode-level split)")
    p.add_argument("--compare_interval", type=str, default="32,36",
                   help="Comma-separated frame intervals for success pair comparisons")
    p.add_argument("--sample_interval", type=int, default=5,
                   help="Frame sampling stride when building pairs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of eval pairs (random subset)")
    p.add_argument("--tolerance", type=int, default=3,
                   help="Absolute error tolerance for within-tolerance metric")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    p.add_argument("--num_visualize", type=int, default=8,
                   help="Number of examples to show in best/worst plots")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (defaults to <model_path>/eval_progresslm_realworld_<timestamp>)")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output directory")
    p.add_argument("--nothink", action="store_true",
                   help="Use nothink mode (direct percentage output, no CoT)")
    p.add_argument("--num_demo_frames", type=int, default=5,
                   help="Number of visual demo frames per task")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_intervals = [int(x.strip()) for x in args.compare_interval.split(",")]

    dataset_paths = [p.strip() for p in args.base_dataset_path.split(",") if p.strip()]
    if not dataset_paths:
        raise ValueError("--base_dataset_path is empty")

    root_output_dir = args.output_dir or os.path.join(
        args.model_name_or_path,
        f"{args.prefix}eval_progresslm_realworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(map(str, compare_intervals))}"
    )
    os.makedirs(root_output_dir, exist_ok=True)

    # ============================
    # Build ALL pairs across all tasks first (matching training script)
    # ============================
    all_pairs = []
    trajectories_by_task = {}  # task_token -> list of raw trajectories (for demo extraction)

    for base_dataset_path in dataset_paths:
        print("\n" + "#" * 80)
        print(f"Loading dataset: {base_dataset_path}")
        print("#" * 80)

        if not is_realworld_format(base_dataset_path):
            print(f"WARNING: {base_dataset_path} does not appear to be a realworld dataset format. Skipping.")
            continue

        trajectories = load_realworld_trajectories(base_dataset_path)
        print(f"Found {len(trajectories)} total trajectories")

        if not trajectories:
            continue

        success_trajs = [t for t in trajectories if t["sf"] == "success"]
        failure_trajs = [t for t in trajectories if t["sf"] == "fail"]
        print(f"  {len(success_trajs)} success + {len(failure_trajs)} failure trajectories")

        matched_failures = match_realworld_failures_to_successes(trajectories)
        print(f"  Matched {len(matched_failures)} failures to successes")

        if matched_failures:
            success_mean_diffs_at_idx = compute_realworld_failure_filter_stats(
                trajectories, local_rank=0, world_size=1
            )
        else:
            success_mean_diffs_at_idx = {}

        task_name = Path(base_dataset_path).name
        task_pairs = build_realworld_frame_pairs(
            success_data=success_trajs,
            failure_data=matched_failures,
            compare_intervals=compare_intervals,
            train_sample_interval=args.sample_interval,
            success_mean_diffs_at_idx=success_mean_diffs_at_idx,
            task_name=task_name,
            local_rank=0,
            world_size=1,
        )
        print(f"Built {len(task_pairs)} pairs for {task_name}")
        all_pairs.extend(task_pairs)

        # Store trajectories for demo extraction, keyed by the task_token that pairs use
        if task_pairs:
            task_token = task_pairs[0]["task_token"]
            trajectories_by_task[task_token] = trajectories

    print(f"\nTotal pairs across all tasks: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs built. Exiting.")
        return

    # ============================
    # Episode-level split matching training script EXACTLY
    # ============================
    all_demo_ids = sorted(set(item["demo_id"] for item in all_pairs))
    rng = random.Random(42)
    rng.shuffle(all_demo_ids)
    n_eval_demos = max(1, int(len(all_demo_ids) * 0.1))
    eval_demo_ids = set(all_demo_ids[:n_eval_demos])
    train_demo_ids = set(all_demo_ids[n_eval_demos:])
    print(f"Episode-level split: {len(train_demo_ids)} train episodes, {len(eval_demo_ids)} eval episodes")
    print(f"Eval episodes: {eval_demo_ids}")

    if args.split == "val":
        keep_ids = eval_demo_ids
    elif args.split == "train":
        keep_ids = train_demo_ids
    else:
        keep_ids = set(all_demo_ids)

    pairs = [p for p in all_pairs if p["demo_id"] in keep_ids]
    print(f"Using {len(pairs)} pairs for '{args.split}' split")
    del all_pairs

    if args.num_samples and args.num_samples < len(pairs):
        subsample_rng = random.Random(args.seed)
        pairs = subsample_rng.sample(pairs, args.num_samples)
        print(f"Subsampled to {len(pairs)} pairs")

    if not pairs:
        print("No pairs after split. Exiting.")
        return

    # ============================
    # Extract visual demo frames per task
    # ============================
    print("\nExtracting visual demo frames per task...")
    visual_demos_by_task = extract_visual_demos_per_task(
        trajectories_by_task, num_demo_frames=args.num_demo_frames
    )
    print(f"Visual demos available for {len(visual_demos_by_task)} tasks")

    # ============================
    # Load model
    # ============================
    print("\nLoading model...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    dtype = dtype_map.get(args.dtype, "auto")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype if dtype != "auto" else None,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.eval()
    print("Model loaded")

    # ============================
    # Run inference on all pairs
    # ============================
    results = run_inference_progresslm(
        model, processor, pairs, visual_demos_by_task,
        args.batch_size, device, nothink=args.nothink,
    )

    # ============================
    # Compute overall metrics
    # ============================
    all_metrics = {}

    metrics = compute_metrics(results, tolerance=args.tolerance)
    all_metrics["overall"] = metrics

    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS (ProgressLM)")
    print("=" * 60)
    print(f"Model:            {args.model_name_or_path}")
    print(f"Datasets:         {', '.join(dataset_paths)}")
    print(f"Split:            {args.split}")
    print(f"Nothink:          {args.nothink}")
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
            name = key.replace("_mae", "")
            count = metrics.get(f"{name}_count", 0)
            mae = metrics[key]
            sign_acc = metrics.get(f"{name}_sign_accuracy", float("nan"))
            print(f"  {name}: n={count}, MAE={mae:.3f}, sign_acc={sign_acc:.3f}")

    # Per-interval breakdown
    interval_breakdown = metrics.get("interval_breakdown", {})
    if interval_breakdown:
        print("-" * 60)
        print("Breakdown by compare interval:")
        for label, stats in sorted(
            interval_breakdown.items(),
            key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1]) if "-" in x[0] else 0),
        ):
            print(f"  {label:>10s}: n={stats['count']:>4d}, sign_acc={stats['sign_accuracy']:.3f}, MAE={stats['mae']:.3f}")

    print("=" * 60)

    # ============================
    # Per-task metrics
    # ============================
    task_tokens_in_results = set(r["task_token"] for r in results)
    for task_token in sorted(task_tokens_in_results):
        task_results = [r for r in results if r["task_token"] == task_token]
        task_metrics = compute_metrics(task_results, tolerance=args.tolerance)
        all_metrics[task_token] = task_metrics

        print(f"\n  Task: {task_token} (n={len(task_results)})")
        print(f"    MAE: {task_metrics.get('mae', float('nan')):.3f}, "
              f"Sign acc: {task_metrics.get('sign_accuracy', float('nan')):.3f}")

        task_interval = task_metrics.get("interval_breakdown", {})
        for lbl, st in sorted(
            task_interval.items(),
            key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1]) if "-" in x[0] else 0),
        ):
            print(f"      {lbl:>10s}: n={st['count']:>4d}, sign_acc={st['sign_accuracy']:.3f}")

    # ============================
    # Save results
    # ============================
    summary = {
        "args": vars(args),
        "datasets": dataset_paths,
        "metrics": metrics,
        "per_task_metrics": {k: v for k, v in all_metrics.items() if k != "overall"},
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(root_output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    completions_path = os.path.join(root_output_dir, "completions.json")
    with open(completions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Completions saved to {completions_path}")

    # ============================
    # Visualize
    # ============================
    if args.visualize:
        visualize_results(results, root_output_dir, num_examples=args.num_visualize)

    print(f"\nAll outputs saved to {root_output_dir}")


if __name__ == "__main__":
    main()
