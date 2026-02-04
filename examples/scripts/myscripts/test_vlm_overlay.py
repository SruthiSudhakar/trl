"""
Test a trained VLM on overlay images and visualize results.

Usage:
python3 examples/scripts/myscripts/test_vlm_overlay.py \
    --model_path outputs/jan29/PnPAll_20260129_222205/checkpoint-9500 \
    --base_dataset_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPStoveToCounter \
    --output_dir outputs/test_results \
    --num_samples 20 \
    --split train

# Test a specific demo:
python3 examples/scripts/myscripts/test_vlm_overlay.py \
    --model_path outputs/jan29/PnPAll_20260129_222205/checkpoint-9500 \
    --base_dataset_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPStoveToCounter \
    --output_dir outputs/test_results \
    --demo_id 10_10_9h61nlai \
    --split train

# List available demos:
python3 examples/scripts/myscripts/test_vlm_overlay.py \
    --base_dataset_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPStoveToCounter \
    --list_demos \
    --split train
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from video_frame_utils import create_side_by_side, extract_frame, find_job_dirs, read_s3_json

# Import data loading functions from the training script
from sft_vlm_overlay_regression_v2 import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
)

random.seed(42)


def load_model_and_processor(model_path):
    """Load the trained model and processor."""
    print(f"Loading model from {model_path}...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Model loaded on {model.device}")
    return model, processor


def run_inference(model, processor, overlay_image, task_token):
    """Run inference on a single overlay image."""
    user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # Process with qwen_vl_utils if available
    try:
        import qwen_vl_utils
        image_input, _ = qwen_vl_utils.process_vision_info(messages)
    except (ImportError, Exception):
        image_input = [overlay_image]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=image_input,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only the generated tokens
    generated = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    # Try to parse as number
    try:
        prediction = float(generated.split()[0].replace(",", ""))
    except (ValueError, IndexError):
        prediction = None

    return generated, prediction


def build_test_samples(base_dataset_path, num_samples=20, split="val", demo_id=None, list_demos=False):
    """Build test samples from the dataset.

    Args:
        base_dataset_path: Path to the dataset
        num_samples: Number of samples to return (ignored if demo_id is specified)
        split: "train" or "val"
        demo_id: If specified, only return samples from this demo (can be partial match)
        list_demos: If True, just list available demos and return None
    """
    print(f"Loading test samples from {base_dataset_path}...")

    job_dirs = find_job_dirs(base_dataset_path)

    # Use last 5 job dirs for validation
    if split == "val":
        job_dirs = job_dirs[-5:]
    else:
        job_dirs = job_dirs[:-5]

    if len(job_dirs) == 0:
        raise ValueError(f"No job directories found in {base_dataset_path}")

    print(f"Found {len(job_dirs)} job directories for {split} split")

    success_data, unfiltered_failure_data = load_trajectories(job_dirs)
    print(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

    failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
    print(f"Matched {len(failure_data)} failures to success trajectories")

    success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(
        success_data, failure_data, max_per_demo=10
    )

    # Build frame pairs (simplified - no filtering stats)
    job_name = Path(job_dirs[0]).name if job_dirs else ""
    combined_data = build_frame_pairs(
        success_data=success_data,
        failure_data=failure_data,
        compare_intervals=[4, 8, 12, 16],
        train_sample_interval=5,
        success_mean_diffs_at_idx={},  # Skip filtering for test
        job_name=job_name,
        local_rank=0,
        world_size=1,
    )

    print(f"Built {len(combined_data)} frame pairs")

    # Get unique demos for listing or filtering
    demos_by_id = defaultdict(list)
    for item in combined_data:
        demo_key = item.get("demo_id_exact", item.get("demo_id", "unknown"))
        demos_by_id[demo_key].append(item)

    # If listing demos, print them and return
    if list_demos:
        print("\n" + "=" * 60)
        print("AVAILABLE DEMOS")
        print("=" * 60)
        for demo_key in sorted(demos_by_id.keys()):
            items = demos_by_id[demo_key]
            demo_type = items[0].get("demo_success", "unknown")
            n_samples = len(items)
            print(f"  {demo_key} ({demo_type}, {n_samples} samples)")
        print("=" * 60 + "\n")
        return None

    # If demo_id is specified, filter to that demo
    if demo_id is not None:
        # Support partial matching
        matching_demos = [k for k in demos_by_id.keys() if demo_id in k]
        if len(matching_demos) == 0:
            print(f"\nNo demos found matching '{demo_id}'")
            print("Available demos:")
            for demo_key in sorted(demos_by_id.keys()):
                print(f"  {demo_key}")
            # if len(demos_by_id) > 20:
            #     print(f"  ... and {len(demos_by_id) - 20} more")
            raise ValueError(f"Demo '{demo_id}' not found")

        if len(matching_demos) > 1:
            print(f"\nMultiple demos match '{demo_id}':")
            for k in matching_demos:
                print(f"  {k}")
            print(f"Using first match: {matching_demos[0]}")

        selected_demo = matching_demos[0]
        samples = demos_by_id[selected_demo]
        # Sort by frame index for sequential viewing
        samples = sorted(samples, key=lambda x: x["frame_idx_1"])
        print(f"\nSelected demo: {selected_demo} ({len(samples)} samples)")
        return samples

    # Sample diverse examples
    if len(combined_data) > num_samples:
        # Try to get mix of success/failure
        success_samples = [x for x in combined_data if x["demo_success"] == "success"]
        failure_samples = [x for x in combined_data if x["demo_success"] == "failure"]

        n_success = min(len(success_samples), num_samples // 2)
        n_failure = min(len(failure_samples), num_samples - n_success)

        samples = random.sample(success_samples, n_success) + random.sample(failure_samples, n_failure)
        random.shuffle(samples)
    else:
        samples = combined_data

    return samples


def visualize_results(results, output_dir, demo_id=None):
    """Create visualization of test results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename suffix based on demo_id
    suffix = f"_{demo_id}" if demo_id else ""

    # Compute metrics
    correct = 0
    sign_correct = 0
    total_with_pred = 0
    errors = []

    for r in results:
        if r["prediction"] is not None:
            total_with_pred += 1
            error = abs(r["prediction"] - r["ground_truth"])
            errors.append(error)

            # Check if sign is correct (positive vs negative)
            if (r["prediction"] > 0) == (r["ground_truth"] > 0):
                sign_correct += 1

            # Check if within threshold
            if error <= 10:
                correct += 1

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Samples with valid predictions: {total_with_pred}")
    if total_with_pred > 0:
        print(f"Sign accuracy: {100 * sign_correct / total_with_pred:.1f}%")
        print(f"Accuracy (within 10): {100 * correct / total_with_pred:.1f}%")
        print(f"Mean absolute error: {np.mean(errors):.2f}")
        print(f"Median absolute error: {np.median(errors):.2f}")
    print("=" * 60 + "\n")

    # Create visualization grid
    n = len(results)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for idx, r in enumerate(results):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        if r["overlay"] is not None:
            ax.imshow(r["overlay"])
        else:
            ax.text(0.5, 0.5, "Failed to load", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")

        gt = r["ground_truth"]
        pred = r["prediction"]
        raw = r["raw_output"]
        demo_type = r["demo_success"]

        # Color based on correctness
        if pred is not None:
            if (pred > 0) == (gt > 0):
                title_color = "#2ecc71"  # Green - correct sign
            else:
                title_color = "#e74c3c"  # Red - wrong sign
        else:
            title_color = "#95a5a6"  # Gray - failed to parse

        pred_str = f"{pred:.0f}" if pred is not None else f"'{raw}'"
        frame_info = f"Frame {r['frame_idx_1']} vs {r['frame_idx_2']}"
        title = f"{frame_info}\nGT: {gt} | Pred: {pred_str} | {demo_type}"
        ax.set_title(title, fontsize=9, color=title_color, fontweight="bold")

    # Build suptitle
    metrics_str = f"Sign Acc: {100*sign_correct/max(1,total_with_pred):.1f}%, MAE: {np.mean(errors) if errors else 0:.1f}"
    if demo_id:
        suptitle = f"Demo: {demo_id} | {metrics_str}"
    else:
        suptitle = f"VLM Test Results | {metrics_str}"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / f"test_results{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {output_path}")

    # Save detailed results JSON
    results_json = []
    for r in results:
        results_json.append({
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "raw_output": r["raw_output"],
            "demo_success": r["demo_success"],
            "demo_id_exact": r["demo_id_exact"],
            "task_token": r["task_token"],
            "frame_idx_1": r["frame_idx_1"],
            "frame_idx_2": r["frame_idx_2"],
        })

    json_path = output_dir / f"test_results{suffix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "metrics": {
                "total_samples": len(results),
                "valid_predictions": total_with_pred,
                "sign_accuracy": sign_correct / max(1, total_with_pred),
                "accuracy_within_10": correct / max(1, total_with_pred),
                "mean_absolute_error": float(np.mean(errors)) if errors else None,
                "median_absolute_error": float(np.median(errors)) if errors else None,
            },
            "results": results_json,
        }, f, indent=2)
    print(f"Saved detailed results to {json_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Test trained VLM on overlay images")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--base_dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--demo_id", type=str, default=None, help="Test specific demo (supports partial match)")
    parser.add_argument("--list_demos", action="store_true", help="List available demos and exit")
    args = parser.parse_args()

    # Build test samples (or list demos)
    samples = build_test_samples(
        args.base_dataset_path,
        args.num_samples,
        args.split,
        demo_id=args.demo_id,
        list_demos=args.list_demos,
    )

    # If just listing demos, exit
    if samples is None:
        return

    # Model is required for actual testing
    if args.model_path is None:
        parser.error("--model_path is required for testing")

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Run inference
    results = []
    print(f"\nRunning inference on {len(samples)} samples...")

    for item in tqdm(samples, desc="Testing"):
        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
        except Exception as e:
            print(f"Failed to load frames: {e}")
            overlay = None

        if overlay is not None:
            raw_output, prediction = run_inference(model, processor, overlay, item["task_token"])
        else:
            raw_output, prediction = "ERROR", None

        results.append({
            "overlay": overlay,
            "ground_truth": item["correct_answer"],
            "prediction": prediction,
            "raw_output": raw_output,
            "demo_success": item["demo_success"],
            "demo_id_exact": item.get("demo_id_exact", item.get("demo_id", "unknown")),
            "task_token": item["task_token"],
            "frame_idx_1": item["frame_idx_1"],
            "frame_idx_2": item["frame_idx_2"],
            "video_path_1": item["video_path_1"],
            "video_path_2": item["video_path_2"],
        })

    # Visualize results
    visualize_results(results, args.output_dir, demo_id=args.demo_id)


if __name__ == "__main__":
    main()
