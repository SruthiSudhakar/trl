"""
Test script for VLM Overlay Regression v2 trained models.

Loads a trained checkpoint and evaluates it on a new dataset using
on-the-fly video frame decoding (matching the v2 training pipeline).

Usage:

CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_v2.py \
    --model_name_or_path outputs/PnPStoveToCounter_expert_data_20251224_032904/checkpoint-10000 \
    --base_dataset_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove \
    --split train \
    --compare_interval 4,8,12,16 \
    --batch_size 300 \
    --num_samples 299 \
    --train_val_split_index 5 \
    --visualize

"""

import argparse
import ast
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from video_frame_utils import create_side_by_side, extract_frame, get_frame_source, find_job_dirs

# ============================================================================
# Constants (must match training script)
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
# Data loading (reuses logic from training script)
# ============================================================================

def load_trajectories(job_dirs):
    success_data = []
    failure_data = []

    for job_dir in job_dirs:
        metadata_path = Path(job_dir) / "eval_log.json"
        if not metadata_path.exists():
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
                failure_data.append({
                    "video_path": video_path,
                    "trajectory_index": int(len(trajectory) / 2),
                    "sf": "fail",
                    "demo_id": demo_id,
                })

    return success_data, failure_data


def build_eval_pairs(success_data, failure_data, compare_intervals, job_name, sample_interval=5):
    """Build frame pair metadata for evaluation (no images, just paths + indices)."""
    pairs = []

    # Find task token
    task_token = None
    for task_key, token in TASK_TOKENS.items():
        if task_key in job_name:
            task_token = token
            break
    if task_token is None:
        raise ValueError(f"No task token found for job name: {job_name}")

    user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)

    # Match failures to successes by demo_id
    success_by_demo = defaultdict(list)
    for sd in success_data:
        success_by_demo[sd["demo_id"]].append(sd)

    # Success pairs: compare frames at different intervals within same trajectory
    for demo in success_data:
        source = get_frame_source(demo["video_path"])
        for interval in compare_intervals:
            max_idx1 = demo["trajectory_index"] - interval - 1
            for idx1 in range(0, max_idx1 + 1, sample_interval):
                idx2 = idx1 + interval
                correct_answer = 32

                # 50% swap
                v1, f1, v2, f2 = source, idx1, source, idx2
                if random.random() < 0.5:
                    v1, f1, v2, f2 = v2, f2, v1, f1
                    correct_answer = -correct_answer

                pairs.append({
                    "video_path_1": v1, "frame_idx_1": f1,
                    "video_path_2": v2, "frame_idx_2": f2,
                    "correct_answer": correct_answer,
                    "user_prompt": user_prompt,
                    "task_token": task_token,
                    "demo_type": "success",
                    "demo_id": demo["demo_id"],
                })

    # Failure vs success pairs
    for demo in failure_data:
        demo_id = demo["demo_id"]
        if demo_id not in success_by_demo:
            continue
        success_demo = success_by_demo[demo_id][0]
        source_fail = get_frame_source(demo["video_path"])
        source_succ = get_frame_source(success_demo["video_path"])
        max_idx = demo["trajectory_index"] - 1

        for idx1 in range(0, max_idx + 1, sample_interval):
            correct_answer = 32
            v1, f1, v2, f2 = source_fail, idx1, source_succ, idx1
            if random.random() < 0.5:
                v1, f1, v2, f2 = v2, f2, v1, f1
                correct_answer = -correct_answer

            pairs.append({
                "video_path_1": v1, "frame_idx_1": f1,
                "video_path_2": v2, "frame_idx_2": f2,
                "correct_answer": correct_answer,
                "user_prompt": user_prompt,
                "task_token": task_token,
                "demo_type": "failure_vs_success",
                "demo_id": demo_id,
            })

    return pairs


# ============================================================================
# Inference
# ============================================================================

def extract_number(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if m is None:
        return None
    return float(m.group(0))


def run_inference(model, processor, pairs, batch_size, device, max_new_tokens=64):
    """Run batched inference and return results."""
    results = []
    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for b in tqdm(range(num_batches), desc="Evaluating"):
        batch = pairs[b * batch_size : (b + 1) * batch_size]
        texts = []
        images_list = []

        for item in batch:
            try:
                frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                overlay = create_side_by_side(frame1, frame2)
            except Exception as e:
                print(f"Warning: failed to load frames: {e}")
                overlay = Image.new("RGB", (256, 128), (128, 128, 128))

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": overlay},
                        {"type": "text", "text": item["user_prompt"]},
                    ],
                },
            ]

            try:
                import qwen_vl_utils
                image_input, _ = qwen_vl_utils.process_vision_info(conversation)
            except (ImportError, Exception):
                image_input = [overlay]

            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            images_list.append(image_input)

        inputs = processor(
            text=texts,
            images=images_list if any(img is not None for img in images_list) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.95, do_sample=True
            )

        for i, item in enumerate(batch):
            gen_tokens = outputs[i, inputs["input_ids"].shape[1] :]
            completion = processor.decode(gen_tokens, skip_special_tokens=True).strip()
            pred = extract_number(completion)
            gt = float(item["correct_answer"])

            if pred is not None:
                pred = max(-100.0, min(100.0, pred))
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
                "raw_completion": completion,
                "error": error,
                "abs_error": abs_error,
                "sign_correct": bool(sign_correct) if sign_correct is not None else None,
                "demo_type": item["demo_type"],
                "demo_id": item["demo_id"],
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

    # --- Plot 2: Sample overlays with predictions ---
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
            ax.set_title(
                f"GT: {r['ground_truth']:.0f} | Pred: {r['prediction']:.1f} | Err: {r['abs_error']:.1f}\n"
                f"{r['demo_type']} | Raw: {r['raw_completion'][:30]}",
                fontsize=9, color=color, fontweight="bold",
            )

        # Hide empty subplots
        for idx in range(len(samples), rows * cols):
            axes_grid[idx // cols, idx % cols].axis("off")

        plt.suptitle(f"{label.capitalize()} Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"samples_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Test a trained VLM overlay regression v2 model on a new dataset")

    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Path to trained checkpoint (full model or PEFT adapter)")
    p.add_argument("--base_model_name_or_path", type=str,
                   default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Base model path (used when loading PEFT adapters)")
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Path to dataset directory containing job subdirs with eval_log.json + MP4s")
    p.add_argument("--split", type=str, default="val",
                   help="train | val | integer (number of job dirs to use)")
    p.add_argument("--train_val_split_index", type=int, default=5,
                   help="Last N job dirs used for val split")
    p.add_argument("--compare_interval", type=str, default="4,8,12,16",
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
                   help="Number of examples to show in best/worst visualizations")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (defaults to <model_path>/eval_<timestamp>)")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output directory")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_intervals = [int(x.strip()) for x in args.compare_interval.split(",")]

    output_dir = args.output_dir or os.path.join(
        args.model_name_or_path, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.prefix}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load dataset ----
    print("Loading dataset...")
    import glob as globmod
    all_dirs = sorted(globmod.glob(f"{args.base_dataset_path}/*"))
    job_dirs = [d for d in all_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "eval_log.json"))]

    if args.split == "val":
        job_dirs = job_dirs[-args.train_val_split_index :]
    elif args.split == "train":
        job_dirs = job_dirs[: -args.train_val_split_index]
    else:
        job_dirs = job_dirs[: int(args.split)]

    print(f"Using {len(job_dirs)} job directories for '{args.split}' split")

    success_data, failure_data = load_trajectories(job_dirs)
    print(f"Loaded {len(success_data)} success + {len(failure_data)} failure trajectories")

    job_name = Path(job_dirs[0]).name if job_dirs else ""
    pairs = build_eval_pairs(success_data, failure_data, compare_intervals, job_name, args.sample_interval)
    print(f"Built {len(pairs)} evaluation pairs")

    if args.num_samples and args.num_samples < len(pairs):
        pairs = random.sample(pairs, args.num_samples)
        print(f"Subsampled to {len(pairs)} pairs")

    # ---- Load model ----
    print("Loading model...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    dtype = dtype_map.get(args.dtype, "auto")

    # Check if checkpoint is a PEFT adapter
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
        print(f"Loading full model from {args.model_name_or_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype if dtype != "auto" else None,
            device_map="auto",
            trust_remote_code=True,
        )
        processor_path = args.model_name_or_path

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model.eval()
    print("Model loaded")

    # ---- Run inference ----
    results = run_inference(model, processor, pairs, args.batch_size, device)

    # ---- Compute metrics ----
    metrics = compute_metrics(results, tolerance=args.tolerance)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:            {args.model_name_or_path}")
    print(f"Dataset:          {args.base_dataset_path}")
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
        if key.endswith("_mae") and key != "mae":
            dtype_name = key.replace("_mae", "")
            count = metrics.get(f"{dtype_name}_count", 0)
            mae = metrics[key]
            sign_acc = metrics.get(f"{dtype_name}_sign_accuracy", float("nan"))
            print(f"  {dtype_name}: n={count}, MAE={mae:.3f}, sign_acc={sign_acc:.3f}")

    print("=" * 60)

    # ---- Save results ----
    summary = {
        "args": vars(args),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
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
        visualize_results(results, output_dir, num_examples=args.num_visualize)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()