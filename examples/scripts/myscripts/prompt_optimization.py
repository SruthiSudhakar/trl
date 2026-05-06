"""
Test script for VLM Overlay Regression v2 trained models.

Loads a trained checkpoint and evaluates it on one or more datasets using
on-the-fly video frame decoding (matching the v2 training pipeline).

Usage:
export GOOGLE_API_KEY="AIzaSyDz5juA63feTZpUReaD7KEIzNiNQVWekL0"
# Single dataset
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_v2.py \
    --base_dataset_path /workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove \
    --split val \
    --compare_interval 4,8,12,16 \
    --batch_size 300 \
    --num_samples 299 \
    --train_val_split_index 5 \
    --visualize

# Multiple datasets (CSV)
CUDA_VISIBLE_DEVICES=7 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_v2.py \
    --user_prompt_path \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove"
    --split train \
    --compare_interval 4,8,12,16 \
    --batch_size 300 \
    --num_samples 299 \
    --train_val_split_index 5 \
    --visualize

    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPMicrowaveToCounter","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToSink","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPSinkToCounter","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCoffeeServeMug","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter","/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToCab" \

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

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from video_frame_utils import create_side_by_side, extract_frame, find_job_dirs

# Import shared code from training script
from sft_vlm_overlay_regression_v2 import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
)
USER_PROMPT_TEMPLATE = """
You are an expert roboticist tasked to predict task completion
percentages for frames of a robot for the task of {task_description}.
The task completion percentages are between 0 and 100, where 100
corresponds to full task completion. We provide several examples of
the robot performing the task at various stages and their
corresponding task completion percentages. Note that these frames are
in random order, so please pay attention to the individual frames
when reasoning about task completion percentage.
Initial robot scene: [IMG]
In the initial robot scene, the task completion percentage is 0.
Now, for the task of {task_description}, output the task completion
percentage for the following frames that are presented in random
order. For each frame, format your response as follow: Frame {i}:
Frame Description: {}, Task Completion Percentages:{}%
Frame 1: [IMG]
...
Frame n: [IMG]
"""

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

            # Build user prompt from task_token
            user_prompt = USER_PROMPT_TEMPLATE.format(task_token=item["task_token"])

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": overlay},
                        {"type": "text", "text": user_prompt},
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
    p = argparse.ArgumentParser(description="Test a trained VLM overlay regression v2 model on new dataset(s)")

    p.add_argument("--model_name_or_path", type=str, required=True, default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Path to trained checkpoint (full model or PEFT adapter)")
    p.add_argument("--base_model_name_or_path", type=str,
                   default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Base model path (used when loading PEFT adapters)")
    # Option B: CSV list
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Dataset path OR comma-separated list of dataset paths")
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
                   help="Output directory (defaults to <model_path>/<prefix>eval_<timestamp>_<intervals>)")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output directory")

    return p.parse_args()


def _split_dataset_paths(base_dataset_path_csv: str):
    # Allow escaping commas via '\,' if you ever need it.
    # We'll do a tiny parser to split only on unescaped commas.
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
    """
    Create a stable, filesystem-friendly tag for a dataset.
    We keep it simple: use last path component; if empty, fallback to 'dataset'.
    """
    name = Path(dataset_path).name.strip()
    if not name:
        name = "dataset"
    # Replace spaces just in case
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

    # Root output dir (one folder for the whole run)
    root_output_dir = args.output_dir or os.path.join(
        args.model_name_or_path,
        f"{args.prefix}eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(map(str, compare_intervals))}"
    )
    os.makedirs(root_output_dir, exist_ok=True)

    # ---- Load model ONCE ----
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

    all_metrics = {}
    all_summaries = []

    # ---- Evaluate each dataset (model stays in memory) ----
    for base_dataset_path in dataset_paths:
        print("\n" + "#" * 80)
        print(f"Evaluating dataset: {base_dataset_path}")
        print("#" * 80)

        dataset_tag = _dataset_output_tag(base_dataset_path)
        output_dir = os.path.join(root_output_dir, dataset_tag)
        os.makedirs(output_dir, exist_ok=True)

        # ---- Load dataset using shared functions from training script ----
        print("Loading dataset...")
        job_dirs = find_job_dirs(base_dataset_path)

        if 'robocasa/datasets' in base_dataset_path:
            job_dirs = job_dirs[:]
        elif args.split == "val":
            if args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[-args.train_val_split_index :]
        elif args.split == "train":
            if args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[: -args.train_val_split_index]
        else:
            job_dirs = job_dirs[: int(args.split)]

        print(f"Using {len(job_dirs)} job directories for '{args.split}' split")

        success_data, unfiltered_failure_data = load_trajectories(job_dirs)
        print(f"Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

        failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
        print(f"Matched {len(failure_data)} failures to success trajectories")

        # ============================
        # Balanced sampling for this task
        # ============================
        if success_data and failure_data:
            success_data, failure_data, success_by_demo, failure_by_demo = balance_by_demo_id(
                success_data, failure_data, 50
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

            print(f"After per-demo cap ({50}): "
                        f"{len(success_data)} -> {len(capped_success)} success, "
                        f"{len(failure_data)} -> {len(capped_failure)} failure")
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
            pairs = random.sample(pairs, args.num_samples)
            print(f"Subsampled to {len(pairs)} pairs")

        # ---- Run inference ----
        results = run_inference(model, processor, pairs, args.batch_size, device)

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
            if key.endswith("_mae") and key != "mae":
                dtype_name = key.replace("_mae", "")
                count = metrics.get(f"{dtype_name}_count", 0)
                mae = metrics[key]
                sign_acc = metrics.get(f"{dtype_name}_sign_accuracy", float("nan"))
                print(f"  {dtype_name}: n={count}, MAE={mae:.3f}, sign_acc={sign_acc:.3f}")

        print("=" * 60)

        # ---- Save results ----
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

        # ---- Visualize ----
        if args.visualize:
            visualize_results(results, output_dir, num_examples=args.num_visualize)

        print(f"\nDataset outputs saved to {output_dir}")

    # ---- Save combined metrics/summaries at root ----
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
