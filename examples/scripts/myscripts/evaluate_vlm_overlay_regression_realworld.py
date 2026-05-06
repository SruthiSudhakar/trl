"""
Evaluation script for VLM Overlay Regression models on realworld dataset format.

Loads a trained checkpoint and evaluates it on realworld datasets that use
the episode/models/frames directory structure.

Usage:
# Single dataset
CUDA_VISIBLE_DEVICES=7 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_realworld.py \
    --model_name_or_path outputs/PutKiwiInCenterOfTable_ObjectCentricDistributionShift_20260222_165812/checkpoint-1000 \
    --base_dataset_path "realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real" \
    --batch_size 50 \
    --num_samples 50 \
    --visualize
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_realworld.py \
    --model_name_or_path outputs/realworld_all_20260223_162917/checkpoint-3800 \
    --base_dataset_path "realworld_dataset/BimanualBikeRotorInstall-Nominal-real" \
    --split val \
    --num_samples 300 \
    --compare_interval 32,36 \
    --batch_size 100 \
    --visualize

# Multiple datasets
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_vlm_overlay_regression_realworld.py \
    --model_name_or_path outputs/realworld_all_20260223_153806/checkpoint-3 \
    --base_dataset_path "realworld_dataset/BimanualBikeRotorInstall-Nominal-real,realworld_dataset/BimanualClearKitchenCounter-Nominal-real,realworld_dataset/BimanualSetUpBreakfastTable-Nominal-real,realworld_dataset/CleanLitterBox-Nominal-real,realworld_dataset/CutAppleIntoSlices-Nominal-real,realworld_dataset/PushCoasterToMug-Nominal-real,realworld_dataset/PushCoasterToMug-ObjectCentricDistributionShift-real,realworld_dataset/PutKiwiInCenterOfTable-ObjectCentricDistributionShift-real,realworld_dataset/PutKiwiInCenterOfTable-StationDistributionShift-real,realworld_dataset/PutKiwiInCenterOfTableSeenTasks_backfill-salem--video,realworld_dataset/TurnMugRightsideUp-Nominal-real,realworld_dataset/TurnMugRightsideUp-ObjectCentricDistributionShift-real,realworld_dataset/TurnMugRightsideUp-StationDistributionShift-real" \
    --split val \
    --compare_interval 32 \
    --batch_size 8 \
    --visualize

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

from video_frame_utils import create_side_by_side, extract_frame

# Import shared code from realworld training script
from sft_vlm_overlay_regression_realworld_data import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    TASK_TOKENS,
    is_realworld_format,
    load_realworld_trajectories,
    match_realworld_failures_to_successes,
    compute_realworld_failure_filter_stats,
    build_realworld_frame_pairs,
)


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
                "compare_interval": item.get("compare_interval", "?"),
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

    # Per compare_interval breakdown
    interval_groups = defaultdict(list)
    for r in valid:
        interval = r.get("compare_interval", "?")
        if interval and interval != "?":
            label = f"intra-{interval}"
        else:
            # Fallback: check if same video
            v1 = r.get("video_path_1")
            v2 = r.get("video_path_2")
            f1 = r.get("frame_idx_1")
            f2 = r.get("frame_idx_2")
            if v1 == v2 and isinstance(f1, int) and isinstance(f2, int):
                label = f"intra-{abs(f2 - f1)}"
            else:
                label = "sf"
        interval_groups[label].append(r)

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
            interval = r.get("compare_interval", "?")
            interval_str = f"int={interval}" if interval else "fail-pair"
            color = "green" if r["abs_error"] <= 3 else "red"
            ax.set_title(
                f"GT: {r['ground_truth']:.0f} | Pred: {r['prediction']:.1f} | Err: {r['abs_error']:.1f}\n"
                f"{r['demo_type']} | {interval_str} | f1={r['frame_idx_1']} f2={r['frame_idx_2']}\n"
                f"Raw: {r['raw_completion'][:30]}",
                fontsize=8, color=color, fontweight="bold",
            )

        for idx in range(len(samples), rows * cols):
            axes_grid[idx // cols, idx % cols].axis("off")

        plt.suptitle(f"{label.capitalize()} Predictions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"samples_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Plot 3: Sign accuracy by interval ---
    interval_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in valid:
        interval = r.get("compare_interval", "?")
        label = f"int={interval}" if interval else "fail-pair"
        interval_stats[label]["total"] += 1
        if r["sign_correct"]:
            interval_stats[label]["correct"] += 1

    if interval_stats:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = sorted(interval_stats.keys())
        accs = [interval_stats[l]["correct"] / interval_stats[l]["total"] for l in labels]
        counts = [interval_stats[l]["total"] for l in labels]
        bars = ax.bar(labels, accs, color="#3498db")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={count}", ha="center", fontsize=9)
        ax.set_ylabel("Sign Accuracy")
        ax.set_title("Sign Accuracy by Compare Interval")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="chance")
        ax.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "accuracy_by_interval.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLM overlay regression on realworld datasets")

    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Path to trained checkpoint (full model or PEFT adapter)")
    p.add_argument("--base_model_name_or_path", type=str,
                   default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                   help="Base model path (used when loading PEFT adapters)")
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
                   help="Number of examples to show in best/worst visualizations")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (defaults to <model_path>/eval_realworld_<timestamp>)")
    p.add_argument("--prefix", type=str, default="",
                   help="Prefix for output directory")

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
        f"{args.prefix}eval_realworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'_'.join(map(str, compare_intervals))}"
    )
    os.makedirs(root_output_dir, exist_ok=True)

    # ============================
    # Build ALL pairs across all tasks first (matching training script)
    # ============================
    all_pairs = []

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

    print(f"\nTotal pairs across all tasks: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs built. Exiting.")
        return

    # ============================
    # Episode-level split matching training script EXACTLY
    # ============================
    # Training script does:
    #   all_demo_ids = sorted(set(item["demo_id"] for item in combined_data))
    #   rng = random.Random(42)
    #   rng.shuffle(all_demo_ids)
    #   n_eval_demos = max(1, int(len(all_demo_ids) * 0.1))
    #   eval_demo_ids = set(all_demo_ids[:n_eval_demos])
    #   train_demo_ids = set(all_demo_ids[n_eval_demos:])
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

    # ---- Load model ----
    print("\nLoading model...")
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

    # ---- Run inference on all pairs ----
    results = run_inference(model, processor, pairs, args.batch_size, device)

    # ---- Compute overall metrics ----
    all_metrics = {}
    all_summaries = []

    metrics = compute_metrics(results, tolerance=args.tolerance)
    all_metrics["overall"] = metrics

    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:            {args.model_name_or_path}")
    print(f"Datasets:         {', '.join(dataset_paths)}")
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
        for label, stats in sorted(interval_breakdown.items(),
                                    key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1]) if "-" in x[0] else 0)):
            print(f"  {label:>10s}: n={stats['count']:>4d}, sign_acc={stats['sign_accuracy']:.3f}, MAE={stats['mae']:.3f}")

    print("=" * 60)

    # ---- Per-task metrics ----
    task_tokens_in_results = set(r["task_token"] for r in results)
    for task_token in sorted(task_tokens_in_results):
        task_results = [r for r in results if r["task_token"] == task_token]
        task_metrics = compute_metrics(task_results, tolerance=args.tolerance)
        all_metrics[task_token] = task_metrics

        print(f"\n  Task: {task_token} (n={len(task_results)})")
        print(f"    MAE: {task_metrics.get('mae', float('nan')):.3f}, "
              f"Sign acc: {task_metrics.get('sign_accuracy', float('nan')):.3f}")

        # Per-interval for this task
        task_interval = task_metrics.get("interval_breakdown", {})
        for lbl, st in sorted(task_interval.items(),
                               key=lambda x: (0, 0) if x[0] == "sf" else (1, int(x[0].split("-")[1]) if "-" in x[0] else 0)):
            print(f"      {lbl:>10s}: n={st['count']:>4d}, sign_acc={st['sign_accuracy']:.3f}")

    # ---- Save results ----
    summary = {
        "args": vars(args),
        "datasets": dataset_paths,
        "metrics": metrics,
        "per_task_metrics": {k: v for k, v in all_metrics.items() if k != "overall"},
        "timestamp": datetime.now().isoformat(),
    }
    all_summaries.append(summary)

    summary_path = os.path.join(root_output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    completions_path = os.path.join(root_output_dir, "completions.json")
    with open(completions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Completions saved to {completions_path}")

    # ---- Visualize ----
    if args.visualize:
        visualize_results(results, root_output_dir, num_examples=args.num_visualize)

    print(f"\nAll outputs saved to {root_output_dir}")


if __name__ == "__main__":
    main()
