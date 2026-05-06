"""
Evaluation script for MLP Baseline model.

Uses the same data pipeline and episode-level split as training,
so val episodes are guaranteed to be held-out data.

Two approaches supported:
  1. If cached features exist in model dir, use them (fast)
  2. Otherwise, extract CLIP features on-the-fly (slower, but works standalone)

Usage:
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/evaluate_mlp_baseline.py \
    --model_path outputs/mlp_baseline_test/best_mlp_head.pt \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPStoveToCounter_mg_fixed_224" \
    --compare_interval 4,8,12,16 \
    --batch_size 64 \
    --visualize
"""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from video_frame_utils import create_side_by_side, extract_frame

from mlp_baseline_overlay import (
    MLPHead,
    load_all_pairs,
    episode_level_split,
    extract_and_cache_features,
    get_cache_path,
)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def visualize_results(eval_data, all_preds, all_labels, output_dir, num_examples=8):
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Map predictions back to +32/-32
    pred_values = [32 if p == 1 else -32 for p in all_preds]
    gt_values = [32 if l == 1 else -32 for l in all_labels]
    correct_mask = [p == l for p, l in zip(all_preds, all_labels)]

    # --- Plot 1: Confusion-style summary ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MLP Baseline Results (n={len(all_preds)})", fontsize=14, fontweight="bold")

    # Accuracy bar
    acc = sum(correct_mask) / len(correct_mask)
    axes[0].bar(["Correct", "Wrong"], [sum(correct_mask), len(correct_mask) - sum(correct_mask)],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_title(f"Overall Accuracy: {acc:.3f}")
    axes[0].set_ylabel("Count")

    # Per-class accuracy
    for cls, cls_name in [(0, "-32"), (1, "+32")]:
        cls_idx = [i for i, l in enumerate(all_labels) if l == cls]
        if cls_idx:
            cls_acc = sum(1 for i in cls_idx if all_preds[i] == cls) / len(cls_idx)
            axes[1].bar(cls_name, cls_acc, color="#3498db")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Per-Class Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="chance")
    axes[1].legend()

    # Prediction distribution
    axes[2].hist(pred_values, bins=[-48, -16, 16, 48], color="#9b59b6", edgecolor="black", alpha=0.7)
    axes[2].set_title("Prediction Distribution")
    axes[2].set_xlabel("Predicted Value")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(viz_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Sample correct and wrong predictions ---
    wrong_idx = [i for i, c in enumerate(correct_mask) if not c]
    right_idx = [i for i, c in enumerate(correct_mask) if c]

    rng = random.Random(42)
    for label, indices in [("correct", right_idx), ("wrong", wrong_idx)]:
        samples = rng.sample(indices, min(num_examples, len(indices))) if indices else []
        if not samples:
            continue

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

        for plot_idx, data_idx in enumerate(samples):
            row, col = plot_idx // cols, plot_idx % cols
            ax = axes_grid[row, col]
            item = eval_data[data_idx]

            try:
                f1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                f2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                overlay = create_side_by_side(f1, f2)
                ax.imshow(overlay)
            except Exception:
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")
            gt = gt_values[data_idx]
            pred = pred_values[data_idx]
            color = "green" if correct_mask[data_idx] else "red"
            demo_type = item.get("demo_success", "?")
            ax.set_title(
                f"GT: {gt} | Pred: {pred} | {demo_type}\n"
                f"f1={item['frame_idx_1']} f2={item['frame_idx_2']}",
                fontsize=9, color=color, fontweight="bold",
            )

        for plot_idx in range(len(samples), rows * cols):
            axes_grid[plot_idx // cols, plot_idx % cols].axis("off")

        plt.suptitle(f"{label.capitalize()} Predictions (MLP Baseline)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"samples_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MLP baseline on held-out val data")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to best_mlp_head.pt")
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Dataset path or comma-separated list (same as training)")
    p.add_argument("--compare_interval", type=str, default="4,8,12,16")
    p.add_argument("--split", type=str, default="train",
                   help="Must match the split used during training")
    p.add_argument("--train_val_split_index", type=int, default=5)
    p.add_argument("--sample_interval", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--extract_batch_size", type=int, default=64,
                   help="Batch size for CLIP feature extraction (if no cache)")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--num_visualize", type=int, default=8)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (defaults to same dir as model_path)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = str(Path(args.model_path).parent)
    output_dir = args.output_dir or str(Path(model_dir) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data with same pipeline as training ----
    print("=" * 60)
    print("Loading data (same pipeline as training)...")
    print("=" * 60)
    combined_data = load_all_pairs(args)

    if not combined_data:
        print("No data loaded. Exiting.")
        return

    # Same episode-level split as training — eval_data is the held-out set
    _, eval_data = episode_level_split(combined_data, seed=args.seed)
    del combined_data

    if not eval_data:
        print("No eval data after split. Exiting.")
        return

    print(f"\nEvaluating on {len(eval_data)} held-out samples")

    # ---- Get eval features (use training cache if available, otherwise extract) ----
    eval_cache = get_cache_path(model_dir, "eval")

    if os.path.exists(eval_cache):
        print(f"Loading cached eval features from {eval_cache}")
        cached = torch.load(eval_cache, map_location="cpu")
        eval_features, eval_labels = cached["features"], cached["labels"]
    else:
        print("No cached features found. Extracting CLIP features...")
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

        eval_features, eval_labels = extract_and_cache_features(
            eval_data, clip_model, clip_processor, device,
            get_cache_path(output_dir, "eval"),
            batch_size=args.extract_batch_size,
        )
        del clip_model, clip_processor
        torch.cuda.empty_cache()

    eval_loader = DataLoader(
        TensorDataset(eval_features, eval_labels),
        batch_size=args.batch_size, shuffle=False,
    )

    # ---- Load model ----
    print(f"Loading model from {args.model_path}")
    clip_dim = eval_features.shape[1]
    model = MLPHead(
        input_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    # ---- Evaluate ----
    val_loss, val_acc, all_preds, all_labels = evaluate(model, eval_loader, criterion, device)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (MLP Baseline)")
    print("=" * 60)
    print(f"Model:          {args.model_path}")
    print(f"Dataset:        {args.base_dataset_path}")
    print(f"Val samples:    {len(eval_data)}")
    print(f"Val Loss:       {val_loss:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")

    # Per-class
    for cls, cls_name in [(0, "-32 (left more progress)"), (1, "+32 (right more progress)")]:
        cls_idx = [i for i, l in enumerate(all_labels) if l == cls]
        if cls_idx:
            cls_correct = sum(1 for i in cls_idx if all_preds[i] == cls)
            print(f"  {cls_name}: {cls_correct}/{len(cls_idx)} = {cls_correct / len(cls_idx):.4f}")

    # Per demo_type
    demo_types = set(item.get("demo_success", "unknown") for item in eval_data)
    for dtype in sorted(demo_types):
        dtype_idx = [i for i, item in enumerate(eval_data) if item.get("demo_success", "unknown") == dtype]
        if dtype_idx:
            dtype_correct = sum(1 for i in dtype_idx if all_preds[i] == all_labels[i])
            print(f"  {dtype}: {dtype_correct}/{len(dtype_idx)} = {dtype_correct / len(dtype_idx):.4f}")

    print("=" * 60)

    # ---- Save results ----
    results = {
        "args": vars(args),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_samples": len(eval_data),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ---- Visualize ----
    if args.visualize:
        visualize_results(eval_data, all_preds, all_labels, output_dir, num_examples=args.num_visualize)

    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
