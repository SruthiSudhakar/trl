"""
MLP Classifier Baseline for Frame Pair Progress Detection.

Uses a frozen CLIP ViT-L/14 encoder + small MLP head to classify
which frame in a side-by-side overlay shows more task progress.
Same data pipeline as sft_vlm_overlay_regression_v2.py for fair comparison.

Two-phase approach:
  1. Extract CLIP features for all pairs and cache to disk (slow, one-time)
  2. Train MLP on cached features (fast, each epoch is seconds)

Usage:
CUDA_VISIBLE_DEVICES=0 python3 examples/scripts/myscripts/mlp_baseline_overlay.py \
    --base_dataset_path "/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_na_na_16_mg_place_PnPStoveToCounter_mg_fixed_224" \
    --compare_interval 4,8,12,16 \
    --split train \
    --epochs 50 \
    --batch_size 64 \
    --output_dir outputs/mlp_baseline_test

CUDA_VISIBLE_DEVICES=2 python3 examples/scripts/myscripts/mlp_baseline_overlay.py \
    --base_dataset_path "/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27,/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToStove,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPStoveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToMicrowave,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPMicrowaveToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToSink,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPSinkToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCoffeeServeMug,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCabToCounter,/workspace/guided_diffusion_policy/data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_PnPCounterToCab" \
    --compare_interval 4,8,12,16 \
    --epochs 10000 \
    --batch_size 512 \
    --output_dir outputs/mlp_baseline_all10 \
    --lr 1e-2

"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from video_frame_utils import create_side_by_side, extract_frame, find_job_dirs

from sft_vlm_overlay_regression_v2 import (
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
    compute_failure_filter_stats,
)


# ============================================================================
# Dataset for feature extraction phase
# ============================================================================

class OverlayDataset(Dataset):
    """Dataset that creates side-by-side overlays and returns CLIP-preprocessed tensors."""

    def __init__(self, pairs, clip_processor):
        self.pairs = pairs
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        try:
            frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
            frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
            overlay = create_side_by_side(frame1, frame2)
        except Exception:
            overlay = Image.new("RGB", (256, 128), (128, 128, 128))

        inputs = self.clip_processor(images=overlay, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label = 1 if item["correct_answer"] > 0 else 0

        return pixel_values, label


# ============================================================================
# Model (MLP head only — used after feature extraction)
# ============================================================================

class MLPHead(nn.Module):
    """Simple MLP classifier on pre-extracted features."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Data Loading (reuses VLM script pipeline)
# ============================================================================

def load_all_pairs(args):
    """Load and build frame pairs using the same pipeline as the VLM training script."""
    task_paths = [p.strip() for p in args.base_dataset_path.split(",") if p.strip()]
    compare_intervals = [int(x.strip()) for x in args.compare_interval.split(",")]
    combined_data = []

    for task_path in task_paths:
        print(f"Loading task: {task_path}")
        job_dirs = find_job_dirs(task_path)

        split = args.split
        if split == "val":
            job_dirs = job_dirs[-args.train_val_split_index:]
        elif split == "train":
            if 'robocasa/datasets' in task_path or args.train_val_split_index == 0:
                job_dirs = job_dirs[:]
            else:
                job_dirs = job_dirs[:-args.train_val_split_index]
        else:
            job_dirs = job_dirs[:int(split)]

        print(f"  Using {len(job_dirs)} job dirs for '{split}' split")

        if not job_dirs:
            continue

        success_data, unfiltered_failure_data = load_trajectories(job_dirs)
        print(f"  Loaded {len(success_data)} success + {len(unfiltered_failure_data)} failure trajectories")

        failure_data = match_failures_to_successes(success_data, unfiltered_failure_data)
        print(f"  Matched {len(failure_data)} failures to successes")

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

        if failure_data:
            stats_cache = Path(task_path) / "failure_filter_stats.json"
            if stats_cache.exists():
                with open(stats_cache) as f:
                    cached = json.load(f)
                    success_mean_diffs_at_idx = cached["success_mean_diffs_at_idx"]
            else:
                success_mean_diffs_at_idx = compute_failure_filter_stats(
                    success_by_demo, failure_by_demo, 0, 1
                )
                with open(stats_cache, "w") as f:
                    json.dump({"success_mean_diffs_at_idx": success_mean_diffs_at_idx}, f)
        else:
            success_mean_diffs_at_idx = {}

        job_name = Path(job_dirs[0]).name if job_dirs else ""
        task_pairs = build_frame_pairs(
            success_data=success_data,
            failure_data=failure_data,
            compare_intervals=compare_intervals,
            train_sample_interval=args.sample_interval,
            success_mean_diffs_at_idx=success_mean_diffs_at_idx,
            job_name=job_name,
            local_rank=0,
            world_size=1,
            task_path=task_path,
        )
        print(f"  Built {len(task_pairs)} pairs")
        combined_data.extend(task_pairs)

    print(f"\nTotal pairs: {len(combined_data)}")
    return combined_data


def episode_level_split(combined_data, seed=42):
    """Split data at episode level, matching training script logic exactly."""
    all_demo_ids = sorted(set(item["demo_id"] for item in combined_data))
    rng = random.Random(seed)
    rng.shuffle(all_demo_ids)
    n_eval = max(1, int(len(all_demo_ids) * 0.1))
    eval_ids = set(all_demo_ids[:n_eval])
    train_ids = set(all_demo_ids[n_eval:])

    train_data = [item for item in combined_data if item["demo_id"] in train_ids]
    eval_data = [item for item in combined_data if item["demo_id"] in eval_ids]

    print(f"Episode-level split: {len(train_ids)} train episodes, {len(eval_ids)} eval episodes")
    print(f"  Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")
    return train_data, eval_data


# ============================================================================
# Feature Extraction & Caching
# ============================================================================

def get_cache_path(output_dir, split_name):
    """Get path for cached features file."""
    return os.path.join(output_dir, f"cached_features_{split_name}.pt")


def extract_and_cache_features(pairs, clip_model, clip_processor, device, cache_path, batch_size=64):
    """Extract CLIP features for all pairs and save to disk.

    Returns (features_tensor, labels_tensor).
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        return cached["features"], cached["labels"]

    print(f"Extracting CLIP features for {len(pairs)} samples...")
    clip_model.eval()

    all_features = []
    all_labels = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="Extracting features"):
        batch_items = pairs[start:start + batch_size]
        images = []

        for item in batch_items:
            try:
                frame1 = extract_frame(item["video_path_1"], item["frame_idx_1"])
                frame2 = extract_frame(item["video_path_2"], item["frame_idx_2"])
                overlay = create_side_by_side(frame1, frame2)
            except Exception:
                overlay = Image.new("RGB", (256, 128), (128, 128, 128))
            images.append(overlay)

        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            features = clip_model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)

        all_features.append(features.cpu())

        labels = torch.tensor([1 if item["correct_answer"] > 0 else 0 for item in batch_items])
        all_labels.append(labels)

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    print(f"Saving cached features to {cache_path} "
          f"(features: {features_tensor.shape}, labels: {labels_tensor.shape})")
    torch.save({"features": features_tensor, "labels": labels_tensor}, cache_path)

    return features_tensor, labels_tensor


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
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


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MLP baseline for frame pair progress detection")
    p.add_argument("--base_dataset_path", type=str, required=True,
                   help="Dataset path or comma-separated list")
    p.add_argument("--compare_interval", type=str, default="4,8,12,16")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--train_val_split_index", type=int, default=5)
    p.add_argument("--sample_interval", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--extract_batch_size", type=int, default=64,
                   help="Batch size for CLIP feature extraction phase")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cache", action="store_true",
                   help="Force re-extraction even if cache exists")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or f"outputs/mlp_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data (same pipeline as VLM script) ----
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    combined_data = load_all_pairs(args)

    if not combined_data:
        print("No data loaded. Exiting.")
        return

    # Episode-level split
    train_data, eval_data = episode_level_split(combined_data, seed=args.seed)
    del combined_data

    # ---- Phase 1: Extract & cache CLIP features ----
    print("\n" + "=" * 60)
    print("Phase 1: CLIP Feature Extraction")
    print("=" * 60)

    train_cache = get_cache_path(output_dir, "train")
    eval_cache = get_cache_path(output_dir, "eval")

    if args.no_cache:
        for p in [train_cache, eval_cache]:
            if os.path.exists(p):
                os.remove(p)

    print(f"Loading CLIP model: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_dim = clip_model.config.projection_dim

    train_features, train_labels = extract_and_cache_features(
        train_data, clip_model, clip_processor, device, train_cache,
        batch_size=args.extract_batch_size,
    )
    eval_features, eval_labels = extract_and_cache_features(
        eval_data, clip_model, clip_processor, device, eval_cache,
        batch_size=args.extract_batch_size,
    )

    # Free CLIP model from GPU
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    print(f"\nTrain features: {train_features.shape}, Eval features: {eval_features.shape}")

    # ---- Phase 2: Train MLP on cached features ----
    print("\n" + "=" * 60)
    print("Phase 2: MLP Training on Cached Features")
    print("=" * 60)

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    eval_loader = DataLoader(
        TensorDataset(eval_features, eval_labels),
        batch_size=args.batch_size, shuffle=False,
    )

    model = MLPHead(
        input_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"MLP parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, eval_loader, criterion, device)

        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_mlp_head.pt"))

        # Print every epoch for first 10, then every 10, then every 100
        should_print = (epoch <= 10 or epoch % 10 == 0 or epoch % 100 == 0
                        or is_best or epoch == args.epochs)
        if should_print:
            print(f"Epoch {epoch:5d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                  f"{'*BEST*' if is_best else ''} | "
                  f"{elapsed:.2f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        })

    # ---- Final evaluation with best model ----
    print("\n" + "=" * 60)
    print("Final Evaluation (best model)")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(output_dir, "best_mlp_head.pt")))
    val_loss, val_acc, val_preds, val_labels = evaluate(model, eval_loader, criterion, device)

    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val Loss:     {val_loss:.4f}")

    for cls, cls_name in [(0, "-32 (left)"), (1, "+32 (right)")]:
        cls_indices = [i for i, l in enumerate(val_labels) if l == cls]
        if cls_indices:
            cls_correct = sum(1 for i in cls_indices if val_preds[i] == cls)
            print(f"  {cls_name}: {cls_correct}/{len(cls_indices)} = {cls_correct/len(cls_indices):.4f}")

    # ---- Save results ----
    results = {
        "args": vars(args),
        "best_val_acc": best_val_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "trainable_params": trainable_params,
        "clip_dim": clip_dim,
        "train_samples": len(train_features),
        "eval_samples": len(eval_features),
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Best model saved to {os.path.join(output_dir, 'best_mlp_head.pt')}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
