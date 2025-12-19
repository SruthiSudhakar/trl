#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation script for CLIP + MLP task completion model trained with sft_simple.py

This script evaluates a trained model that combines CLIP vision encoder with a 2-layer MLP
to predict task completion percentage (0-100) from robot observation images.

Example usage:
CUDA_VISIBLE_DEVICES=6 python3 examples/scripts/evaluate_sft_simple.py \
    --model_path outputs/clip-mlp-task-completion_20250918_021842/checkpoint-6200 \
    --batch_size 500 \
    --tolerance 15 \
    --dataset_path ID \
    --dataset_split train_demos
CUDA_VISIBLE_DEVICES=6 python3 examples/scripts/evaluate_sft_simple.py \
    --model_path outputs/clip-mlp-task-completion_20250918_021842/checkpoint-6200 \
    --batch_size 500 \
    --tolerance 15 \
    --dataset_path ID \
    --dataset_split val_demos
CUDA_VISIBLE_DEVICES=5 python3 examples/scripts/evaluate_sft_simple.py \
    --model_path outputs/clip-mlp-task-completion_20250918_021842/checkpoint-6200 \
    --batch_size 500 \
    --tolerance 15 \
    --dataset_path OOD \
    --dataset_split all_demos

"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset


@dataclass
class EvaluationArgs:
    """Arguments for the evaluation script."""
    model_path: str = field(
        default="outputs/clip-mlp-task-completion",
        metadata={"help": "Path to the trained model checkpoint"}
    )
    dataset_path: str = field(
        default="ID",
        metadata={"help": "Path to the HDF5 dataset file"}
    )
    cache_dir: str = field(
        default="/workspace/image_cache",
        metadata={"help": "Directory for cached images"}
    )
    clip_model_name: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Name of the CLIP model to use"}
    )
    hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for the MLP"}
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for the MLP"}
    )
    batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for evaluation"}
    )
    tolerance: float = field(
        default=10.0,
        metadata={"help": "Tolerance for accuracy calculation (± percentage points)"}
    )
    dataset_split: str = field(
        default="val_demos",
        metadata={"help": "Dataset split to evaluate on (train/val)"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run evaluation on"}
    )
    save_predictions: bool = field(
        default=True,
        metadata={"help": "Save individual predictions to file"}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to evaluate (None = all)"}
    )


class TaskCompletionModel(nn.Module):
    """
    A model that combines CLIP vision encoder with a 2-layer MLP
    to predict task completion percentage.
    """

    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", hidden_dim=256, dropout_rate=0.1):
        super().__init__()

        # Load CLIP model and processor
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP weights (optional - can be unfrozen for fine-tuning)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get CLIP vision output dimension
        clip_output_dim = self.clip.vision_model.config.hidden_size

        # 2-layer MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(clip_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1 (will be scaled to 0-100)
        )

        # Image normalization (CLIP standard)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    def forward(self, pixel_values, labels=None):
        """
        Forward pass of the model.

        Args:
            pixel_values: Tensor of shape (batch_size, 3, height, width)
            labels: Optional tensor of shape (batch_size,) with values in [0, 100]

        Returns:
            Dictionary with 'loss' and 'logits' keys
        """
        # Get CLIP vision features
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)

        # Use pooled output (CLS token) if available, otherwise pool manually
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            image_features = vision_outputs.pooler_output
        else:
            # Use last hidden state and take mean pooling
            image_features = vision_outputs.last_hidden_state.mean(dim=1)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Pass through MLP to get completion percentage (0-1)
        completion_ratio = self.mlp(image_features).squeeze(-1)

        # Scale to 0-100
        predictions = completion_ratio * 100.0

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Convert labels to 0-1 range for loss calculation
            labels_normalized = labels / 100.0
            loss = nn.functional.mse_loss(completion_ratio, labels_normalized)

        return {
            'loss': loss,
            'logits': predictions  # Called 'logits' for compatibility but these are predictions
        }


def load_dataset_from_cache(cache_dir, dataset_path, split='val'):
    """Load pre-extracted images from cache directory."""

    # Generate unique cache subdirectory based on dataset path
    if dataset_path=="ID":
        dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5' 
    elif dataset_path=="OOD":
        dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im224.hdf5'
    dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    image_dir = Path(cache_dir) / dataset_hash

    # Check if images have been extracted
    cache_marker = image_dir / '.cache_complete'
    metadata_file = image_dir / 'metadata.json'

    if not cache_marker.exists():
        raise RuntimeError(
            f"Images not extracted yet! Please run:\n"
            f"python extract_images_from_hdf5.py --hdf5-path '{dataset_path}' --output-dir '{cache_dir}'\n"
            f"Expected cache directory: {image_dir}"
        )

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Check if split information exists
    if 'split' not in metadata:
        raise RuntimeError(
            f"Split information not found in metadata! Please run:\n"
            f"python split_demos.py --cache-dir '{cache_dir}' --dataset-path '{dataset_path}'\n"
        )

    split_info = metadata['split']

    split_info = metadata.get('split', {})
    if split == 'all_demos':
        print(f"USING ALL DEMOS")
        print(f"USING ALL DEMOS")
        print(f"USING ALL DEMOS")
        print(f"USING ALL DEMOS")
        selected_demos = list(metadata['demos'].keys())
    else:
        selected_demos = split_info[split]

    print(f"Loading pre-extracted images from {image_dir} for {split} split")

    if not selected_demos:
        raise ValueError(f"No demos found for {split} split!")

    data = []
    for demo_name in selected_demos:
        demo_info = metadata['demos'][demo_name]
        num_frames = demo_info['num_frames']
        demo_dir = image_dir / demo_name

        # Sample frames at regular intervals (every 16 frames)
        for idx in range(0, num_frames, 16):
            idx_image_path = str(demo_dir / f'frame_{idx:04d}.png')
            progress = float(idx / num_frames * 100)  # Progress as 0-100

            # Load and convert image to RGB if needed
            image = Image.open(idx_image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            data.append({
                "image": image,
                "label": progress,
                "demo_name": demo_name,
                "frame_idx": idx,
                "image_path": idx_image_path
            })

    print(f"Loaded {len(data)} images from {len(selected_demos)} {split} demos")

    return Dataset.from_list(data)


def preprocess_function(examples, processor, target_size=224):
    """Preprocess images for CLIP model."""

    # Process images using CLIP processor
    images = examples['image']
    if not isinstance(images, list):
        images = [images]
        labels = [examples['label']]
    else:
        labels = examples['label']

    # Resize images to target size
    resized_images = []
    for img in images:
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.BILINEAR)
        resized_images.append(img)

    # Use CLIP processor to prepare images
    inputs = processor(images=resized_images, return_tensors="pt")

    # Extract pixel_values and ensure it's properly shaped
    pixel_values = inputs['pixel_values']

    # For single images, ensure we don't have an extra dimension
    if len(images) == 1 and pixel_values.dim() == 4:
        pixel_values = pixel_values[0]  # Remove batch dimension for single image

    return {
        'pixel_values': pixel_values,
        'labels': torch.tensor(labels[0] if len(labels) == 1 else labels, dtype=torch.float32)
    }


class SimpleEvaluator:
    def __init__(self, args: EvaluationArgs):
        self.args = args
        self.device = torch.device(args.device)

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.model_path) / f"evaluation_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.load_model()

    def load_model(self):
        """Load the trained model and processor"""
        print(f"Loading model from {self.args.model_path}")

        # Check if model checkpoint exists
        model_file = Path(self.args.model_path) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = Path(self.args.model_path) / "model.safetensors"
            if not model_file.exists():
                raise FileNotFoundError(f"No model checkpoint found at {self.args.model_path}")

        # Initialize model
        self.model = TaskCompletionModel(
            clip_model_name=self.args.clip_model_name,
            hidden_dim=self.args.hidden_dim,
            dropout_rate=self.args.dropout_rate
        )

        # Load state dict
        if model_file.suffix == ".bin":
            state_dict = torch.load(model_file, map_location=self.device)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(model_file)

        # Handle potential key mismatches
        if any(k.startswith("module.") for k in state_dict.keys()):
            # Remove "module." prefix if present (from DataParallel)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Get processor
        self.processor = CLIPProcessor.from_pretrained(self.args.clip_model_name)

        print("Model loaded successfully")

    def evaluate_batch(self, batch):
        """Evaluate a batch of samples"""
        with torch.no_grad():
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)

            predictions = outputs['logits'].cpu().numpy()
            labels = labels.cpu().numpy()
            loss = outputs['loss'].item() if outputs['loss'] is not None else None

        return predictions, labels, loss

    def calculate_metrics(self, all_predictions, all_labels):
        """Calculate evaluation metrics"""
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Calculate basic metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(mse)

        # Calculate accuracy within tolerance
        within_tolerance = np.abs(predictions - labels) <= self.args.tolerance
        accuracy = np.mean(within_tolerance)

        # Calculate correlation
        correlation = np.corrcoef(predictions, labels)[0, 1]

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "accuracy": float(accuracy),
            "correlation": float(correlation),
            "total_samples": len(predictions),
            "tolerance": self.args.tolerance
        }

    def run_evaluation(self):
        """Run the full evaluation"""
        print("Loading dataset...")
        dataset = load_dataset_from_cache(
            self.args.cache_dir,
            self.args.dataset_path,
            split=self.args.dataset_split
        )

        # Limit samples if specified
        if self.args.num_samples:
            dataset = dataset.select(range(min(self.args.num_samples, len(dataset))))

        print(f"Evaluating on {len(dataset)} samples from {self.args.dataset_split} split")

        # Preprocess dataset
        print("Preprocessing dataset...")
        dataset = dataset.map(
            lambda x: preprocess_function(x, self.processor),
            batched=False,
            remove_columns=['image']
        )

        # Set format for PyTorch
        dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # Evaluate
        all_predictions = []
        all_labels = []
        all_losses = []
        results = []

        print("Running evaluation...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            predictions, labels, loss = self.evaluate_batch(batch)

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            if loss is not None:
                all_losses.append(loss)

            # Store detailed results
            batch_start = batch_idx * self.args.batch_size
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                idx = batch_start + i
                if idx < len(dataset):
                    result = {
                        "index": idx,
                        "demo_name": dataset[idx].get('demo_name', 'unknown'),
                        "frame_idx": dataset[idx].get('frame_idx', -1),
                        "ground_truth": float(label),
                        "prediction": float(pred),
                        "error": float(pred - label),
                        "abs_error": float(abs(pred - label)),
                        "within_tolerance": bool(abs(pred - label) <= self.args.tolerance)
                    }
                    results.append(result)

        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        if all_losses:
            metrics["avg_loss"] = float(np.mean(all_losses))

        # Prepare summary
        summary = {
            "model_path": str(self.args.model_path),
            "dataset_path": self.args.dataset_path,
            "dataset_split": self.args.dataset_split,
            "num_samples": len(dataset),
            "batch_size": self.args.batch_size,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "args": vars(self.args)
        }

        # Save results
        self.save_results(summary, results)

        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {self.args.model_path}")
        print(f"Dataset: {self.args.dataset_split} split")
        print(f"Samples evaluated: {metrics['total_samples']}")
        print(f"\nMetrics:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Accuracy (±{self.args.tolerance}%): {metrics['accuracy']*100:.2f}%")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        if 'avg_loss' in metrics:
            print(f"  Average Loss: {metrics['avg_loss']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")

        return summary

    def save_results(self, summary: Dict, predictions: List[Dict]):
        """Save evaluation results to files"""
        # Save summary
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed predictions if requested
        if self.args.save_predictions and predictions:
            # Save as JSON
            predictions_path = self.output_dir / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(predictions, f, indent=2)

            # Save as CSV for easier analysis
            df = pd.DataFrame(predictions)
            csv_path = self.output_dir / "predictions.csv"
            df.to_csv(csv_path, index=False)

            # Create error distribution plot if matplotlib is available
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Scatter plot: predictions vs ground truth
                ax = axes[0, 0]
                ax.scatter(df['ground_truth'], df['prediction'], alpha=0.5, s=1)
                ax.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
                ax.set_xlabel('Ground Truth (%)')
                ax.set_ylabel('Prediction (%)')
                ax.set_title('Predictions vs Ground Truth')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Error distribution
                ax = axes[0, 1]
                ax.hist(df['error'], bins=50, edgecolor='black')
                ax.set_xlabel('Error (prediction - ground truth)')
                ax.set_ylabel('Frequency')
                ax.set_title('Error Distribution')
                ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Absolute error distribution
                ax = axes[1, 0]
                ax.hist(df['abs_error'], bins=50, edgecolor='black')
                ax.axvline(x=self.args.tolerance, color='r', linestyle='--',
                          label=f'Tolerance (±{self.args.tolerance}%)')
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Frequency')
                ax.set_title('Absolute Error Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Error vs ground truth
                ax = axes[1, 1]
                ax.scatter(df['ground_truth'], df['abs_error'], alpha=0.5, s=1)
                ax.axhline(y=self.args.tolerance, color='r', linestyle='--',
                          label=f'Tolerance (±{self.args.tolerance}%)')
                ax.set_xlabel('Ground Truth (%)')
                ax.set_ylabel('Absolute Error')
                ax.set_title('Absolute Error vs Ground Truth')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_path = self.output_dir / "evaluation_plots.png"
                plt.savefig(plot_path, dpi=150)
                plt.close()

                print(f"Plots saved to {plot_path}")

            except ImportError:
                pass  # matplotlib not available


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP + MLP task completion model")

    # Add arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--dataset_path", type=str,
                       default="ID",
                       help="Path to the HDF5 dataset file")
    parser.add_argument("--cache_dir", type=str, default="/workspace/image_cache",
                       help="Directory for cached images")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32",
                       help="Name of the CLIP model to use")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for the MLP")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout rate for the MLP")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--tolerance", type=float, default=10.0,
                       help="Tolerance for accuracy calculation (± percentage points)")
    parser.add_argument("--dataset_split", type=str, default="val_demos",
                       help="Dataset split to evaluate on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save individual predictions to file")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")

    args = parser.parse_args()

    # Convert to dataclass
    eval_args = EvaluationArgs(**vars(args))

    # Run evaluation
    evaluator = SimpleEvaluator(eval_args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()