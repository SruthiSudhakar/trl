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

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "Pillow>=9.4.0",
#     "peft",
#     "trackio",
#     "kernels",
#     "transformers",
#     "torch",
#     "torchvision",
# ]
# ///

"""
Task Completion Prediction using CLIP + 2-Layer MLP

This script trains a simple model that combines CLIP vision encoder with a 2-layer MLP
to predict task completion percentage (0-100) from robot observation images.

Example usage:
CUDA_VISIBLE_DEVICES=1,5,6,7 accelerate launch --num_processes=4 --multi_gpu --gpu_ids=1,5,6,7 \
    examples/scripts/sft_simple.py \
    --output_dir "outputs/clip-mlp-task-completion_$(date +%Y%m%d_%H%M%S)" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10000 \
    --learning_rate 1e-3 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --report_to wandb

CUDA_VISIBLE_DEVICES=1,5,6,7 accelerate launch --num_processes=4 --multi_gpu --gpu_ids=1,5,6,7 \
    examples/scripts/sft_simple.py \
    --output_dir "outputs/clip-mlp-task-completion_$(date +%Y%m%d_%H%M%S)" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10000 \
    --learning_rate 1e-4 \
    --logging_steps 100 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --report_to wandb
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from datasets import load_dataset, Dataset
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
import numpy as np
import wandb


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

@dataclass
class ScriptArguments:
    """Arguments for the training script."""
    dataset_path: str = field(
        default="/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5",
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
    wandb_project: str = field(
        default="clip-mlp-task-completion",
        metadata={"help": "Weights & Biases project name"}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases entity/team name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name"}
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
        # Use the vision model's hidden size directly since we're using pooled features
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


def load_dataset_from_cache(cache_dir, dataset_path, split='train'):
    """Load pre-extracted images from cache directory."""

    # Generate unique cache subdirectory based on dataset path
    dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    image_dir = Path(cache_dir) / dataset_hash

    # Check if images have been extracted
    cache_marker = image_dir / '.cache_complete'
    metadata_file = image_dir / 'metadata.json'

    if not cache_marker.exists():
        # Try hardcoded path as fallback
        image_dir = Path("/workspace/image_cache/7b998b96")
        metadata_file = image_dir / 'metadata.json'
        if not metadata_file.exists():
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

    if split == 'val':
        selected_demos = split_info['val_demos']
    else:  # train
        selected_demos = split_info['train_demos']

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
            progress = int(idx / num_frames * 100)  # Progress as 0-100

            # Load and convert image to RGB if needed
            image = Image.open(idx_image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            data.append({
                "image": image,
                "label": progress
            })

    print(f"Loaded {len(data)} images from {len(selected_demos)} {split} demos")

    return Dataset.from_list(data)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred

    # Calculate MSE and MAE
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(mse)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


class ImageRegressionTrainer(Trainer):
    """Custom trainer for image regression tasks."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the loss for regression."""
        labels = inputs.pop("labels")
        pixel_values = inputs.pop("pixel_values")

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss


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


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Set some default training arguments if not provided
    if training_args.output_dir is None:
        training_args.output_dir = "outputs/clip-mlp-task-completion"

    # Adjust training arguments for regression
    training_args.remove_unused_columns = False
    training_args.label_names = ['labels']

    # Initialize wandb if report_to includes "wandb" (only on main process)
    if "wandb" in training_args.report_to and training_args.local_rank in [-1, 0]:
        wandb.init(
            project=script_args.wandb_project,
            entity=script_args.wandb_entity,
            name=script_args.wandb_run_name or training_args.output_dir.split("/")[-1],
            config={
                "dataset_path": script_args.dataset_path,
                "clip_model": script_args.clip_model_name,
                "hidden_dim": script_args.hidden_dim,
                "dropout_rate": script_args.dropout_rate,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            }
        )

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset_from_cache(
        script_args.cache_dir,
        script_args.dataset_path,
        split='train'
    )

    # Create train/validation split
    dataset_dict = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Initialize model
    print(f"Initializing model with CLIP: {script_args.clip_model_name}")
    model = TaskCompletionModel(
        clip_model_name=script_args.clip_model_name,
        hidden_dim=script_args.hidden_dim,
        dropout_rate=script_args.dropout_rate
    )

    # Get processor for preprocessing
    processor = CLIPProcessor.from_pretrained(script_args.clip_model_name)

    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, processor),
        batched=False,
        remove_columns=['image', 'label']
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, processor),
        batched=False,
        remove_columns=['image', 'label']
    )

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    eval_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

    # Initialize trainer
    trainer = ImageRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,# if training_args.eval_strategy != "no" else None,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

    # Log final metrics to wandb (only on main process)
    if "wandb" in training_args.report_to and training_args.local_rank in [-1, 0]:
        # Evaluate final model
        final_metrics = trainer.evaluate()
        wandb.log({f"final/{k}": v for k, v in final_metrics.items()})

        # Log model architecture info
        wandb.config.update({
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })

    # Push to hub if requested
    if training_args.push_to_hub:
        trainer.push_to_hub()

    # Finish wandb run (only on main process)
    if "wandb" in training_args.report_to and training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    main()