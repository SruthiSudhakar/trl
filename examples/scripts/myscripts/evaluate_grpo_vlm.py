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
Evaluation script for GRPO VLM models trained with grpo_vlm.py

Usage:
CUDA_VISIBLE_DEVICES=2 python3 examples/scripts/evaluate_grpo_vlm.py \
    --base_model_name_or_path /workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct \
    --model_name_or_path /workspace/hf_trl/trl/outputs/grpo-Qwen2.5-VL-7B-Instruct_20250917_173822/checkpoint-16100 \
    --batch_size 300 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --temperature 0.1 \
    --tolerance 5 \
    --dataset_path ID \
    --dataset_split train_demos

    --dataset_path /workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im224.hdf5 \
    --dataset_path /workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/important/demo_gentex_im128_randcams_new_images_val_kbpckt_firsthalf.hdf5 \

"""

import argparse
import json
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import pdb
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig
from PIL import Image
import qwen_vl_utils
from datetime import datetime

task_description = "Pick and place an object from the sink to the plate on the counter"
SYSTEM_PROMPT =f"""You are an expert roboticist tasked to predict task completion "
    percentage for a frame of a robot for the task of {task_description}.
    The task completion percentages are between 0 and 100, where 100
    corresponds to full task completion.
"""
problem = f"""Here is a frame showing the robot performing the task. The frame shows the robot's current state.\n
    For the task of {task_description}, output the task completion percentage (an integer from 0-100) for this current frame. 
    Please answer the question in the following format: <think> your reasoning </think> <answer> your answer </answer>. 
"""

@dataclass
class EvaluationArgs:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to the trained model checkpoint"}
    )
    base_model_name_or_path: str = field(
        default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path to the base model (if using PEFT/LoRA)"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation"}
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Data type for model (float32, float16, bfloat16)"}
    )
    max_prompt_length: int = field(
        default=2048,
        metadata={"help": "Maximum prompt length"}
    )
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "Maximum completion length"}
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for generation (lower = more deterministic)"}
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for generation"}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to evaluate (None = all)"}
    )
    dataset_split: str = field(
        default="test",
        metadata={"help": "Dataset split to evaluate on (train/test)"}
    )
    dataset_path: str = field(
        default="/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5",
        metadata={"help": "dataset path to hdf5"}
    )
    save_completions: bool = field(
        default=True,
        metadata={"help": "Save individual completions to file"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to run evaluation on"}
    )
    tolerance: int = field(
        default=10,
        metadata={"help": "Tolerance value for accuracy evaluation (±tolerance from ground truth)"}
    )
    enable_ordering_eval: bool = field(
        default=True,
        metadata={"help": "Enable secondary evaluation that checks if image pairs are ordered correctly"}
    )


class GRPOEvaluator:
    def __init__(self, args: EvaluationArgs):
        self.args = args
        self.output_dir = self.args.model_name_or_path or self.args.base_model_name_or_path
        self.output_dir += '/eval_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.load_model()
        
        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self):
        """Load the trained model and processor"""
        print(f"Loading model from {self.args.model_name_or_path or self.args.base_model_name_or_path}")
        
        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        dtype = dtype_map.get(self.args.dtype, "auto")
        
        if self.args.model_name_or_path and os.path.exists(os.path.join(self.args.model_name_or_path, "adapter_config.json")):
            # Check if this is a PEFT model        
            # Load base model first
            base_model_path = self.args.base_model_name_or_path or self.args.model_name_or_path
            print(f"Loading base model from {base_model_path}")
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load PEFT adapter
            print(f"Loading PEFT adapter from {self.args.model_name_or_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.args.model_name_or_path,
                is_trainable=False,
            )
            
            # Use base model path for processor
            processor_path = base_model_path
        elif self.args.model_name_or_path:
            # Load full finetuned model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.args.model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=True,
            )
            processor_path = self.args.model_name_or_path
        else:
            # Load base model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.args.base_model_name_or_path,
                torch_dtype=dtype if dtype != "auto" else None,
                device_map="auto",
                trust_remote_code=True,
            )
            processor_path = self.args.base_model_name_or_path

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        
        # Set model to eval mode
        self.model.eval()
        print("Model loaded successfully")
    
    def load_dataset(self):
        """Load the evaluation dataset"""
        # Using the same dataset loading logic as training script
        dataset_path = self.args.dataset_path
        if dataset_path=="ID":
            dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5' 
        elif dataset_path=="OOD":
            dataset_path = '/workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im224.hdf5'
        cache_dir = Path('/workspace/image_cache')
        
        # Generate unique cache subdirectory based on dataset path
        dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
        image_dir = cache_dir / dataset_hash
        
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
        
        # Get split information
        split_info = metadata.get('split', {})
        if self.args.dataset_split == 'all_demos':
            print(f"USING ALL DEMOS")
            print(f"USING ALL DEMOS")
            print(f"USING ALL DEMOS")
            print(f"USING ALL DEMOS")
            selected_demos = list(metadata['demos'].keys())
        else:
            selected_demos = split_info[self.args.dataset_split]
        
        # Limit number of samples if specified
        if self.args.num_samples:
            selected_demos = selected_demos[:self.args.num_samples]
        
        print(f"Evaluating on {len(selected_demos)} samples from {self.args.dataset_split} split")
        
        # Create dataset
        data = []
        for demo_name in selected_demos:
            demo_info = metadata['demos'][demo_name]
            num_frames = demo_info['num_frames']
            demo_dir = image_dir / demo_name
            
            for idx in range(0, num_frames, 16):
                idx_image_path = str(demo_dir / f'frame_{idx:04d}.png')
                completion_percentage = str(int(idx / num_frames * 100))  # Progress as 0-100
                entry = {
                    "demo_name": demo_name,
                    "frame_idx": idx,
                    "images": idx_image_path,  # Current frame only
                    "problem": problem,
                    "solution": str(completion_percentage),
                    "task_description": task_description,
                }
                data.append(entry)
        
        return data
    
    def evaluate_ordering(self, results_by_demo: Dict[str, List[Dict]]) -> Dict:
        """Evaluate if pairs of images within each demo are ordered correctly.

        A pair is considered correctly ordered if the later frame has a higher or equal
        predicted completion percentage than the earlier frame.

        Returns:
            Dictionary with ordering evaluation metrics
        """
        ordering_results = []
        demo_ordering_accuracy = {}

        for demo_name, demo_results in results_by_demo.items():
            # Sort results by frame index
            sorted_results = sorted(demo_results, key=lambda x: x['frame_idx'])

            # Skip if not enough valid predictions
            valid_results = [r for r in sorted_results if r['extracted_answer'] is not None]
            if len(valid_results) < 2:
                continue

            demo_correct_pairs = 0
            demo_total_pairs = 0

            # Check ordering for consecutive pairs
            for i in range(len(valid_results) - 1):
                earlier = valid_results[i]
                later = valid_results[i + 1]

                # Check if ordering is correct (later >= earlier)
                is_ordered_correctly = later['extracted_answer'] > earlier['extracted_answer']

                ordering_results.append({
                    'demo_name': demo_name,
                    'earlier_frame': earlier['frame_idx'],
                    'later_frame': later['frame_idx'],
                    'earlier_predicted': earlier['extracted_answer'],
                    'later_predicted': later['extracted_answer'],
                    'earlier_gt': earlier['ground_truth'],
                    'later_gt': later['ground_truth'],
                    'is_ordered_correctly': is_ordered_correctly,
                    'prediction_diff': later['extracted_answer'] - earlier['extracted_answer']
                })

                if is_ordered_correctly:
                    demo_correct_pairs += 1
                demo_total_pairs += 1

            if demo_total_pairs > 0:
                demo_ordering_accuracy[demo_name] = demo_correct_pairs / demo_total_pairs

        # Calculate overall metrics
        total_correct = sum(1 for r in ordering_results if r['is_ordered_correctly'])
        total_pairs = len(ordering_results)
        overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0

        # Calculate statistics on prediction differences
        pred_diffs = [r['prediction_diff'] for r in ordering_results]

        return {
            'overall_ordering_accuracy': overall_accuracy,
            'total_correct_pairs': total_correct,
            'total_pairs': total_pairs,
            'demo_ordering_accuracy': demo_ordering_accuracy,
            'mean_prediction_diff': np.mean(pred_diffs) if pred_diffs else 0,
            'std_prediction_diff': np.std(pred_diffs) if pred_diffs else 0,
            'negative_diff_count': sum(1 for d in pred_diffs if d < 0),
            'detailed_results': ordering_results
        }

    def evaluate_accuracy(self, completion: str, ground_truth: str) -> tuple:
        """Evaluate if the completion matches ground truth
        Returns: (accuracy, extracted_answer)
        """
        try:
            # Extract answer from completion
            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.IGNORECASE | re.DOTALL)
            if not answer_match:
                return 0.0, None
            
            answer_text = answer_match.group(1).strip()
            number_match = re.search(r'\b(\d{1,3})\b', answer_text)
            if not number_match:
                return 0.0, None
            
            extracted_number = int(number_match.group(1))
            if not (0 <= extracted_number <= 100):
                return 0.0, extracted_number
            
            # Check if within tolerance (±tolerance)
            gt_value = int(ground_truth)
            if abs(extracted_number - gt_value) <= self.args.tolerance:
                return 1.0, extracted_number
            else:
                return 0.0, extracted_number
        except Exception as e:
            print(f"Error evaluating completion: {e}")
            return 0.0, None
    
    def generate_completion(self, batch: List[Dict]) -> List[str]:
        """Generate completions for a batch of inputs"""
        # Prepare batch inputs
        prompts = []
        images_batch = []
        for item in batch:
            conversation  = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["images"]},
                        {"type": "text", "text": problem},
                    ],
                }
            ]
            image_input, video_inputs = qwen_vl_utils.process_vision_info(conversation)
            images_batch.append(image_input)
            # Process inputs
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            
        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
        ).to(self.device)
        
        # Generate completions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=self.args.temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        completions = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            completion = self.processor.decode(generated_tokens, skip_special_tokens=True)
            completions.append(completion)
        
        return completions
    
    def run_evaluation(self):
        """Run the full evaluation"""
        print("Loading dataset...")
        dataset = self.load_dataset()
        
        # Prepare for batch processing
        results = []
        all_completions = []
        accuracies = []
        results_by_demo = {}  # For ordering evaluation
        
        # Process in batches
        batch_size = self.args.batch_size
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        print(f"Starting evaluation on {len(dataset)} samples...")
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            # Generate completions
            completions = self.generate_completion(batch)
            
            # Evaluate each completion
            for item, completion in zip(batch, completions):
                accuracy, extracted_answer = self.evaluate_accuracy(completion, item["solution"])
                accuracies.append(accuracy)
                
                result = {
                    "demo_name": item["demo_name"],
                    "frame_idx": item["frame_idx"],
                    "ground_truth": item["solution"],
                    "extracted_answer": extracted_answer,
                    "completion": completion,
                    "accuracy": accuracy,
                    "task_description": item["task_description"],
                }
                results.append(result)
                all_completions.append(result)

                # Group results by demo for ordering evaluation
                if self.args.enable_ordering_eval:
                    if item["demo_name"] not in results_by_demo:
                        results_by_demo[item["demo_name"]] = []
                    results_by_demo[item["demo_name"]].append(result)
        
        # Calculate metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Run ordering evaluation if enabled
        ordering_metrics = None
        if self.args.enable_ordering_eval and results_by_demo:
            print("\nRunning ordering evaluation...")
            ordering_metrics = self.evaluate_ordering(results_by_demo)
        
        # Prepare summary
        summary = {
            "model_path": self.args.model_name_or_path or self.args.base_model_name_or_path,
            "num_samples": len(dataset),
            "batch_size": self.args.batch_size,
            "temperature": self.args.temperature,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "total_correct": sum(accuracies),
            "total_samples": len(accuracies),
            "tolerance": self.args.tolerance,
            "timestamp": datetime.now().isoformat(),
            "args": vars(self.args),
        }

        # Add ordering metrics to summary if available
        if ordering_metrics:
            summary["ordering_evaluation"] = {
                "overall_accuracy": ordering_metrics["overall_ordering_accuracy"],
                "total_correct_pairs": ordering_metrics["total_correct_pairs"],
                "total_pairs": ordering_metrics["total_pairs"],
                "mean_prediction_diff": ordering_metrics["mean_prediction_diff"],
                "std_prediction_diff": ordering_metrics["std_prediction_diff"],
                "negative_diff_count": ordering_metrics["negative_diff_count"],
                "num_demos_evaluated": len(ordering_metrics["demo_ordering_accuracy"])
            }
        
        # Save results
        self.save_results(summary, all_completions, ordering_metrics)
        
        # Print summary
        print("\n" + "="*50)
        print("GRPO EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {self.args.model_name_or_path or self.args.base_model_name_or_path}")
        print(f"Samples evaluated: {len(dataset)}")
        print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Correct predictions: {sum(accuracies)}/{len(accuracies)} ({mean_accuracy*100:.2f}%)")
        print(f"Tolerance: ±{self.args.tolerance}")

        if ordering_metrics:
            print("\n" + "-"*50)
            print("ORDERING EVALUATION SUMMARY")
            print("-"*50)
            print(f"Ordering Accuracy: {ordering_metrics['overall_ordering_accuracy']:.4f}")
            print(f"Correctly ordered pairs: {ordering_metrics['total_correct_pairs']}/{ordering_metrics['total_pairs']} ({ordering_metrics['overall_ordering_accuracy']*100:.2f}%)")
            print(f"Mean prediction difference: {ordering_metrics['mean_prediction_diff']:.2f} ± {ordering_metrics['std_prediction_diff']:.2f}")
            print(f"Pairs with negative difference (wrong order): {ordering_metrics['negative_diff_count']}")
            print(f"Demos evaluated: {len(ordering_metrics['demo_ordering_accuracy'])}")

        print(f"\nResults saved to: {self.output_dir}")
        
        return summary
    
    def save_results(self, summary: Dict, completions: List[Dict], ordering_metrics: Optional[Dict] = None):
        """Save evaluation results to files"""
        # Save summary
        summary_path = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed completions if requested
        if self.args.save_completions:
            completions_path = os.path.join(self.output_dir, "completions.json")
            with open(completions_path, "w") as f:
                json.dump(completions, f, indent=2)
            
            # Also save as CSV for easier analysis
            df = pd.DataFrame(completions)
            csv_path = os.path.join(self.output_dir, "completions.csv")
            df.to_csv(csv_path, index=False)

            # Save ordering evaluation details if available
            if ordering_metrics and self.args.enable_ordering_eval:
                # Save detailed ordering results
                ordering_path = os.path.join(self.output_dir, "ordering_evaluation.json")
                with open(ordering_path, "w") as f:
                    json.dump(ordering_metrics, f, indent=2)

                # Save ordering results as CSV
                if ordering_metrics['detailed_results']:
                    ordering_df = pd.DataFrame(ordering_metrics['detailed_results'])
                    ordering_csv_path = os.path.join(self.output_dir, "ordering_results.csv")
                    ordering_df.to_csv(ordering_csv_path, index=False)

                # Save per-demo accuracy
                if ordering_metrics['demo_ordering_accuracy']:
                    demo_accuracy_df = pd.DataFrame([
                        {"demo_name": k, "ordering_accuracy": v}
                        for k, v in ordering_metrics['demo_ordering_accuracy'].items()
                    ])
                    demo_accuracy_path = os.path.join(self.output_dir, "demo_ordering_accuracy.csv")
                    demo_accuracy_df.to_csv(demo_accuracy_path, index=False)
        
        print(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO VLM model")
    
    # Add arguments
    parser.add_argument("--model_name_or_path", type=str, required=False,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--base_model_name_or_path", type=str, required=True, 
                        default="/workspace/cosmos-reason1/data/huggingface/transformers/Qwen2.5-VL-7B-Instruct",
                       help="Path to the base model (if using PEFT/LoRA)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16", "auto"],
                       help="Data type for model")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                       help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024,
                       help="Maximum completion length")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for generation")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")
    parser.add_argument("--dataset_split", type=str, default="test",
                       help="Dataset split to evaluate on")
    # /workspace/guided_diffusion_policy/externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/important/demo_gentex_im128_randcams_new_images_val_kbpckt_firsthalf.hdf5
    parser.add_argument("--dataset_path", type=str, default="ID",
                       help="dataset_path")
    parser.add_argument("--save_completions", action="store_true", default=True,
                       help="Save individual completions to file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--tolerance", type=int, default=10,
                       help="Tolerance value for accuracy evaluation (±tolerance from ground truth)")
    parser.add_argument("--enable_ordering_eval", action="store_true", default=True,
                       help="Enable secondary evaluation that checks if image pairs are ordered correctly")
    
    args = parser.parse_args()
    
    # Convert to dataclass
    eval_args = EvaluationArgs(**vars(args))
    
    # Run evaluation
    evaluator = GRPOEvaluator(eval_args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()