"""Re-evaluate a saved checkpoint on a (possibly different) eval set.

Replicates eval pair construction from sft_vlm_lego.py and runs the same
sign-accuracy callback logic offline. Single-GPU.

Example:
CUDA_VISIBLE_DEVICES=7 python examples/scripts/myscripts/reeval_checkpoint.py \
--checkpoint outputs/PnPRedLegoToBrownBowl_20260508_205501_moredata/checkpoint-200 \
--dataset_root /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl \
--eval_failure_indices 18-20,318-320 \
--eval_success_indices 18-20,318-320 \
--max_pixels 960x540

"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_lego import (  # noqa: E402
    LEGO_SYSTEM_PROMPT,
    LEGO_USER_PROMPT_TEMPLATE,
    build_pairs_for_demos,
    load_lego_demos,
    parse_indices,
)
from sft_vlm_overlay_regression_v2 import TASK_TOKENS  # noqa: E402
from video_frame_utils import extract_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIGNED_INT_RE = re.compile(r"-?\d+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="/proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl")
    p.add_argument("--task_name", default="PnPRedLegoToBrownBowl")
    p.add_argument("--eval_failure_indices", default="18-20,318-320")
    p.add_argument("--eval_success_indices", default="18-20,318-320")
    p.add_argument("--compare_interval", default="4,8,12,16")
    p.add_argument("--train_sample_interval", type=int, default=8)
    p.add_argument("--failure_last_frac", type=float, default=0.50)
    p.add_argument("--failure_min_frames", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=200)
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--source_split_threshold", type=int, default=100,
                   help="Demo idx >= this is treated as the 'new' eval-source group; below is 'old'")
    args = p.parse_args()

    intervals = [int(x) for x in args.compare_interval.split(",")]
    task_token = TASK_TOKENS[args.task_name]

    succ_idxs = parse_indices(args.eval_success_indices)
    fail_idxs = parse_indices(args.eval_failure_indices)
    successes = load_lego_demos(args.dataset_root, succ_idxs, "success")
    failures = load_lego_demos(args.dataset_root, fail_idxs, "failure")

    # Use the same rng seed as the training script's eval split (Random(43)).
    eval_pairs = build_pairs_for_demos(
        successes, failures, intervals, args.train_sample_interval, task_token,
        args.failure_last_frac, args.failure_min_frames,
        rng=random.Random(43),
    )
    logger.info(f"Built {len(eval_pairs)} eval pairs")

    # Match the training callback: deterministic shuffle + truncate to max_pairs.
    rng = random.Random(1234)
    indices = list(range(len(eval_pairs)))
    rng.shuffle(indices)
    n = min(args.max_pairs, len(indices))
    indices = indices[:n]

    # Set processor pixel budget to match training.
    w, h = (int(x) for x in args.max_pixels.split("x"))
    max_pixels = w * h
    processor = AutoProcessor.from_pretrained(args.checkpoint, max_pixels=max_pixels)

    logger.info(f"Loading model from {args.checkpoint}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()

    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})
    bucket_by_source = defaultdict(lambda: {"correct": 0, "total": 0, "unparsed": 0})

    for k, i in enumerate(indices):
        ex = eval_pairs[i]
        bucket = ex["bucket"]
        target_ans = int(ex["correct_answer"])
        demo_id = ex.get("demo_id", -1)
        src = "new" if demo_id >= args.source_split_threshold else "old"
        user_text = LEGO_USER_PROMPT_TEMPLATE.format(task_token=task_token)

        f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
        f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])

        mm = [
            {"role": "system", "content": [{"type": "text", "text": LEGO_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f1},
                    {"type": "image", "image": f2},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text = processor.apply_chat_template(mm, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[[f1, f2]], return_tensors="pt", padding=True)
        inputs = {kk: v.to(model.device) if hasattr(v, "to") else v for kk, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        m = SIGNED_INT_RE.search(gen)
        pred_ans = int(m.group(0)) if m else None

        for d in (buckets[bucket], bucket_by_source[(bucket, src)]):
            d["total"] += 1
            if pred_ans is None:
                d["unparsed"] += 1
            elif (pred_ans > 0) == (target_ans > 0):
                d["correct"] += 1

        if (k + 1) % 25 == 0:
            logger.info(f"  ... {k + 1}/{len(indices)}")

    print("\n=== Per-bucket sign accuracy ===")
    total = sum(b["total"] for b in buckets.values())
    correct = sum(b["correct"] for b in buckets.values())
    print(f"overall: {correct}/{total} = {correct / max(total, 1):.4f}")
    for k in sorted(buckets):
        b = buckets[k]
        print(f"  {k}: {b['correct']}/{b['total']} = {b['correct'] / max(b['total'], 1):.4f}  (unparsed={b['unparsed']})")

    print("\n=== fail_vs_succ split by demo source (old=18-20, new=318-320) ===")
    for k in sorted(bucket_by_source):
        if k[0] != "fail_vs_succ":
            continue
        b = bucket_by_source[k]
        print(f"  {k[0]} / {k[1]}: {b['correct']}/{b['total']} = {b['correct'] / max(b['total'], 1):.4f}")

    print("\n=== succ_vs_succ split by demo source ===")
    for k in sorted(bucket_by_source):
        if not k[0].startswith("succ_vs_succ"):
            continue
        b = bucket_by_source[k]
        print(f"  {k[0]} / {k[1]}: {b['correct']}/{b['total']} = {b['correct'] / max(b['total'], 1):.4f}")


if __name__ == "__main__":
    main()
