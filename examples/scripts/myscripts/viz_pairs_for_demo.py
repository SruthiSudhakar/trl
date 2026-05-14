"""
Visualize every pair that build_pairs_for_demos would construct for a given
demo index (success_N and/or failure_N), with no model inference. Useful for
sanity-checking the windowing / alignment behavior of fail_vs_succ pairs.

Usage:
python examples/scripts/myscripts/viz_pairs_for_demo.py \
--indices 110 \
--compare_interval 2,4,8,16 \
--out_dir tmp/pair_viz_110 \
--buckets fail_vs_succ \
--failure_last_frac 0.5
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_upright_bottle import (  # noqa: E402
    build_pairs_for_demos,
    load_upright_bottle_demos,
    parse_indices,
)
from sft_vlm_overlay_regression_v2 import TASK_TOKENS  # noqa: E402
from video_frame_utils import extract_frame  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/proj/vondrick3/datasets/expert_data_jgd_UprightBottle")
    p.add_argument("--task_name", default="UprightBottle")
    p.add_argument("--indices", required=True,
                   help="indices to visualize (comma/range), used for BOTH success and failure")
    p.add_argument("--compare_interval", default="4,8,12,16")
    p.add_argument("--sample_interval", type=int, default=4)
    p.add_argument("--failure_last_frac", type=float, default=0.75)
    p.add_argument("--failure_min_frames", type=int, default=8)
    p.add_argument("--buckets", default="all",
                   help="comma list: 'all', 'fail_vs_succ', 'succ_vs_succ'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def render(ex, out_path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    try:
        f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
        f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])
        axL.imshow(f1)
        axR.imshow(f2)
    except Exception as e:
        axL.text(0.5, 0.5, f"frame1 fail\n{e}", ha="center", va="center",
                 transform=axL.transAxes)
        axR.text(0.5, 0.5, "frame2 fail", ha="center", va="center",
                 transform=axR.transAxes)
    axL.axis("off")
    axR.axis("off")
    axL.set_title(f"frame 1: {Path(ex['video_path_1']).name} @ {ex['frame_idx_1']}",
                  fontsize=8, color="#555555")
    axR.set_title(f"frame 2: {Path(ex['video_path_2']).name} @ {ex['frame_idx_2']}",
                  fontsize=8, color="#555555")
    cap = (
        f"GT={int(ex['correct_answer']):+d}  bucket={ex['bucket']}  "
        f"demo={ex['demo_id_exact']}"
    )
    axL.text(0.0, 1.15, cap, transform=axL.transAxes, fontsize=10,
             fontweight="bold", color="#2c3e50", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    task_token = TASK_TOKENS[args.task_name]
    idxs = parse_indices(args.indices)
    intervals = [int(x) for x in args.compare_interval.split(",")]

    successes = load_upright_bottle_demos(args.dataset_root, idxs, "success")
    failures = load_upright_bottle_demos(args.dataset_root, idxs, "failure")

    pairs = build_pairs_for_demos(
        successes, failures, intervals, args.sample_interval, task_token,
        args.failure_last_frac, args.failure_min_frames,
        rng=random.Random(args.seed),
    )
    print(f"Built {len(pairs)} total pairs from "
          f"{len(successes)} success / {len(failures)} failure demos")

    keep = set()
    if args.buckets == "all":
        keep = None
    else:
        keep = set(args.buckets.split(","))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for k, ex in enumerate(pairs):
        bucket = ex["bucket"]
        bucket_class = "fail_vs_succ" if bucket.startswith("fail_vs_succ") else "succ_vs_succ"
        if keep is not None and bucket_class not in keep and bucket not in keep:
            continue
        sub = out_dir / bucket
        sub.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{k:04d}_{ex['demo_id_exact']}_"
            f"f1={ex['frame_idx_1']}_f2={ex['frame_idx_2']}.png"
        )
        render(ex, sub / fname)
        counts[bucket] = counts.get(bucket, 0) + 1
    print(f"Saved PNGs under {out_dir}")
    for b in sorted(counts):
        print(f"  {b}: {counts[b]}")


if __name__ == "__main__":
    main()
