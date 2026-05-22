"""
Held-out eval for Stacking SFT checkpoint.

Builds success-vs-success and (optionally) failure-vs-success pairs from a
held-out set of demo indices (default 48-50, which were held out from training
in launch_stacking.sh), runs model.generate, parses the signed integer answer,
and reports sign accuracy overall + per bucket (interval x camera).

Usage:
CUDA_VISIBLE_DEVICES=1 python examples/scripts/myscripts/eval_stacking.py \
--checkpoint /proj/vondrick3/sruthi/Appaji/trl/outputs/Stacking_20260520_150251/checkpoint-800 \
--success_indices 48-50 \
--failure_indices 48-50 \
--compare_interval 3,4,8,12,16 \
--max_succ_pairs 500 --max_fail_pairs 500 \
--max_pixels 960x540 \
--failure_last_frac 0.95

CUDA_VISIBLE_DEVICES=2 python examples/scripts/myscripts/eval_stacking.py \
--checkpoint /proj/vondrick3/sruthi/Appaji/trl/outputs/Stacking_20260520_155659_848x480/checkpoint-750 \
--success_indices 48-50 \
--failure_indices 48-50 \
--compare_interval 3,4,8,12,16 \
--max_succ_pairs 500 --max_fail_pairs 500 \
--max_pixels 848x480 \
--failure_last_frac 0.95
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_stacking import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_pairs_for_demos,
    load_stacking_demos,
    parse_indices,
)
from sft_vlm_overlay_regression_v2 import TASK_TOKENS  # noqa: E402
from video_frame_utils import extract_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIGNED_INT_RE = re.compile(r"-?\d+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="/proj/vondrick3/datasets/expert_data_jgd_stacking")
    p.add_argument("--task_name", default="Stacking")
    p.add_argument("--success_indices", default="48-50")
    p.add_argument("--failure_indices", default="48-50",
                   help="comma/range list of failure indices for held-out eval")
    p.add_argument("--compare_interval", default="3,4,8,12,16")
    p.add_argument("--sample_interval", type=int, default=2,
                   help="step through subsampled frames when building pairs (matches train_sample_interval)")
    p.add_argument("--failure_last_frac", type=float, default=0.95)
    p.add_argument("--failure_min_frames", type=int, default=8)
    p.add_argument("--max_pixels", default="848x480",
                   help="WxH, matches training max_pixels")
    p.add_argument("--max_succ_pairs", type=int, default=400,
                   help="cap on number of succ_vs_succ pairs to evaluate (0 = no cap)")
    p.add_argument("--max_fail_pairs", type=int, default=400,
                   help="cap on number of fail_vs_succ pairs to evaluate (0 = no cap)")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_json", default=None,
                   help="optional path to dump per-bucket metrics + qualitative examples")
    p.add_argument("--n_qualitative", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--viz_dir", default=None,
                   help="directory to save eval visualizations (default: <checkpoint>/heldout_eval_viz)")
    p.add_argument("--viz_n_per_example_pngs", type=int, default=40,
                   help="number of per-example PNGs to save (one PNG per pair)")
    p.add_argument("--viz_grid_n", type=int, default=24,
                   help="number of cells per correct/incorrect grid figure")
    return p.parse_args()


def _render_example_panel(ax_left, ax_right, ex, gen_text, pred_ans, correct, target_ans):
    """Draw two frames into the given matplotlib axes with a colored caption."""
    try:
        f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
        f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])
        ax_left.imshow(f1)
        ax_right.imshow(f2)
    except Exception as e:
        ax_left.text(0.5, 0.5, f"frame1 fail\n{e}", ha="center", va="center", transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, "frame2 fail", ha="center", va="center", transform=ax_right.transAxes)
    ax_left.axis("off")
    ax_right.axis("off")

    if pred_ans is None:
        color = "#7f8c8d"
        verdict = "UNPARSED"
    else:
        color = "#2ecc71" if correct else "#e74c3c"
        verdict = "OK" if correct else "WRONG"
    pred_str = f"{pred_ans}" if pred_ans is not None else f"<unparsed: {gen_text!r}>"
    ax_left.set_title("frame 1 (earlier label)", fontsize=8, color="#555555")
    ax_right.set_title("frame 2 (later label)", fontsize=8, color="#555555")
    cap = (
        f"[{verdict}] GT={target_ans:+d}  Pred={pred_str}  "
        f"| {ex.get('bucket','?')}  | {ex.get('demo_id_exact','?')}"
    )
    ax_left.text(
        0.0, 1.15, cap, transform=ax_left.transAxes, fontsize=9, color=color,
        fontweight="bold", ha="left", va="bottom",
    )


def save_per_example_pngs(records, out_dir, n):
    """Write up to n individual PNGs (one per pair) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(records[:n]):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
        _render_example_panel(axL, axR, r["ex"], r["gen"], r["pred_ans"], r["correct"], r["target"])
        verdict_tag = "ok" if r["correct"] else ("unparsed" if r["pred_ans"] is None else "wrong")
        fname = f"{k:03d}_{verdict_tag}_{r['ex'].get('demo_id_exact','demo')}_{r['ex'].get('bucket','bucket')}.png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)


def save_grid(records, out_path, title, max_cells):
    """Save a grid of example panels (2 axes per example: frame1, frame2)."""
    if not records:
        return
    n = min(len(records), max_cells)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(5 * n_cols * 2, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_rows * n_cols):
        r_idx, c_idx = i // n_cols, (i % n_cols) * 2
        if i >= n:
            axes[r_idx, c_idx].axis("off")
            axes[r_idx, c_idx + 1].axis("off")
            continue
        r = records[i]
        _render_example_panel(axes[r_idx, c_idx], axes[r_idx, c_idx + 1],
                              r["ex"], r["gen"], r["pred_ans"], r["correct"], r["target"])
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved grid -> {out_path}")


def save_bucket_chart(buckets, out_path, overall_acc):
    if not buckets:
        return
    keys = sorted(buckets.keys())
    accs = [buckets[k]["correct"] / max(buckets[k]["total"], 1) for k in keys]
    ns = [buckets[k]["total"] for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(keys) + 4), 5))
    bars = ax.bar(range(len(keys)), accs, color="#3498db")
    ax.axhline(overall_acc, color="#e67e22", linestyle="--", label=f"overall={overall_acc:.3f}")
    ax.axhline(0.5, color="#95a5a6", linestyle=":", label="chance=0.5")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("sign accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sign accuracy per bucket")
    for i, (b, n) in enumerate(zip(bars, ns)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bucket chart -> {out_path}")


def main():
    args = parse_args()

    if args.task_name not in TASK_TOKENS:
        raise ValueError(f"Unknown task '{args.task_name}'. Add it to TASK_TOKENS.")
    task_token = TASK_TOKENS[args.task_name]

    succ_idxs = parse_indices(args.success_indices)
    fail_idxs = parse_indices(args.failure_indices) if args.failure_indices else []
    intervals = [int(x) for x in args.compare_interval.split(",")]

    successes = load_stacking_demos(args.dataset_root, succ_idxs, "success")
    failures = load_stacking_demos(args.dataset_root, fail_idxs, "failure") if fail_idxs else []

    pairs = build_pairs_for_demos(
        successes, failures, intervals, args.sample_interval, task_token,
        args.failure_last_frac, args.failure_min_frames,
        rng=random.Random(args.seed),
    )
    logger.info(f"Built {len(pairs)} pairs from {len(successes)} success / {len(failures)} failure demos")
    logger.info(f"Bucket counts: {Counter(p['bucket'] for p in pairs)}")

    rng = random.Random(args.seed)
    succ_pairs = [p for p in pairs if not p["bucket"].startswith("fail_vs_succ")]
    fail_pairs = [p for p in pairs if p["bucket"].startswith("fail_vs_succ")]
    rng.shuffle(succ_pairs)
    rng.shuffle(fail_pairs)
    if args.max_succ_pairs and len(succ_pairs) > args.max_succ_pairs:
        succ_pairs = succ_pairs[:args.max_succ_pairs]
    if args.max_fail_pairs and len(fail_pairs) > args.max_fail_pairs:
        fail_pairs = fail_pairs[:args.max_fail_pairs]
    pairs = succ_pairs + fail_pairs
    rng.shuffle(pairs)
    logger.info(
        f"Evaluating {len(pairs)} pairs "
        f"(succ_vs_succ={len(succ_pairs)} cap={args.max_succ_pairs or 'none'}, "
        f"fail_vs_succ={len(fail_pairs)} cap={args.max_fail_pairs or 'none'})"
    )

    dtype = getattr(torch, args.dtype)
    logger.info(f"Loading model from {args.checkpoint} (dtype={args.dtype})")
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    mp_w, mp_h = (int(x) for x in args.max_pixels.lower().split("x"))
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        max_pixels=mp_w * mp_h,
    )

    buckets = {}
    qualitative = []
    records = []

    for i, ex in enumerate(pairs):
        try:
            bucket = ex.get("bucket", "unknown")
            target_ans = int(ex["correct_answer"])
            user_text = USER_PROMPT_TEMPLATE.format(task_token=ex.get("task_token", task_token))

            f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
            f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])

            mm_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": f1},
                    {"type": "image", "image": f2},
                    {"type": "text", "text": user_text},
                ]},
            ]
            text = processor.apply_chat_template(
                mm_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[[f1, f2]],
                return_tensors="pt", padding=True,
            )
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()

            m = SIGNED_INT_RE.search(gen)
            pred_ans = int(m.group(0)) if m else None

            stats = buckets.setdefault(bucket, {"correct": 0, "total": 0, "unparsed": 0})
            stats["total"] += 1
            is_correct = False
            if pred_ans is None:
                stats["unparsed"] += 1
            elif (pred_ans > 0) == (target_ans > 0):
                stats["correct"] += 1
                is_correct = True

            records.append({
                "ex": ex,
                "gen": gen,
                "pred_ans": pred_ans,
                "target": target_ans,
                "correct": is_correct,
                "bucket": bucket,
            })

            if len(qualitative) < args.n_qualitative:
                qualitative.append({
                    "target": target_ans,
                    "pred_text": gen,
                    "pred_ans": pred_ans,
                    "bucket": bucket,
                    "demo": ex.get("demo_id_exact", "?"),
                })

            if (i + 1) % 20 == 0:
                running = sum(b["correct"] for b in buckets.values()) / max(sum(b["total"] for b in buckets.values()), 1)
                logger.info(f"[{i+1}/{len(pairs)}] running sign_acc={running:.3f}")
        except Exception as e:
            logger.warning(f"example {i} failed: {e}")

    total = sum(b["total"] for b in buckets.values())
    correct = sum(b["correct"] for b in buckets.values())
    unparsed = sum(b["unparsed"] for b in buckets.values())

    def _agg(suffix):
        cs = [b for k, b in buckets.items() if k.endswith(suffix)]
        tot = sum(b["total"] for b in cs)
        cor = sum(b["correct"] for b in cs)
        unp = sum(b["unparsed"] for b in cs)
        return cor / max(tot, 1), unp / max(tot, 1), tot

    cam0_acc, cam0_unp, cam0_n = _agg("_cam0")
    cam1_acc, cam1_unp, cam1_n = _agg("_cam1")

    summary = {
        "checkpoint": args.checkpoint,
        "success_indices": args.success_indices,
        "failure_indices": args.failure_indices,
        "n_evaluated": total,
        "sign_acc_overall": correct / max(total, 1),
        "unparsed_overall": unparsed / max(total, 1),
        "sign_acc_cam0": cam0_acc, "unparsed_cam0": cam0_unp, "n_cam0": cam0_n,
        "sign_acc_cam1": cam1_acc, "unparsed_cam1": cam1_unp, "n_cam1": cam1_n,
        "per_bucket": {
            k: {
                "sign_acc": v["correct"] / max(v["total"], 1),
                "unparsed": v["unparsed"] / max(v["total"], 1),
                "n": v["total"],
            }
            for k, v in sorted(buckets.items())
        },
        "qualitative": qualitative,
    }

    logger.info("=" * 60)
    logger.info(f"Held-out eval results (n={total})")
    logger.info(f"  sign_acc_overall: {summary['sign_acc_overall']:.4f}")
    logger.info(f"  unparsed_overall: {summary['unparsed_overall']:.4f}")
    logger.info(f"  sign_acc_cam0:    {cam0_acc:.4f}  (n={cam0_n})")
    logger.info(f"  sign_acc_cam1:    {cam1_acc:.4f}  (n={cam1_n})")
    logger.info("  per-bucket:")
    for k, v in summary["per_bucket"].items():
        logger.info(f"    {k:32s}  acc={v['sign_acc']:.4f}  unparsed={v['unparsed']:.4f}  n={v['n']}")
    logger.info("=" * 60)

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info(f"Wrote summary to {args.out_json}")

    viz_dir = Path(args.viz_dir) if args.viz_dir else Path(args.checkpoint) / "heldout_eval_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing visualizations to {viz_dir}")

    correct_recs = [r for r in records if r["correct"]]
    wrong_recs = [r for r in records if (not r["correct"]) and r["pred_ans"] is not None]
    unparsed_recs = [r for r in records if r["pred_ans"] is None]

    rng2 = random.Random(args.seed + 1)
    by_key = defaultdict(list)
    for r in records:
        verdict = "ok" if r["correct"] else ("unparsed" if r["pred_ans"] is None else "wrong")
        by_key[(r["bucket"], verdict)].append(r)
    mixed = []
    keys = list(by_key.keys())
    rng2.shuffle(keys)
    while len(mixed) < args.viz_n_per_example_pngs and any(by_key[k] for k in keys):
        for k in keys:
            if by_key[k] and len(mixed) < args.viz_n_per_example_pngs:
                mixed.append(by_key[k].pop(rng2.randrange(len(by_key[k]))))

    save_per_example_pngs(mixed, viz_dir / "per_example", args.viz_n_per_example_pngs)
    save_grid(
        rng2.sample(correct_recs, min(args.viz_grid_n, len(correct_recs))) if correct_recs else [],
        viz_dir / "grid_correct.png",
        f"Correct predictions (sample of {min(args.viz_grid_n, len(correct_recs))} / {len(correct_recs)})",
        args.viz_grid_n,
    )
    save_grid(
        rng2.sample(wrong_recs, min(args.viz_grid_n, len(wrong_recs))) if wrong_recs else [],
        viz_dir / "grid_wrong.png",
        f"Wrong predictions (sample of {min(args.viz_grid_n, len(wrong_recs))} / {len(wrong_recs)})",
        args.viz_grid_n,
    )
    if unparsed_recs:
        save_grid(
            rng2.sample(unparsed_recs, min(args.viz_grid_n, len(unparsed_recs))),
            viz_dir / "grid_unparsed.png",
            f"Unparsed predictions ({len(unparsed_recs)} total)",
            args.viz_grid_n,
        )

    save_bucket_chart(buckets, viz_dir / "bucket_accuracy.png", summary["sign_acc_overall"])

    fvs_recs = [r for r in records if r["bucket"].startswith("fail_vs_succ")]
    fvs_dir = viz_dir / "fail_vs_succ"
    for verdict in ("ok", "wrong", "unparsed"):
        (fvs_dir / verdict).mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(fvs_recs):
        if r["pred_ans"] is None:
            verdict_tag = "unparsed"
        else:
            verdict_tag = "ok" if r["correct"] else "wrong"
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
        _render_example_panel(axL, axR, r["ex"], r["gen"], r["pred_ans"], r["correct"], r["target"])
        fname = f"{k:04d}_{r['ex'].get('demo_id_exact','demo')}_{r['ex'].get('bucket','bucket')}.png"
        fig.tight_layout()
        fig.savefig(fvs_dir / verdict_tag / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
    logger.info(f"Saved {len(fvs_recs)} fail_vs_succ per-pair PNGs under {fvs_dir}")

    with open(viz_dir / "predictions.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps({
                "demo": r["ex"].get("demo_id_exact", "?"),
                "bucket": r["bucket"],
                "video_path_1": r["ex"]["video_path_1"],
                "frame_idx_1": r["ex"]["frame_idx_1"],
                "video_path_2": r["ex"]["video_path_2"],
                "frame_idx_2": r["ex"]["frame_idx_2"],
                "target": r["target"],
                "pred_ans": r["pred_ans"],
                "gen": r["gen"],
                "correct": r["correct"],
            }) + "\n")
    logger.info(f"Wrote predictions.jsonl ({len(records)} rows)")


if __name__ == "__main__":
    main()
