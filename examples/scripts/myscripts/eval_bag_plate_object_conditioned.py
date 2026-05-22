"""
Held-out eval for the BagPlate object-conditioned ranking VLM checkpoint.

Reads `matches.csv`, keeps only rows where EITHER demo is in the held-out
index set, then for each kept (sub-frame, camera) emits two examples - one
per target object in the pair. Runs model.generate, parses the signed integer
answer, and reports sign accuracy overall + per (pair_type x target x camera).

Usage:
CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/eval_bag_plate_object_conditioned.py \
--checkpoint /proj/vondrick3/sruthi/Appaji/trl/outputs/BagPlateObjCond_20260516_151157/checkpoint-1200 \
--eval_box_indices 18-20 \
--eval_glass_indices 18-20 \
--eval_remote_indices 18-20 \
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
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_bag_plate_object_conditioned import (  # noqa: E402
    OBJECT_KINDS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_pairs_from_csv,
    load_bag_plate_demos,
    parse_indices,
    read_matches_csv,
)
from video_frame_utils import extract_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIGNED_INT_RE = re.compile(r"-?\d+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="/proj/vondrick3/datasets/expert_data_jgd_bag_plate")
    p.add_argument(
        "--matches_csv",
        default="/proj/vondrick3/datasets/expert_data_jgd_bag_plate/similar_pairs_out_grb/matches.csv",
    )
    p.add_argument("--eval_box_indices", default="18-20")
    p.add_argument("--eval_glass_indices", default="18-20")
    p.add_argument("--eval_remote_indices", default="18-20")
    p.add_argument("--pair_sample_step", type=int, default=2,
                   help="step through last-frac sub-frames when building eval pairs")
    p.add_argument("--failure_last_frac", type=float, default=0.95)
    p.add_argument("--failure_min_frames", type=int, default=8)
    p.add_argument("--max_pixels", default="960x540",
                   help="WxH, matches training max_pixels")
    p.add_argument("--max_eval_pairs", type=int, default=600,
                   help="cap on number of eval pairs to evaluate (0 = no cap)")
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
    ax_left.set_title("frame 1", fontsize=8, color="#555555")
    ax_right.set_title("frame 2", fontsize=8, color="#555555")
    cap = (
        f"[{verdict}] target={ex.get('target_object','?')}  GT={target_ans:+d}  Pred={pred_str}  "
        f"| {ex.get('bucket','?')}  | {ex.get('demo_id_exact','?')}"
    )
    ax_left.text(
        0.0, 1.15, cap, transform=ax_left.transAxes, fontsize=9, color=color,
        fontweight="bold", ha="left", va="bottom",
    )


def save_per_example_pngs(records, out_dir, n):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(records[:n]):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
        _render_example_panel(axL, axR, r["ex"], r["gen"], r["pred_ans"], r["correct"], r["target"])
        verdict_tag = "ok" if r["correct"] else ("unparsed" if r["pred_ans"] is None else "wrong")
        fname = (
            f"{k:03d}_{verdict_tag}_{r['ex'].get('demo_id_exact','demo')}"
            f"_target-{r['ex'].get('target_object','?')}_{r['ex'].get('camera','?')}.png"
        )
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)


def save_grid(records, out_path, title, max_cells):
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
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
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

    eval_sets = {
        "box": set(parse_indices(args.eval_box_indices)),
        "glass": set(parse_indices(args.eval_glass_indices)),
        "remote": set(parse_indices(args.eval_remote_indices)),
    }

    # Load only the demos referenced by the CSV (the training script does the same).
    needed = {k: set() for k in OBJECT_KINDS}
    for kind_a, idx_a, kind_b, idx_b, _rank, _iou in read_matches_csv(args.matches_csv):
        needed[kind_a].add(idx_a)
        needed[kind_b].add(idx_b)
    logger.info("CSV references: " + ", ".join(f"{len(needed[k])} {k}" for k in OBJECT_KINDS))

    loaded = {
        k: load_bag_plate_demos(args.dataset_root, sorted(needed[k]), k)
        for k in OBJECT_KINDS
    }
    demos_by_key = {}
    for k, lst in loaded.items():
        for d in lst:
            demos_by_key[(d["kind"], d["idx"])] = d

    _train_pairs, eval_pairs, route_stats = build_pairs_from_csv(
        args.matches_csv,
        demos_by_key,
        eval_sets,
        args.failure_last_frac,
        args.failure_min_frames,
        args.pair_sample_step,
        rng=random.Random(args.seed),
    )
    logger.info(
        f"Row routing: train={route_stats['n_train_rows']}, eval={route_stats['n_eval_rows']}, "
        f"missing_demo={route_stats['n_missing_demo']}"
    )
    logger.info(f"Built {len(eval_pairs)} eval pairs (train pairs discarded)")

    rng = random.Random(args.seed)
    rng.shuffle(eval_pairs)
    if args.max_eval_pairs and len(eval_pairs) > args.max_eval_pairs:
        eval_pairs = eval_pairs[:args.max_eval_pairs]
    logger.info(
        f"Evaluating {len(eval_pairs)} pairs (cap={args.max_eval_pairs or 'none'})"
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

    for i, ex in enumerate(eval_pairs):
        try:
            bucket = ex.get("bucket", "unknown")
            target_ans = int(ex["correct_answer"])
            target_object = ex.get("target_object", "unknown")
            user_text = USER_PROMPT_TEMPLATE.format(target_object=target_object)

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
                    "target_object": target_object,
                    "demo": ex.get("demo_id_exact", "?"),
                })

            if (i + 1) % 20 == 0:
                running = sum(b["correct"] for b in buckets.values()) / max(sum(b["total"] for b in buckets.values()), 1)
                logger.info(f"[{i+1}/{len(eval_pairs)}] running sign_acc={running:.3f}")
        except Exception as e:
            logger.warning(f"example {i} failed: {e}")

    total = sum(b["total"] for b in buckets.values())
    correct = sum(b["correct"] for b in buckets.values())
    unparsed = sum(b["unparsed"] for b in buckets.values())

    def _agg(predicate):
        cs = [b for k, b in buckets.items() if predicate(k)]
        tot = sum(b["total"] for b in cs)
        cor = sum(b["correct"] for b in cs)
        unp = sum(b["unparsed"] for b in cs)
        return cor / max(tot, 1), unp / max(tot, 1), tot

    cam0_acc, cam0_unp, cam0_n = _agg(lambda k: k.endswith("_cam0"))
    cam1_acc, cam1_unp, cam1_n = _agg(lambda k: k.endswith("_cam1"))
    per_target = {
        t: _agg(lambda k, t=t: f"target-{t}_" in k) for t in OBJECT_KINDS
    }
    per_pair_type = {
        pt: _agg(lambda k, pt=pt: k.startswith(pt + "_"))
        for pt in ("box-glass", "box-remote", "glass-remote")
    }

    summary = {
        "checkpoint": args.checkpoint,
        "matches_csv": args.matches_csv,
        "eval_box_indices": args.eval_box_indices,
        "eval_glass_indices": args.eval_glass_indices,
        "eval_remote_indices": args.eval_remote_indices,
        "n_evaluated": total,
        "sign_acc_overall": correct / max(total, 1),
        "unparsed_overall": unparsed / max(total, 1),
        "sign_acc_cam0": cam0_acc, "unparsed_cam0": cam0_unp, "n_cam0": cam0_n,
        "sign_acc_cam1": cam1_acc, "unparsed_cam1": cam1_unp, "n_cam1": cam1_n,
        "per_target": {
            t: {"sign_acc": acc, "unparsed": unp, "n": n_t}
            for t, (acc, unp, n_t) in per_target.items()
        },
        "per_pair_type": {
            pt: {"sign_acc": acc, "unparsed": unp, "n": n_pt}
            for pt, (acc, unp, n_pt) in per_pair_type.items()
        },
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
    logger.info(f"  sign_acc_overall:    {summary['sign_acc_overall']:.4f}")
    logger.info(f"  unparsed_overall:    {summary['unparsed_overall']:.4f}")
    logger.info(f"  sign_acc_cam0:       {cam0_acc:.4f}  (n={cam0_n})")
    logger.info(f"  sign_acc_cam1:       {cam1_acc:.4f}  (n={cam1_n})")
    for t, (acc, _unp, n_t) in per_target.items():
        logger.info(f"  sign_acc_target_{t:<6}: {acc:.4f}  (n={n_t})")
    for pt, (acc, _unp, n_pt) in per_pair_type.items():
        logger.info(f"  sign_acc_pair_{pt:<14}: {acc:.4f}  (n={n_pt})")
    logger.info("  per-bucket:")
    for k, v in summary["per_bucket"].items():
        logger.info(f"    {k:40s}  acc={v['sign_acc']:.4f}  unparsed={v['unparsed']:.4f}  n={v['n']}")
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

    # Per-pair-type folders of all per-example PNGs, grouped by verdict.
    by_pair_type_dir = viz_dir / "by_pair_type"
    for r in records:
        pt = r["ex"].get("pair_type", "unknown")
        if r["pred_ans"] is None:
            verdict_tag = "unparsed"
        else:
            verdict_tag = "ok" if r["correct"] else "wrong"
        out_d = by_pair_type_dir / pt / verdict_tag
        out_d.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(records):
        pt = r["ex"].get("pair_type", "unknown")
        if r["pred_ans"] is None:
            verdict_tag = "unparsed"
        else:
            verdict_tag = "ok" if r["correct"] else "wrong"
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
        _render_example_panel(axL, axR, r["ex"], r["gen"], r["pred_ans"], r["correct"], r["target"])
        fname = (
            f"{k:04d}_{r['ex'].get('demo_id_exact','demo')}"
            f"_target-{r['ex'].get('target_object','?')}_{r['ex'].get('camera','?')}.png"
        )
        fig.tight_layout()
        fig.savefig(by_pair_type_dir / pt / verdict_tag / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
    logger.info(f"Saved {len(records)} per-pair PNGs under {by_pair_type_dir}")

    with open(viz_dir / "predictions.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps({
                "demo": r["ex"].get("demo_id_exact", "?"),
                "bucket": r["bucket"],
                "pair_type": r["ex"].get("pair_type", "?"),
                "target_object": r["ex"].get("target_object", "?"),
                "camera": r["ex"].get("camera", "?"),
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
