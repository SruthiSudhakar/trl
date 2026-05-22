"""
Trainer for BagPlate pairwise progress comparison (two camera views), with
demo pairs driven by an external matches CSV instead of same-index matching.

Layout assumed at --dataset_root:
    <kind>_<N>_<timestamp>.npy            # dict with image_paths_cam0 / image_paths_cam1
    <kind>_<N>_<timestamp>_frames_cam0/   # frame_000000.jpg ...
    <kind>_<N>_<timestamp>_frames_cam1/   # frame_000000.jpg ...
where kind in {"box","glass","remote"}.

Semantics: `box` is the success category. `glass` and `remote` are both
failure categories. Each row of --matches_csv has columns
    rank,pair_type,iou,demo_a,demo_b
with pair_type in {"box-glass","box-remote"}, demo_a a box demo (e.g.
"box_8") and demo_b a failure demo (e.g. "glass_9" or "remote_4").

Pairing: for each CSV row, pair the LAST K sub-frames of the failure demo
(where the failure has clearly happened) against the temporally aligned
sub-frames of the matched box demo. K = max(failure_min_frames,
round(failure_last_frac * n_sub)). For every logical pair, we emit TWO
examples - one using cam0 frames and one using cam1 frames - so train/eval
are balanced 50/50 across camera views.

Train/eval split is by demo holdout: a row is routed to eval only if BOTH
its box demo AND its failure demo are in the corresponding eval index sets;
to train only if NEITHER is; rows that mix train and eval demos are dropped.
"""

import csv
import gc
import glob
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

# Sibling import works regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_overlay_regression_v2 import TASK_TOKENS, visualize_dataset  # noqa: E402
from video_frame_utils import extract_frame  # noqa: E402

random.seed(42)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUBSAMPLE = 2  # 30 Hz -> 10 Hz

# Prompts: two separate images, "first" vs "second".
SYSTEM_PROMPT = (
    "Compare robot task progress. You will see two images. "
    "Respond with a number: positive if the second image shows more progress, "
    "negative if the first does."
)
USER_PROMPT_TEMPLATE = (
    "Task: {task_token}\n"
    "Which image shows more task progress (the first or the second)? "
    "Respond with a number from -100 to 100."
)

ANSWER_MAGNITUDE = 32  # binary +/- this magnitude

DEMO_NAME_RE = re.compile(r"^(box|glass|remote)_(\d+)$")


def parse_indices(s: str) -> list:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_bag_plate_demos(root: str, indices: list, kind: str) -> list:
    demos = []
    for n in indices:
        matches = sorted(glob.glob(f"{root}/{kind}_{n}_*.npy"))
        if not matches:
            logger.warning(f"No {kind}_{n}_* found under {root}, skipping")
            continue
        path = matches[0]
        meta = np.load(path, allow_pickle=True).item()
        frames_dir_cam0 = path[:-4] + "_frames_cam0"
        frames_dir_cam1 = path[:-4] + "_frames_cam1"
        if not os.path.isdir(frames_dir_cam0):
            logger.warning(f"Frames dir missing: {frames_dir_cam0}, skipping")
            continue
        if not os.path.isdir(frames_dir_cam1):
            logger.warning(f"Frames dir missing: {frames_dir_cam1}, skipping")
            continue
        n0 = len(meta["image_paths_cam0"])
        n1 = len(meta["image_paths_cam1"])
        if n0 != n1:
            logger.warning(
                f"cam0/cam1 frame counts differ for {path}: {n0} vs {n1}; "
                f"using min({n0},{n1})"
            )
        demos.append({
            "video_path_cam0": frames_dir_cam0,
            "video_path_cam1": frames_dir_cam1,
            "n_frames": min(n0, n1),
            "idx": n,
            "kind": kind,
        })
    logger.info(f"Loaded {len(demos)} {kind} demos")
    return demos


def read_matches_csv(csv_path: str):
    """Yield (kind, box_idx, fail_idx, rank, iou) tuples from matches.csv.

    `kind` is "glass" or "remote" (the failure category in this row).
    """
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pair_type = row["pair_type"].strip()
            if pair_type not in ("box-glass", "box-remote"):
                logger.warning(f"Unknown pair_type {pair_type!r}, skipping row {row}")
                continue
            fail_kind = pair_type.split("-")[1]
            ma = DEMO_NAME_RE.match(row["demo_a"].strip())
            mb = DEMO_NAME_RE.match(row["demo_b"].strip())
            if not ma or not mb or ma.group(1) != "box" or mb.group(1) != fail_kind:
                logger.warning(f"Malformed row, skipping: {row}")
                continue
            box_idx = int(ma.group(2))
            fail_idx = int(mb.group(2))
            yield fail_kind, box_idx, fail_idx, int(row["rank"]), float(row["iou"])


def build_pairs_from_csv(
    csv_path,
    demos_by_key,
    eval_sets,
    task_token,
    failure_last_frac,
    failure_min_frames,
    pair_sample_step,
    rng=None,
):
    """Read matches.csv and emit (failure, success) pairs for each row.

    demos_by_key: dict keyed by (kind, idx) -> demo dict from load_bag_plate_demos.
    eval_sets: dict {"box": set[int], "glass": set[int], "remote": set[int]}.

    Returns (train_pairs, eval_pairs, stats). A row goes to eval if EITHER its
    box demo OR its failure demo is in the corresponding eval index set; this
    guarantees the held-out demos never appear in any training pair.
    """
    rng = rng or random.Random(42)
    user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)
    train_pairs, eval_pairs = [], []
    CAMERAS = ("cam0", "cam1")
    stats = {"n_train_rows": 0, "n_eval_rows": 0, "n_missing_demo": 0}

    def emit(out_list, v1, f1, v2, f2, ans, demo_kind, demo_idx, bucket, camera):
        out_list.append({
            "video_path_1": v1, "frame_idx_1": f1,
            "video_path_2": v2, "frame_idx_2": f2,
            "correct_answer": ans,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "completion": [
                {"role": "assistant", "content": str(ans)},
            ],
            "demo_id": demo_idx,
            "demo_id_exact": f"{demo_kind}_{demo_idx}",
            "demo_success": "failure",
            "bucket": bucket,
            "camera": camera,
            "job_name": "BagPlateMatched",
            "task_token": task_token,
        })

    for fail_kind, box_idx, fail_idx, rank, iou in read_matches_csv(csv_path):
        box_demo = demos_by_key.get(("box", box_idx))
        fail_demo = demos_by_key.get((fail_kind, fail_idx))
        if box_demo is None or fail_demo is None:
            stats["n_missing_demo"] += 1
            logger.warning(
                f"Row rank={rank}: missing demo on disk "
                f"(box_{box_idx}={'OK' if box_demo else 'MISSING'}, "
                f"{fail_kind}_{fail_idx}={'OK' if fail_demo else 'MISSING'}); skipping"
            )
            continue

        box_eval = box_idx in eval_sets["box"]
        fail_eval = fail_idx in eval_sets[fail_kind]
        if box_eval or fail_eval:
            out_list = eval_pairs
            stats["n_eval_rows"] += 1
        else:
            out_list = train_pairs
            stats["n_train_rows"] += 1

        f_sub_max = (fail_demo["n_frames"] - 1) // SUBSAMPLE
        s_sub_max = (box_demo["n_frames"] - 1) // SUBSAMPLE
        n_sub = f_sub_max + 1
        if n_sub < 1:
            continue
        k = max(failure_min_frames, int(round(failure_last_frac * n_sub)))
        k = min(k, n_sub)
        last_indices = list(range(n_sub - k, n_sub))
        offset = rng.randint(0, pair_sample_step - 1) if pair_sample_step > 1 else 0
        kept = last_indices[offset::pair_sample_step] if pair_sample_step > 1 else last_indices

        for i in kept:
            f_src = i * SUBSAMPLE
            s_src = min(i, s_sub_max) * SUBSAMPLE
            for cam in CAMERAS:
                f_path = fail_demo[f"video_path_{cam}"]
                s_path = box_demo[f"video_path_{cam}"]
                ans = ANSWER_MAGNITUDE
                v1, vf1, v2, vf2 = f_path, f_src, s_path, s_src
                if rng.random() < 0.5:
                    v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                    ans = -ans
                emit(
                    out_list, v1, vf1, v2, vf2, ans,
                    fail_kind, fail_idx,
                    f"matched_{fail_kind}_{cam}", cam,
                )

        logger.info(
            f"Row rank={rank} ({fail_kind}): box_{box_idx} vs {fail_kind}_{fail_idx}, "
            f"n_sub={n_sub}, k={k}, kept={len(kept)} sub-frames x {len(CAMERAS)} cams "
            f"-> {'eval' if out_list is eval_pairs else 'train'}"
        )

    return train_pairs, eval_pairs, stats


@dataclass
class TwoImageCollator(DataCollatorForVisionLanguageModeling):
    """DataCollatorForVisionLanguageModeling that resolves two video frames per
    example and provides them as a list of two PIL images.
    """

    def _collate_language_modeling(self, examples):
        for example in examples:
            f1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            f2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            example["images"] = [f1, f2]
        return super()._collate_language_modeling(examples)

    def _collate_prompt_completion(self, examples):
        for example in examples:
            f1 = extract_frame(example["video_path_1"], example["frame_idx_1"])
            f2 = extract_frame(example["video_path_2"], example["frame_idx_2"])
            example["images"] = [f1, f2]
        output = super()._collate_prompt_completion(examples)
        if "mm_token_type_ids" in output:
            mm = output["mm_token_type_ids"]
            seq_len = output["input_ids"].shape[1]
            if mm.shape[1] < seq_len:
                pad = torch.zeros(
                    (mm.shape[0], seq_len - mm.shape[1]),
                    dtype=mm.dtype, device=mm.device,
                )
                output["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
            elif mm.shape[1] > seq_len:
                output["mm_token_type_ids"] = mm[:, :seq_len]
        return output


class PairwiseSignAccuracyCallback(TrainerCallback):
    """Run model.generate on a sample of the eval dataset, parse the signed
    integer in each response, and log per-bucket sign accuracy + a few
    qualitative examples. Distributed-aware.
    """

    SIGNED_INT_RE = re.compile(r"-?\d+")

    def __init__(self, eval_dataset, processor, max_pairs=200, n_qualitative=3):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.max_pairs = max_pairs
        self.n_qualitative = n_qualitative

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_dataset is None or len(self.eval_dataset) == 0 or self.processor is None:
            return
        model = kwargs.get("model")
        if model is None:
            return

        try:
            import torch.distributed as dist
            world = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world, rank = 1, 0

        rng = random.Random(1234)
        all_indices = list(range(len(self.eval_dataset)))
        rng.shuffle(all_indices)
        n = min(self.max_pairs, len(all_indices))
        all_indices = all_indices[:n]

        my_indices = all_indices[rank::world]
        buckets = {}
        qualitative = []

        was_training = model.training
        model.eval()
        try:
            for i in my_indices:
                try:
                    ex = self.eval_dataset[i]
                    bucket = ex.get("bucket", "unknown")
                    target_ans = int(ex["correct_answer"])
                    task_token = ex.get("task_token", "[UNKNOWN]")
                    user_text = USER_PROMPT_TEMPLATE.format(task_token=task_token)

                    f1 = extract_frame(ex["video_path_1"], ex["frame_idx_1"])
                    f2 = extract_frame(ex["video_path_2"], ex["frame_idx_2"])

                    mm_messages = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f1},
                                {"type": "image", "image": f2},
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ]
                    text = self.processor.apply_chat_template(
                        mm_messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text],
                        images=[[f1, f2]],
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    gen = self.processor.batch_decode(
                        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )[0].strip()

                    m = self.SIGNED_INT_RE.search(gen)
                    pred_ans = int(m.group(0)) if m else None

                    stats = buckets.setdefault(bucket, {"correct": 0, "total": 0, "unparsed": 0})
                    stats["total"] += 1
                    if pred_ans is None:
                        stats["unparsed"] += 1
                    elif (pred_ans > 0) == (target_ans > 0):
                        stats["correct"] += 1

                    if rank == 0 and len(qualitative) < self.n_qualitative:
                        qualitative.append({
                            "target": target_ans,
                            "pred_text": gen,
                            "bucket": bucket,
                            "demo": ex.get("demo_id_exact", "?"),
                        })
                except Exception as e:
                    logger.warning(f"PairwiseSignAccuracyCallback: example {i} failed: {e}")

            if world > 1:
                gathered = [None] * world
                try:
                    import torch.distributed as dist
                    dist.all_gather_object(gathered, buckets)
                    merged = {}
                    for b in gathered:
                        for k, v in b.items():
                            m = merged.setdefault(k, {"correct": 0, "total": 0, "unparsed": 0})
                            for kk in m:
                                m[kk] += v.get(kk, 0)
                    buckets = merged
                except Exception as e:
                    logger.warning(f"all_gather_object failed: {e}")

            total = sum(b["total"] for b in buckets.values())
            correct = sum(b["correct"] for b in buckets.values())
            unparsed = sum(b["unparsed"] for b in buckets.values())
            overall = correct / max(total, 1)

            def _agg(suffix):
                cs = [b for k, b in buckets.items() if k.endswith(suffix)]
                tot = sum(b["total"] for b in cs)
                cor = sum(b["correct"] for b in cs)
                unp = sum(b["unparsed"] for b in cs)
                return cor / max(tot, 1), unp / max(tot, 1), tot
            cam0_acc, cam0_unp, cam0_n = _agg("_cam0")
            cam1_acc, cam1_unp, cam1_n = _agg("_cam1")

            metrics = kwargs.get("metrics")
            if metrics is not None:
                metrics["eval_sign_acc_overall"] = overall
                metrics["eval_unparsed_overall"] = unparsed / max(total, 1)
                metrics["eval_sign_acc_cam0"] = cam0_acc
                metrics["eval_sign_acc_cam1"] = cam1_acc

            if rank == 0:
                logs = {
                    "eval/sign_acc_overall": overall,
                    "eval/unparsed_overall": unparsed / max(total, 1),
                    "eval/sign_acc_cam0": cam0_acc,
                    "eval/sign_acc_cam1": cam1_acc,
                    "eval/unparsed_cam0": cam0_unp,
                    "eval/unparsed_cam1": cam1_unp,
                    "eval/n_cam0": cam0_n,
                    "eval/n_cam1": cam1_n,
                }
                for k in sorted(buckets):
                    b = buckets[k]
                    logs[f"eval/sign_acc_{k}"] = b["correct"] / max(b["total"], 1)
                    logs[f"eval/unparsed_{k}"] = b["unparsed"] / max(b["total"], 1)
                    logs[f"eval/n_{k}"] = b["total"]

                logger.info(f"=== Sign accuracy @ step {state.global_step} (n={total}) ===")
                for k, v in sorted(logs.items()):
                    logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                if qualitative:
                    logger.info("=== Qualitative samples ===")
                    for q in qualitative:
                        logger.info(json.dumps(q))

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(logs, step=state.global_step)
                except Exception:
                    pass
        finally:
            if was_training:
                model.train()


@dataclass
class BagPlateArgs:
    dataset_root: str = "/proj/vondrick3/datasets/expert_data_jgd_bag_plate"
    matches_csv: str = "/proj/vondrick3/datasets/expert_data_jgd_bag_plate/similar_pairs_out_v1/matches.csv"
    eval_box_indices: str = "18-20"
    eval_glass_indices: str = "18-20"
    eval_remote_indices: str = "18-20"
    pair_sample_step: int = 2
    failure_last_frac: float = 0.95
    failure_min_frames: int = 8
    eval_max_pairs: int = 200
    just_visualize: bool = False
    task_name: str = "BagPlate"
    max_pixels: str = "640x360"  # WxH; processor budget per image


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, BagPlateArgs))
    script_args, training_args, model_args, cfg = parser.parse_args_and_config()
    training_args.max_length = None
    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_sign_acc_overall"
    training_args.greater_is_better = True
    training_args.completion_only_loss = True

    task_token = TASK_TOKENS[cfg.task_name]

    eval_sets = {
        "box": set(parse_indices(cfg.eval_box_indices)),
        "glass": set(parse_indices(cfg.eval_glass_indices)),
        "remote": set(parse_indices(cfg.eval_remote_indices)),
    }

    # Collect all demo indices referenced by the CSV, then load just those demos.
    needed = {"box": set(), "glass": set(), "remote": set()}
    for fail_kind, box_idx, fail_idx, _rank, _iou in read_matches_csv(cfg.matches_csv):
        needed["box"].add(box_idx)
        needed[fail_kind].add(fail_idx)
    logger.info(
        f"CSV references: {len(needed['box'])} box, "
        f"{len(needed['glass'])} glass, {len(needed['remote'])} remote demos"
    )

    box_demos = load_bag_plate_demos(cfg.dataset_root, sorted(needed["box"]), "box")
    glass_demos = load_bag_plate_demos(cfg.dataset_root, sorted(needed["glass"]), "glass")
    remote_demos = load_bag_plate_demos(cfg.dataset_root, sorted(needed["remote"]), "remote")

    demos_by_key = {}
    for d in box_demos + glass_demos + remote_demos:
        demos_by_key[(d["kind"], d["idx"])] = d

    train_pairs, eval_pairs, route_stats = build_pairs_from_csv(
        cfg.matches_csv,
        demos_by_key,
        eval_sets,
        task_token,
        cfg.failure_last_frac,
        cfg.failure_min_frames,
        cfg.pair_sample_step,
        rng=random.Random(42),
    )
    logger.info(
        f"Row routing: train={route_stats['n_train_rows']}, eval={route_stats['n_eval_rows']}, "
        f"missing_demo={route_stats['n_missing_demo']}"
    )
    logger.info(f"Pair counts -> train: {len(train_pairs)}, eval: {len(eval_pairs)}")

    # Held-out demos (those in any eval_*_indices set) must NEVER appear in
    # train rows. Non-eval-indexed demos may appear in both splits.
    held_out = {("box", i) for i in eval_sets["box"]} \
        | {("glass", i) for i in eval_sets["glass"]} \
        | {("remote", i) for i in eval_sets["remote"]}
    train_demo_keys = {(p["demo_id_exact"].split("_")[0], int(p["demo_id_exact"].split("_")[1]))
                       for p in train_pairs}
    # demo_id_exact only carries the failure-side demo; also collect box demos
    # from pair video paths via a second pass over pairs.
    for p in train_pairs:
        # video_path_1/2 end with "<root>/<kind>_<idx>_<ts>_frames_camN"
        for vp in (p["video_path_1"], p["video_path_2"]):
            base = os.path.basename(vp)
            m = re.match(r"^(box|glass|remote)_(\d+)_", base)
            if m:
                train_demo_keys.add((m.group(1), int(m.group(2))))
    leaks = train_demo_keys & held_out
    if leaks:
        raise RuntimeError(f"Held-out demos leaked into train: {sorted(leaks)}")
    logger.info(f"Train demos (failure side): {sorted({(p['demo_id_exact']) for p in train_pairs})}")
    logger.info(f"Eval demos  (failure side): {sorted({(p['demo_id_exact']) for p in eval_pairs})}")
    logger.info(f"Train bucket counts: {Counter(p['bucket'] for p in train_pairs)}")
    logger.info(f"Eval bucket counts:  {Counter(p['bucket'] for p in eval_pairs)}")

    rng = random.Random(42)
    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)

    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0

    if rank == 0:
        try:
            if train_pairs:
                visualize_dataset(train_pairs, training_args.output_dir, split_name="train", num_examples=20)
            if eval_pairs:
                visualize_dataset(eval_pairs, training_args.output_dir, split_name="eval", num_examples=20)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    if cfg.just_visualize:
        sys.exit(0)

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs) if eval_pairs and training_args.eval_strategy != "no" else None
    del train_pairs, eval_pairs
    gc.collect()
    logger.info(f"Train {len(train_ds)} / Eval {len(eval_ds) if eval_ds else 0}")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quant = get_quantization_config(model_args)
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=get_kbit_device_map() if quant is not None else None,
        quantization_config=quant,
    )
    mp_w, mp_h = (int(x) for x in cfg.max_pixels.lower().split("x"))
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        max_pixels=mp_w * mp_h,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
    )
    if isinstance(trainer.data_collator, DataCollatorForVisionLanguageModeling):
        old = trainer.data_collator
        trainer.data_collator = TwoImageCollator(
            processor=old.processor,
            max_length=old.max_length,
            completion_only_loss=old.completion_only_loss,
            pad_to_multiple_of=None,
        )
    if eval_ds is not None:
        trainer.add_callback(PairwiseSignAccuracyCallback(
            eval_ds, processor,
            max_pairs=cfg.eval_max_pairs,
            n_qualitative=3,
        ))

    has_ckpt = os.path.isdir(training_args.output_dir) and any(
        d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, d))
        for d in os.listdir(training_args.output_dir)
    )
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(training_args.output_dir)
