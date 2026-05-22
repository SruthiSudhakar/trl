"""
Trainer for BagPlate pairwise progress comparison (two camera views).

Layout assumed at --dataset_root:
    <kind>_<N>_<timestamp>.npy            # dict with image_paths_cam0 / image_paths_cam1
    <kind>_<N>_<timestamp>_frames_cam0/   # frame_000000.jpg ...
    <kind>_<N>_<timestamp>_frames_cam1/   # frame_000000.jpg ...
where kind in {"box","glass","remote"}.

Semantics: `box` is the success category. `glass` and `remote` are both
failure categories (two distinct failure modes); we still treat them
identically when forming pairs, but tag them separately in `bucket` and
`demo_id_exact` so the eval callback reports per-mode metrics and the
train/eval overlap check works correctly across kinds.

Pairing (same as upright bottle):
- success-vs-success: same box demo, later sub-frame = more progress.
- failure-vs-success: pair the LAST K sub-frames of glass_N / remote_N
  (where the failure has clearly happened) against the temporally aligned
  sub-frames of box_N. K = max(failure_min_frames, round(failure_last_frac * len)).

For every logical pair, we emit TWO examples - one using cam0 frames and one
using cam1 frames - so train/eval are balanced 50/50 across camera views.
"""

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


def build_pairs_for_demos(
    successes,
    failures,
    intervals,
    train_step,
    task_token,
    failure_last_frac,
    failure_min_frames,
    rng=None,
    include_succ_vs_succ=True,
):
    """Build success-vs-success and failure-vs-success pairs.

    `successes` is the list of `box` demos. `failures` is the concatenation
    of `glass` and `remote` demos; each carries its own `kind` so we can tag
    buckets and demo_id_exact distinctly (glass_1 vs remote_1 would otherwise
    collide on demo_id_exact = "failure_1").
    """
    rng = rng or random.Random(42)
    user_prompt = USER_PROMPT_TEMPLATE.format(task_token=task_token)
    succ_by_idx = {s["idx"]: s for s in successes}
    out = []
    CAMERAS = ("cam0", "cam1")

    def emit(v1, f1, v2, f2, ans, demo_success, demo_kind, demo_id, bucket, camera):
        out.append({
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
            "demo_id": demo_id,
            "demo_id_exact": f"{demo_kind}_{demo_id}",
            "demo_success": demo_success,
            "bucket": bucket,
            "camera": camera,
            "job_name": "BagPlate",
            "task_token": task_token,
        })

    # success-vs-success: later frame from same box demo shows more progress.
    if include_succ_vs_succ:
        for s in successes:
            sub_max = (s["n_frames"] - 1) // SUBSAMPLE
            for interval in intervals:
                if sub_max <= interval:
                    continue
                offset = rng.randint(0, train_step - 1)
                count = 0
                for i1 in range(offset, sub_max - interval, train_step):
                    i2 = i1 + interval
                    f1, f2 = i1 * SUBSAMPLE, i2 * SUBSAMPLE
                    for cam in CAMERAS:
                        v_path = s[f"video_path_{cam}"]
                        ans = ANSWER_MAGNITUDE
                        v1, vf1, v2, vf2 = v_path, f1, v_path, f2
                        if rng.random() < 0.5:
                            v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                            ans = -ans
                        emit(
                            v1, vf1, v2, vf2, ans, "success", "box", s["idx"],
                            f"succ_vs_succ_int{interval}_{cam}", cam,
                        )
                    count += 1
                logger.info(f"box_{s['idx']} interval={interval}: {count} pairs x {len(CAMERAS)} cameras")
    else:
        logger.info("Skipping success-vs-success pair generation (include_succ_vs_succ=False)")

    # failure-vs-success: last K sub-frames of glass_N / remote_N vs aligned box_N.
    for f in failures:
        s = succ_by_idx.get(f["idx"])
        if s is None:
            logger.warning(f"{f['kind']}_{f['idx']} has no matching box_{f['idx']}, skipping")
            continue
        f_sub_max = (f["n_frames"] - 1) // SUBSAMPLE
        s_sub_max = (s["n_frames"] - 1) // SUBSAMPLE
        n_sub = f_sub_max + 1
        if n_sub < 1:
            continue
        k = max(failure_min_frames, int(round(failure_last_frac * n_sub)))
        k = min(k, n_sub)
        last_indices = list(range(n_sub - k, n_sub))
        offset = rng.randint(0, train_step - 1)
        kept = last_indices[offset::train_step]
        for i in kept:
            f_src = i * SUBSAMPLE
            s_src = min(i, s_sub_max) * SUBSAMPLE
            for cam in CAMERAS:
                f_path = f[f"video_path_{cam}"]
                s_path = s[f"video_path_{cam}"]
                ans = ANSWER_MAGNITUDE
                v1, vf1, v2, vf2 = f_path, f_src, s_path, s_src
                if rng.random() < 0.5:
                    v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                    ans = -ans
                emit(
                    v1, vf1, v2, vf2, ans, "failure", f["kind"], f["idx"],
                    f"fail_vs_succ_{f['kind']}_{cam}", cam,
                )
        logger.info(
            f"{f['kind']}_{f['idx']}: n_sub={n_sub}, k={k}, "
            f"last_range=[{last_indices[0]}..{last_indices[-1]}], "
            f"emitted {len(kept)} pairs x {len(CAMERAS)} cameras"
        )

    return out


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
    box_indices: str = "1-20"
    glass_indices: str = "1-20"
    remote_indices: str = "1-20"
    eval_box_indices: str = "18-20"
    eval_glass_indices: str = "18-20"
    eval_remote_indices: str = "18-20"
    compare_interval: str = "4,8,12,16"
    train_sample_interval: int = 4
    failure_last_frac: float = 0.95
    failure_min_frames: int = 8
    eval_max_pairs: int = 200
    just_visualize: bool = False
    task_name: str = "BagPlate"
    max_pixels: str = "640x360"  # WxH; processor budget per image
    balance_fail_vs_succ: bool = False
    include_succ_vs_succ: bool = True


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, BagPlateArgs))
    script_args, training_args, model_args, cfg = parser.parse_args_and_config()
    training_args.max_length = None
    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_sign_acc_overall"
    training_args.greater_is_better = True
    training_args.completion_only_loss = True

    intervals = [int(x) for x in cfg.compare_interval.split(",")]
    task_token = TASK_TOKENS[cfg.task_name]

    box_idxs = parse_indices(cfg.box_indices)
    glass_idxs = parse_indices(cfg.glass_indices)
    remote_idxs = parse_indices(cfg.remote_indices)
    eval_box_set = set(parse_indices(cfg.eval_box_indices))
    eval_glass_set = set(parse_indices(cfg.eval_glass_indices))
    eval_remote_set = set(parse_indices(cfg.eval_remote_indices))

    box_demos = load_bag_plate_demos(cfg.dataset_root, box_idxs, "box")
    glass_demos = load_bag_plate_demos(cfg.dataset_root, glass_idxs, "glass")
    remote_demos = load_bag_plate_demos(cfg.dataset_root, remote_idxs, "remote")

    train_box = [d for d in box_demos if d["idx"] not in eval_box_set]
    eval_box = [d for d in box_demos if d["idx"] in eval_box_set]
    train_glass = [d for d in glass_demos if d["idx"] not in eval_glass_set]
    eval_glass = [d for d in glass_demos if d["idx"] in eval_glass_set]
    train_remote = [d for d in remote_demos if d["idx"] not in eval_remote_set]
    eval_remote = [d for d in remote_demos if d["idx"] in eval_remote_set]

    train_failures = train_glass + train_remote
    eval_failures = eval_glass + eval_remote

    logger.info(
        f"Demo split -> train: {len(train_box)} box, {len(train_glass)} glass, "
        f"{len(train_remote)} remote; eval: {len(eval_box)} box, "
        f"{len(eval_glass)} glass, {len(eval_remote)} remote"
    )

    train_pairs = build_pairs_for_demos(
        train_box, train_failures, intervals, cfg.train_sample_interval, task_token,
        cfg.failure_last_frac, cfg.failure_min_frames,
        rng=random.Random(42),
        include_succ_vs_succ=cfg.include_succ_vs_succ,
    )
    eval_pairs = build_pairs_for_demos(
        eval_box, eval_failures, intervals, cfg.train_sample_interval, task_token,
        cfg.failure_last_frac, cfg.failure_min_frames,
        rng=random.Random(43),
        include_succ_vs_succ=cfg.include_succ_vs_succ,
    )
    logger.info(f"Pair counts -> train: {len(train_pairs)}, eval: {len(eval_pairs)}")

    if cfg.balance_fail_vs_succ:
        succ_pairs = [p for p in train_pairs if not p["bucket"].startswith("fail_vs_succ")]
        fail_pairs = [p for p in train_pairs if p["bucket"].startswith("fail_vs_succ")]
        if fail_pairs and succ_pairs:
            reps = len(succ_pairs) // len(fail_pairs)
            remainder = len(succ_pairs) - reps * len(fail_pairs)
            balance_rng = random.Random(44)
            balanced_fail = fail_pairs * reps + balance_rng.sample(fail_pairs, remainder)
            train_pairs = succ_pairs + balanced_fail
            balance_rng.shuffle(train_pairs)
            logger.info(
                f"Balanced fail_vs_succ: {len(fail_pairs)} unique -> {len(balanced_fail)} "
                f"after replication ({reps}x + {remainder} extra) to match {len(succ_pairs)} succ_vs_succ pairs. "
                f"New train total: {len(train_pairs)}"
            )
        else:
            logger.warning("balance_fail_vs_succ requested but one bucket is empty; skipping.")

    train_demos = {(p["demo_success"], p["demo_id_exact"]) for p in train_pairs}
    eval_demos = {(p["demo_success"], p["demo_id_exact"]) for p in eval_pairs}
    overlap = train_demos & eval_demos
    if overlap:
        raise RuntimeError(f"Train/eval demo overlap detected: {overlap}")
    logger.info(f"Train demos: {sorted(train_demos)}")
    logger.info(f"Eval demos:  {sorted(eval_demos)}")
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
