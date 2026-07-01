"""
Generalizable trainer for robocasa policy-rollout pairwise progress comparison.

Unlike sft_vlm_stacking.py (which reads .npy metadata + pre-extracted jpg frame
dirs), this script trains directly on policy-rollout evaluation videos. You point
it at the directory that holds all rollout subdirs and pick a task; it finds every
rollout subdir for that task and builds the same two-image progress-comparison
dataset that sft_vlm_stacking.py uses.

Expected layout at --pretrain_root (one subdir per rollout):
    <TASK>_<...>_seed<seed>/eval_log.json   # per-episode success + metadata
    <TASK>_<...>_seed<seed>/media/seed1000XX.mp4

eval_log.json fields we use:
    eval_args.task                       # task name, matched against --task
    test/sim_max_reward_1000XX  = 1.0/0.0  # episode 1000XX success / failure
The mp4 path is reconstructed as <rollout_dir>/media/seed1000XX.mp4 (the
test/sim_video_* paths stored in the json are stale).

Initial conditions: the env reset seed is fixed across rollouts, so episode index
E (100000..100049) denotes the SAME initial condition in every rollout. The same E
succeeds in some rollouts and fails in others, which gives failure<->success pairs
that share an initial condition.

Pairing (mirrors sft_vlm_stacking, single camera):
- success-vs-success: same success video, later sub-frame = more progress.
- failure-vs-success: LAST K sub-frames of a failure video at episode E vs the
  temporally aligned sub-frames of up to --max_succ_per_fail success videos at the
  SAME episode E (same initial condition).

Frames are decoded on the fly from the mp4 via decord (extract_frame), so there is
no on-disk conversion step. The dataset rows use the exact schema consumed by
sft_vlm_stacking's TwoImageCollator / PairwiseSignAccuracyCallback / visualize.
"""

import gc
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from collections import Counter
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

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

# Sibling imports work regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_overlay_regression_v2 import TASK_TOKENS, visualize_dataset  # noqa: E402
from sft_vlm_stacking import (  # noqa: E402
    ANSWER_MAGNITUDE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    PairwiseSignAccuracyCallback,
    TwoImageCollator,
)
from video_frame_utils import find_job_dirs  # noqa: E402

# For multi-task training we want a per-video natural-language description in
# the prompt. PairwiseSignAccuracyCallback re-renders the user text at eval
# time from the imported USER_PROMPT_TEMPLATE (single {task_token} slot), so
# rather than override the template (which would diverge at eval), we fold the
# description into the same task_token slot: each row stores
#     task_token = "[CURATED_TOKEN] — natural-language description"
# and both train and eval render identical text via the existing template.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)

DEFAULT_PRETRAIN_ROOT = (
    "/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/data/jgd/2026.05.23/"
    "23.28.09_train_diffusion_unet_hybrid_target_atomic_seen_jgd_CloseToasterOvenDoor/"
    "evals/epoch=0100-train_loss=0.0027/pretrain"
)


def task_token_for(task: str) -> str:
    """Reuse the curated token if the task is known, else auto-derive an
    UPPER_SNAKE token (CloseToasterOvenDoor -> [CLOSE_TOASTER_OVEN_DOOR])."""
    if task in TASK_TOKENS:
        return TASK_TOKENS[task]
    return "[" + re.sub(r"(?<!^)(?=[A-Z])", "_", task).upper() + "]"


@lru_cache(maxsize=8192)
def video_num_frames(path: str) -> int:
    """Number of frames in a video, or 0 if the file is unreadable/corrupt.

    Some rollout mp4s are written with a broken codec header (decord raises
    DECORDError "Cannot create buffer source" / invalid pixel format). Treat
    those as having 0 frames so callers can skip them instead of crashing the
    whole run.
    """
    import decord
    try:
        return len(decord.VideoReader(path, num_threads=1))
    except Exception as e:
        logger.warning(f"Unreadable/corrupt video, treating as 0 frames: {path} ({e})")
        return 0


def load_recovered_lang(rollout_dir: str) -> dict:
    """Read recovered_lang.json sitting next to eval_log.json.
    Returns {video_basename: lang_text}. Empty dict if missing/unreadable."""
    path = os.path.join(rollout_dir, "recovered_lang.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}
    return {k: v.get("lang", "") for k, v in data.items() if isinstance(v, dict)}


def load_rollout_demos(pretrain_root: str, task: str):
    """Scan every rollout subdir for `task` and group videos by episode index
    (= initial condition). Returns a (by_ep, lang_by_path) tuple where
    by_ep = {ep: {"succ": [mp4...], "fail": [mp4...]}} and
    lang_by_path = {mp4_path: natural-language description} pulled from each
    rollout dir's recovered_lang.json."""
    dirs = find_job_dirs(pretrain_root)
    by_ep, lang_by_path = {}, {}
    n_succ = n_fail = n_dirs = n_corrupt = n_lang = 0
    for d in dirs:
        log_path = os.path.join(d, "eval_log.json")
        try:
            with open(log_path) as fh:
                log = json.load(fh)
        except Exception as e:
            logger.warning(f"Could not read {log_path}: {e}")
            continue
        if log.get("eval_args", {}).get("task") != task:
            continue
        n_dirs += 1
        lang_in_dir = load_recovered_lang(d)
        for k, v in log.items():
            if not k.startswith("test/sim_max_reward_"):
                continue
            ep = int(k.rsplit("_", 1)[1])
            vname = f"seed{ep}.mp4"
            vpath = os.path.join(d, "media", vname)
            if not os.path.isfile(vpath):
                logger.warning(f"Missing video {vpath}, skipping")
                continue
            # Drop unreadable/corrupt videos up front so they never enter the
            # pairing pool (otherwise decord crashes the run when they are read).
            if video_num_frames(vpath) < 2:
                n_corrupt += 1
                continue
            slot = by_ep.setdefault(ep, {"succ": [], "fail": []})
            if float(v) >= 1.0:
                slot["succ"].append(vpath)
                n_succ += 1
            else:
                slot["fail"].append(vpath)
                n_fail += 1
            lang = lang_in_dir.get(vname)
            if lang:
                lang_by_path[vpath] = lang
                n_lang += 1
    pairable = sum(1 for s in by_ep.values() if s["succ"] and s["fail"])
    logger.info(
        f"[{task}] {n_dirs} rollout dirs -> {len(by_ep)} initial conditions; "
        f"{n_succ} success videos, {n_fail} failure videos, "
        f"{n_corrupt} corrupt videos skipped, "
        f"{pairable} pairable initial conditions (have both succ & fail); "
        f"{n_lang} videos with recovered lang"
    )
    return by_ep, lang_by_path


def build_rollout_pairs(
    by_ep,
    episodes,
    intervals,
    train_step,
    task_token,
    task_name,
    lang_by_path,
    fallback_desc,
    failure_last_frac,
    failure_min_frames,
    max_succ_per_fail,
    subsample,
    video_skip_frac=0.0,
    rng=None,
):
    """Build success-vs-success and failure-vs-success pairs for the given
    episode indices. Single camera; bucket names end in "_cam0" so the existing
    PairwiseSignAccuracyCallback per-camera aggregation works unchanged. Each
    pair's user-prompt carries the natural-language description for video_1
    (with v2 / fallback as backstop), so the model can disambiguate tasks in
    multi-task training."""
    rng = rng or random.Random(42)
    out = []
    CAM = "cam0"

    def emit(v1, f1, v2, f2, ans, kind, demo_id, bucket):
        desc = lang_by_path.get(v1) or lang_by_path.get(v2) or fallback_desc
        # Fold the description into the same task_token slot the eval callback
        # reads, so train and eval render identical user text.
        combined_token = f"{task_token} — {desc}"
        user_prompt = USER_PROMPT_TEMPLATE.format(task_token=combined_token)
        out.append({
            "video_path_1": v1, "frame_idx_1": f1,
            "video_path_2": v2, "frame_idx_2": f2,
            "correct_answer": ans,
            # Prompt-completion format so completion_only_loss can mask the long
            # constant prefix and only train on the answer tokens.
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "completion": [
                {"role": "assistant", "content": str(ans)},
            ],
            "demo_id": demo_id,
            "demo_id_exact": f"{kind}_{demo_id}",
            "demo_success": kind,
            "bucket": f"{task_name}_{bucket}",
            "camera": CAM,
            "job_name": task_name,
            "task_token": combined_token,
            "task_token_pure": task_token,
            "task_name": task_name,
            "task_description": desc,
        })

    # success-vs-success: later sub-frame from the same success video = more progress.
    n_ss = 0
    for ep in episodes:
        for spath in by_ep[ep]["succ"]:
            sub_max = (video_num_frames(spath) - 1) // subsample
            # drop the last video_skip_frac of each video; usable = count of kept frames
            usable = int(round((1.0 - video_skip_frac) * (sub_max + 1)))
            for interval in intervals:
                if usable <= interval:
                    continue
                offset = rng.randint(0, train_step - 1)
                for i1 in range(offset, usable - interval, train_step):
                    i2 = i1 + interval
                    f1, f2 = i1 * subsample, i2 * subsample
                    ans = ANSWER_MAGNITUDE
                    v1, vf1, v2, vf2 = spath, f1, spath, f2
                    if rng.random() < 0.5:
                        v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                        ans = -ans
                    emit(v1, vf1, v2, vf2, ans, "success", ep,
                         f"succ_vs_succ_int{interval}_{CAM}")
                    n_ss += 1
    logger.info(f"succ_vs_succ pairs: {n_ss}")

    # failure-vs-success: last K sub-frames of a failure at episode E vs aligned
    # sub-frames of matched success videos at the SAME episode E.
    n_fs = 0
    for ep in episodes:
        succs = by_ep[ep]["succ"]
        fails = by_ep[ep]["fail"]
        if not succs or not fails:
            continue
        for fpath in fails:
            f_sub_max = (video_num_frames(fpath) - 1) // subsample
            n_sub = f_sub_max + 1
            if n_sub < 1:
                continue
            # drop the last video_skip_frac of the video first; usable = kept-frame count
            usable = int(round((1.0 - video_skip_frac) * n_sub))
            k = max(failure_min_frames, int(round(failure_last_frac * usable)))
            k = min(k, usable)
            last_indices = list(range(usable - k, usable))
            for spath in rng.sample(succs, min(max_succ_per_fail, len(succs))):
                s_sub_max = (video_num_frames(spath) - 1) // subsample
                offset = rng.randint(0, train_step - 1)
                for i in last_indices[offset::train_step]:
                    f_src = i * subsample
                    s_src = min(i, s_sub_max) * subsample
                    ans = ANSWER_MAGNITUDE
                    v1, vf1, v2, vf2 = fpath, f_src, spath, s_src
                    if rng.random() < 0.5:
                        v1, vf1, v2, vf2 = v2, vf2, v1, vf1
                        ans = -ans
                    emit(v1, vf1, v2, vf2, ans, "failure", ep, f"fail_vs_succ_{CAM}")
                    n_fs += 1
    logger.info(f"fail_vs_succ pairs: {n_fs}")
    return out


@dataclass
class RolloutArgs:
    pretrain_root: str = DEFAULT_PRETRAIN_ROOT
    # List[str] -> HfArgumentParser uses nargs="+", so the CLI is
    # `--task TaskA TaskB TaskC ...` (single task still works).
    task: List[str] = field(default_factory=lambda: ["CloseToasterOvenDoor"])
    num_eval_episodes: int = 10  # held-out initial conditions (with successes) for eval
    compare_interval: str = "5,10,20,40"
    train_sample_interval: int = 4
    failure_last_frac: float = 0.5
    failure_min_frames: int = 8
    video_skip_frac: float = 0.0  # drop the first N fraction of frames from ALL videos
    max_succ_per_fail: int = 3  # success videos paired per failure video at the same episode
    subsample: int = 1  # frame stride (videos are 10 fps)
    eval_max_pairs: int = 200
    just_visualize: bool = False
    max_pixels: str = "256x256"  # WxH; processor budget per image
    balance_fail_vs_succ: bool = False


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, RolloutArgs))
    script_args, training_args, model_args, cfg = parser.parse_args_and_config()
    training_args.max_length = None
    training_args.remove_unused_columns = False
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "eval_sign_acc_overall"
    training_args.greater_is_better = True
    # Force loss only on the assistant completion tokens (prompt prefix is long
    # and constant; without this the model memorizes it and collapses).
    training_args.completion_only_loss = True

    intervals = [int(x) for x in cfg.compare_interval.split(",")]

    # Multi-task: build pairs per-task, then concat. by_ep is rebuilt per task so
    # pairs are always within a single task by construction (no cross-task pairs).
    # When --balance_fail_vs_succ is set, balance is applied INSIDE the loop so
    # every task contributes equal succ-vs-succ and fail-vs-succ counts (a global
    # balance would let tasks with more failures dominate the fail share).
    train_pairs, eval_pairs = [], []
    train_rng = random.Random(42)
    eval_rng = random.Random(43)
    balance_rng = random.Random(44) if cfg.balance_fail_vs_succ else None
    for task in cfg.task:
        task_token = task_token_for(task)
        # Fallback description if a video isn't in recovered_lang.json: split
        # CamelCase task name into lowercase words ("PackDessert" -> "pack dessert").
        fallback_desc = re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower()
        logger.info(
            f"[{task}] token={task_token}  fallback_desc='{fallback_desc}'"
        )

        by_ep, lang_by_path = load_rollout_demos(cfg.pretrain_root, task)
        if not by_ep:
            logger.warning(f"[{task}] no rollouts found, skipping")
            continue
        episodes_all = sorted(by_ep.keys())
        if not any(by_ep[e]["succ"] for e in episodes_all):
            raise ValueError(
                f"No successful rollouts for task '{task}' under {cfg.pretrain_root}; "
                f"cannot build training pairs."
            )

        # Hold out the last num_eval_episodes PAIRABLE initial conditions per task.
        pair_eps = [e for e in episodes_all if by_ep[e]["succ"] and by_ep[e]["fail"]]
        n_eval = min(cfg.num_eval_episodes, max(0, len(pair_eps) - 1))
        eval_eps = set(pair_eps[-n_eval:]) if n_eval > 0 else set()
        train_eps = [e for e in episodes_all if e not in eval_eps]
        eval_eps_list = sorted(eval_eps)
        if not eval_eps_list:
            logger.warning(
                f"No held-out eval initial conditions for task '{task}' "
                f"(pairable={len(pair_eps)}); training without an eval split."
            )
        logger.info(
            f"[{task}] split -> train: {len(train_eps)} initial conditions, "
            f"eval: {len(eval_eps_list)} initial conditions ({eval_eps_list})"
        )

        task_train_pairs = build_rollout_pairs(
            by_ep, train_eps, intervals, cfg.train_sample_interval,
            task_token, task, lang_by_path, fallback_desc,
            cfg.failure_last_frac, cfg.failure_min_frames, cfg.max_succ_per_fail,
            cfg.subsample, video_skip_frac=cfg.video_skip_frac, rng=train_rng,
        )
        task_eval_pairs = build_rollout_pairs(
            by_ep, eval_eps_list, intervals, cfg.train_sample_interval,
            task_token, task, lang_by_path, fallback_desc,
            cfg.failure_last_frac, cfg.failure_min_frames, cfg.max_succ_per_fail,
            cfg.subsample, video_skip_frac=cfg.video_skip_frac, rng=eval_rng,
        )

        if cfg.balance_fail_vs_succ:
            # Per-task balance: upsample fail-vs-succ pairs to match
            # succ-vs-succ count within THIS task only.
            succ_p = [p for p in task_train_pairs if p["demo_success"] != "failure"]
            fail_p = [p for p in task_train_pairs if p["demo_success"] == "failure"]
            if fail_p and succ_p:
                reps = len(succ_p) // len(fail_p)
                remainder = len(succ_p) - reps * len(fail_p)
                balanced_fail = fail_p * reps + balance_rng.sample(fail_p, remainder)
                task_train_pairs = succ_p + balanced_fail
                logger.info(
                    f"[{task}] Balanced fail_vs_succ: {len(fail_p)} unique -> "
                    f"{len(balanced_fail)} to match {len(succ_p)} succ_vs_succ; "
                    f"task train total: {len(task_train_pairs)}"
                )
            else:
                logger.warning(
                    f"[{task}] balance_fail_vs_succ requested but one bucket is "
                    f"empty (succ={len(succ_p)}, fail={len(fail_p)}); skipping."
                )

        train_pairs += task_train_pairs
        eval_pairs += task_eval_pairs
    logger.info(
        f"Pair counts -> train: {len(train_pairs)}, eval: {len(eval_pairs)} "
        f"across {len(cfg.task)} tasks ({list(cfg.task)})"
    )
    if not train_pairs:
        raise ValueError(
            f"No training pairs built across tasks {list(cfg.task)} under "
            f"{cfg.pretrain_root}; check task names and rollout layout."
        )

    # (balance_fail_vs_succ is applied per-task inside the loop above so each
    # task's succ/fail counts are matched independently; no global step here.)

    # Key on task_name too: the same episode index can legitimately appear in
    # different tasks, but train/eval must still be disjoint within each task.
    train_demos = {(p["task_name"], p["demo_success"], p["demo_id"]) for p in train_pairs}
    eval_demos = {(p["task_name"], p["demo_success"], p["demo_id"]) for p in eval_pairs}
    overlap = train_demos & eval_demos
    if overlap:
        raise RuntimeError(f"Train/eval initial-condition overlap detected: {overlap}")
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

    mp_w, mp_h = (int(x) for x in cfg.max_pixels.lower().split("x"))
    max_pixels_budget = mp_w * mp_h

    if rank == 0:
        try:
            if train_pairs:
                visualize_dataset(
                    train_pairs, training_args.output_dir, split_name="train",
                    num_examples=20, max_pixels=max_pixels_budget,
                )
            if eval_pairs:
                visualize_dataset(
                    eval_pairs, training_args.output_dir, split_name="eval",
                    num_examples=20, max_pixels=max_pixels_budget,
                )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    if cfg.just_visualize:
        sys.exit(0)

    train_ds = Dataset.from_list(train_pairs)
    # HF's eval loop iterates the full eval_ds to compute eval_loss every
    # eval_steps. Our real metric (eval_sign_acc_overall) comes from the
    # PairwiseSignAccuracyCallback, which already subsamples to eval_max_pairs.
    # Cap eval_ds at 500 so eval_loss is cheap; eval_pairs was shuffled above.
    eval_ds = Dataset.from_list(eval_pairs[:500]) if eval_pairs and training_args.eval_strategy != "no" else None
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
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        max_pixels=max_pixels_budget,
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

    # Drop the frame-count cache built during pair construction; readers are now
    # opened per-process in video_frame_utils so no fork-unsafe decord handles
    # leak into the DataLoader workers.
    video_num_frames.cache_clear()

    # Auto-resume if output_dir already contains a checkpoint-* directory.
    has_ckpt = os.path.isdir(training_args.output_dir) and any(
        d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, d))
        for d in os.listdir(training_args.output_dir)
    )
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    trainer.save_model(training_args.output_dir)
