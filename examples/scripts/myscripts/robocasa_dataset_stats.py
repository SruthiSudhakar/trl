"""Compute per-task dataset statistics for the 17-task multitask RoboCasa VLM
critic finetuning run (see launch_robocasa.sh).

For each task we report:
    # Succ.  : number of success rollout videos
    # Fail   : number of failure rollout videos
    # S-S    : success-vs-success pair count built for training (pre-balance)
    # S-F    : success-vs-failure pair count built for training (pre-balance)

Counts are produced by calling the exact same helpers used by the trainer
(load_rollout_demos / build_rollout_pairs) with the launch_robocasa.sh args.
"""
import json
import logging
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft_vlm_robocasa import (  # noqa: E402
    build_rollout_pairs,
    load_recovered_lang,
    task_token_for,
    video_num_frames,
)
from video_frame_utils import find_job_dirs  # noqa: E402

logger = logging.getLogger(__name__)


def load_rollout_demos_no_vlm(pretrain_root: str, task: str):
    """Same as sft_vlm_robocasa.load_rollout_demos, but skip rollout dirs whose
    basename contains 'vlm' (case-insensitive)."""
    dirs = [d for d in find_job_dirs(pretrain_root)
            if "vlm" not in os.path.basename(d).lower()]
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
                continue
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
    print(f"[{task}] {n_dirs} non-VLM rollout dirs, {n_succ} succ, {n_fail} fail")
    return by_ep, lang_by_path

TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
PRETRAIN_ROOT = (
    "/proj/vondrick3/sruthi/Appaji/released_checkpoints_groot/gr00t_n1-5/"
    "multitask_learning/checkpoint-120000/evals/pretrain"
)
COMPARE_INTERVAL = [4, 8, 12, 16]
TRAIN_SAMPLE_INTERVAL = 4
FAILURE_LAST_FRAC = 0.95
FAILURE_MIN_FRAMES = 8
MAX_SUCC_PER_FAIL = 1
SUBSAMPLE = 1
NUM_EVAL_EPISODES = 8


def main():
    rows = []
    for task in TASKS:
        by_ep, lang_by_path = load_rollout_demos_no_vlm(PRETRAIN_ROOT, task)
        if not by_ep:
            rows.append((task, 0, 0, 0, 0))
            continue

        n_succ = sum(len(s["succ"]) for s in by_ep.values())
        n_fail = sum(len(s["fail"]) for s in by_ep.values())

        episodes_all = sorted(by_ep.keys())
        pair_eps = [e for e in episodes_all if by_ep[e]["succ"] and by_ep[e]["fail"]]
        n_eval = min(NUM_EVAL_EPISODES, max(0, len(pair_eps) - 1))
        eval_eps = set(pair_eps[-n_eval:]) if n_eval > 0 else set()
        train_eps = [e for e in episodes_all if e not in eval_eps]

        token = task_token_for(task)
        fallback_desc = re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower()
        train_pairs = build_rollout_pairs(
            by_ep, train_eps, COMPARE_INTERVAL, TRAIN_SAMPLE_INTERVAL,
            token, task, lang_by_path, fallback_desc,
            FAILURE_LAST_FRAC, FAILURE_MIN_FRAMES, MAX_SUCC_PER_FAIL,
            SUBSAMPLE, video_skip_frac=0.0, rng=random.Random(42),
        )
        ss = sum(1 for p in train_pairs if p["demo_success"] != "failure")
        sf = sum(1 for p in train_pairs if p["demo_success"] == "failure")
        rows.append((task, n_succ, n_fail, ss, sf))
        print(f"{task:30s}  succ={n_succ:4d}  fail={n_fail:4d}  S-S={ss:6d}  S-F={sf:6d}")

    print("\n=== LaTeX rows ===")
    for task, ns, nf, ss, sf in rows:
        print(f"{task:30s} & {ns:3d} & {nf:3d} & {ss:6,d} & {sf:6,d} \\\\")


if __name__ == "__main__":
    main()
