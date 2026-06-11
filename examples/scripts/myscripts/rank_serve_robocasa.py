"""Long-running ranker server for RoboCasa policy-rollout checkpoints.

Same inbox/poll job protocol as rank_serve.py, but resolves the task token the
way sft_vlm_robocasa.py does at train time: if the task is in the curated
TASK_TOKENS it uses that, otherwise it auto-derives an UPPER_SNAKE token
(CloseToasterOvenDoor -> [CLOSE_TOASTER_OVEN_DOOR]). The pairwise-comparison
prompt sft_vlm_robocasa trains on (from sft_vlm_stacking) is byte-identical to
the LEGO prompt rank_videos already uses, so we reuse rank_videos' machinery and
only swap in the right token.

Loads the Qwen2.5-VL ranker once at startup and reuses it across jobs.

Job file format (written atomically by the producer — tmp -> rename):
    {
      "name": "<stem>",
      "output_subdir": "/abs/.../<name>",       # ranking.json is written here
      "video_paths": ["/abs/.../0.mp4", "/abs/.../1.mp4", ...],  # >= 2
      "task_name": "CloseToasterOvenDoor"
    }

Usage (under the vlmoverlay env):
    conda activate vlmoverlay
    cd /proj/vondrick3/sruthi/Appaji/trl
    CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_serve_robocasa.py \
        --inbox_dir /tmp/rc_rank/inbox --done_dir /tmp/rc_rank/done --error_dir /tmp/rc_rank/err \
        --task_name CloseToasterOvenDoor --gpu_ids 0 --batch_size 4 \
        --checkpoint /proj/vondrick3/sruthi/Appaji/trl/outputs/CloseToasterOvenDoor_20260524_170526_robocasa/checkpoint-750
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_videos as rv  # noqa: E402  (load_ranker_multi, rank_videos_multi, TASK_TOKENS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def task_token_for(task: str) -> str:
    """Curated token if known, else auto-derive UPPER_SNAKE (matches sft_vlm_robocasa)."""
    if task in rv.TASK_TOKENS:
        return rv.TASK_TOKENS[task]
    return "[" + re.sub(r"(?<!^)(?=[A-Z])", "_", task).upper() + "]"


def process_job(job_path, processors, models, default_task_name, batch_size, frame_fraction):
    with open(job_path) as f:
        job = json.load(f)
    name = job.get("name", job_path.stem)
    output_subdir = Path(job["output_subdir"])
    video_paths = job["video_paths"]
    task_name = job.get("task_name", default_task_name)

    if len(video_paths) < 2:
        raise ValueError(f"job {name}: expected >=2 video paths, got {len(video_paths)}")
    for v in video_paths:
        if not (os.path.isfile(v) or os.path.isdir(v)):
            raise FileNotFoundError(f"job {name}: missing video {v}")
    if not output_subdir.is_dir():
        raise FileNotFoundError(f"job {name}: output_subdir does not exist: {output_subdir}")

    # Register the (possibly auto-derived) token so rank_videos_multi's TASK_TOKENS[...] lookup works.
    rv.TASK_TOKENS[task_name] = task_token_for(task_name)
    logger.info(f"[rank_serve_robocasa] {name}: task={task_name} token={rv.TASK_TOKENS[task_name]} "
                f"({len(video_paths)} videos)")

    result = rv.rank_videos_multi(
        video_paths, processors, models, task_name=task_name,
        batch_size=batch_size, frame_fraction=frame_fraction,
    )
    # write atomically so the producer never reads a half-written ranking.json
    out_path = output_subdir / "ranking.json"
    tmp_path = output_subdir / "ranking.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)
    logger.info(f"[rank_serve_robocasa] wrote {out_path} (winner_idx={result['winner_idx']})")


def archive(job_path, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / job_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(job_path), str(target))


def main():
    p = argparse.ArgumentParser(description="Long-running RoboCasa video ranker server.")
    p.add_argument("--inbox_dir", required=True)
    p.add_argument("--done_dir", required=True)
    p.add_argument("--error_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_name", default="CloseToasterOvenDoor")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu_ids", default="0", help="Comma-separated GPU ids to shard pairs across.")
    p.add_argument("--frame_fraction", type=float, default=1.0,
                   help="Which frame to compare; fraction of the last index (1.0=last).")
    p.add_argument("--poll_interval", type=float, default=1.0)
    args = p.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    processors, models = rv.load_ranker_multi(args.checkpoint, args.max_pixels, gpu_ids)
    logger.info(f"[rank_serve_robocasa] loaded ranker on {len(gpu_ids)} GPU(s): {gpu_ids}")

    inbox, done_dir, error_dir = Path(args.inbox_dir), Path(args.done_dir), Path(args.error_dir)
    for d in (inbox, done_dir, error_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"[rank_serve_robocasa] ready. watching {inbox} (poll {args.poll_interval}s)")
    try:
        while True:
            jobs = sorted(
                (p for p in inbox.glob("*.json") if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
            )
            for job_path in jobs:
                try:
                    process_job(job_path, processors, models, args.task_name,
                                args.batch_size, args.frame_fraction)
                    archive(job_path, done_dir)
                except Exception as e:
                    logger.error(f"[rank_serve_robocasa] ERROR on {job_path.name}: {e!r}")
                    logger.error(traceback.format_exc())
                    archive(job_path, error_dir)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("[rank_serve_robocasa] shutting down on KeyboardInterrupt")


if __name__ == "__main__":
    main()
