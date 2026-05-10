"""Long-running ranker server. Pairs with HunyuanVideo-1.5-train-sruthi/serve.py.

Watches an inbox of small JSON job files. Each job names 2 or more video paths
plus the output subdir where a `ranking.json` should land. Loads the
Qwen2.5-VL ranker once at startup and reuses it across jobs. The C(N, 2)
pairwise comparisons within a job are run in mini-batches (--batch_size).

Job file format (written atomically by the producer — tmp → rename):
    {
      "name": "<manifest_stem>",
      "output_subdir": "/abs/.../<name>",
      "video_paths": ["/abs/.../0.mp4", "/abs/.../1.mp4", ...],
      "task_name": "PnPRedLegoToBrownBowl"
    }

Usage:
conda activate vlmoverlay
cd /proj/vondrick3/sruthi/Appaji/trl
CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_serve.py \
--inbox_dir /proj/vondrick3/HunyuanVideo-1.5-train-sruthi/rank_inbox \
--done_dir  /proj/vondrick3/HunyuanVideo-1.5-train-sruthi/rank_done \
--error_dir /proj/vondrick3/HunyuanVideo-1.5-train-sruthi/rank_errors
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_videos import DEFAULT_CHECKPOINT, load_ranker, rank_videos  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def process_job(job_path: Path, processor, model, default_task_name: str, batch_size: int):
    """Read a job json, run rank_videos, write ranking.json next to the videos."""
    with open(job_path, "r") as f:
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

    logger.info(f"[rank_serve] processing {name} ({len(video_paths)} videos): {video_paths}")
    result = rank_videos(
        video_paths, processor, model, task_name=task_name, batch_size=batch_size,
    )

    out_path = output_subdir / "ranking.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[rank_serve] wrote {out_path} (winner={result['winner']})")


def archive(job_path: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / job_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(job_path), str(target))


def serve(args):
    processor, model = load_ranker(args.checkpoint, args.max_pixels)

    inbox = Path(args.inbox_dir)
    done_dir = Path(args.done_dir)
    error_dir = Path(args.error_dir)
    for d in (inbox, done_dir, error_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"[rank_serve] ready. watching {inbox} (poll {args.poll_interval}s)")

    try:
        while True:
            jobs = sorted(
                (p for p in inbox.glob("*.json") if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
            )
            for job_path in jobs:
                try:
                    process_job(job_path, processor, model, args.task_name, args.batch_size)
                    archive(job_path, done_dir)
                except Exception as e:
                    logger.error(f"[rank_serve] ERROR on {job_path.name}: {e!r}")
                    logger.error(traceback.format_exc())
                    archive(job_path, error_dir)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("[rank_serve] shutting down on KeyboardInterrupt")


def main():
    p = argparse.ArgumentParser(description="Long-running 4-video ranker server.")
    p.add_argument("--inbox_dir", required=True)
    p.add_argument("--done_dir", required=True)
    p.add_argument("--error_dir", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task_name", default="PnPRedLegoToBrownBowl")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--batch_size", type=int, default=4, help="Pairwise comparisons per generate() call")
    p.add_argument("--poll_interval", type=float, default=1.0)
    args = p.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
