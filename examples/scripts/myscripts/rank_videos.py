"""Pick the best of 4 candidate videos using a trained Qwen2.5-VL checkpoint.

Extracts the last frame of each video, runs the 6 unordered pairwise
comparisons (one ordering each) through the checkpoint, tallies which video's
frame got the "more task progress" label most often, and prints that video
path on stdout. Per-pair generations and vote tally go to stderr.

Example:
    CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_videos.py \\
        v1.mp4 v2.mp4 v3.mp4 v4.mp4

Test 1 — discriminative test (1 failure vs 3 successes). The failure clip should get 0 votes; one of the successes     
wins:                                                                                                                  
  cd /proj/vondrick3/sruthi/Appaji/trl                                                                                   
  CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_videos.py \                                              
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/failure_18_20260504-225619_frames \                            
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_18_20260504-225458_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_19_20260504-225707_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_20_20260504-225931_frames
                                                             
  Test 2 — 4 successes (subtle differences, just a smoke test). Stdout should be one of the 4 paths; stderr shows
  per-pair generations:                                      
  CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_videos.py \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_18_20260504-225458_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_19_20260504-225707_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_20_20260504-225931_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_318_20260508-213508_frames
                                                             
  Test 3 — capture-the-winner pattern (mimics the upstream tmux call). Verifies stdout is exactly one path and stderr is
  everything else:                                           
  winner=$(CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_videos.py \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/failure_18_20260504-225619_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_18_20260504-225458_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_19_20260504-225707_frames \
    /proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl/success_20_20260504-225931_frames)
  echo "WINNER=$winner"

"""

import argparse
import glob
import logging
import os
import re
import sys
from itertools import combinations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_lego import LEGO_SYSTEM_PROMPT, LEGO_USER_PROMPT_TEMPLATE  # noqa: E402
from sft_vlm_overlay_regression_v2 import TASK_TOKENS  # noqa: E402
from video_frame_utils import _get_video_reader, extract_frame  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/proj/vondrick3/sruthi/Appaji/trl/outputs/"
    "PnPRedLegoToBrownBowl_20260508_165458_960x540/checkpoint-200"
)

SIGNED_INT_RE = re.compile(r"-?\d+")


def get_last_frame(path: str):
    """Return the last frame of an .mp4 file or a frame directory as a PIL Image."""
    if path.endswith(".mp4"):
        vr = _get_video_reader(path)
        last_idx = len(vr) - 1
        if last_idx < 0:
            raise ValueError(f"Video has no frames: {path}")
        return extract_frame(path, last_idx)

    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "frame_*.png"))
            + glob.glob(os.path.join(path, "frame_*.jpg"))
        )
        if not files:
            raise ValueError(f"No frame_*.png/jpg files in directory: {path}")
        last_name = os.path.basename(files[-1])
        m = re.search(r"frame_(\d+)\.(?:png|jpg)$", last_name)
        if not m:
            raise ValueError(f"Could not parse frame index from {last_name}")
        return extract_frame(path, int(m.group(1)))

    raise FileNotFoundError(f"Not a .mp4 file or frame directory: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs=4, help="Exactly 4 video paths (.mp4 or frame dirs)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task_name", default="PnPRedLegoToBrownBowl")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    args = p.parse_args()

    for v in args.videos:
        if not (os.path.isfile(v) or os.path.isdir(v)):
            raise FileNotFoundError(f"Video path does not exist: {v}")

    task_token = TASK_TOKENS[args.task_name]
    user_text = LEGO_USER_PROMPT_TEMPLATE.format(task_token=task_token)

    logger.info("Extracting last frame from each of the 4 videos")
    frames = [get_last_frame(v) for v in args.videos]

    w, h = (int(x) for x in args.max_pixels.split("x"))
    max_pixels = w * h

    logger.info(f"Loading model from {args.checkpoint}")
    processor = AutoProcessor.from_pretrained(args.checkpoint, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()

    votes = [0, 0, 0, 0]
    for i, j in combinations(range(4), 2):
        f1, f2 = frames[i], frames[j]
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
        pred = int(m.group(0)) if m else None

        if pred is None:
            logger.warning(f"  pair ({i},{j}) unparsed gen={gen!r} — no vote")
            winner_idx = None
        elif pred > 0:
            votes[j] += 1
            winner_idx = j
        elif pred < 0:
            votes[i] += 1
            winner_idx = i
        else:
            logger.warning(f"  pair ({i},{j}) pred=0 — no vote")
            winner_idx = None

        logger.info(
            f"  pair ({i},{j}): gen={gen!r} pred={pred} "
            f"winner={winner_idx if winner_idx is not None else 'none'}"
        )

    logger.info("Vote tally:")
    for k, v in enumerate(args.videos):
        logger.info(f"  [{k}] votes={votes[k]}  {v}")

    max_votes = max(votes)
    tied = [k for k, c in enumerate(votes) if c == max_votes]
    if len(tied) > 1:
        tied_paths = [args.videos[k] for k in tied]
        logger.warning(f"Tie ({max_votes} votes) between: {tied_paths} — picking first")
    winner = args.videos[tied[0]]

    print(winner)


if __name__ == "__main__":
    main()
