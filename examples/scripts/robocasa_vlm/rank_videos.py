"""Pick the best of N candidate videos using a trained Qwen2.5-VL checkpoint.

Extracts the last frame of each video, runs the C(N, 2) unordered pairwise
comparisons (one ordering each) through the checkpoint in batches, tallies
which video's frame got the "more task progress" label most often, and
prints that video path on stdout. Per-pair generations and vote tally go to
stderr.

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
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_frame_utils import _get_video_reader, extract_frame  # noqa: E402

LEGO_SYSTEM_PROMPT = (
    "Compare robot task progress. You will see two images. "
    "Respond with a number: positive if the second image shows more progress, "
    "negative if the first does."
)
LEGO_USER_PROMPT_TEMPLATE = (
    "Task: {task_token}\n"
    "Which image shows more task progress (the first or the second)? "
    "Respond with a number from -100 to 100."
)

TASK_TOKENS = {
    "PnPCounterToCab": "[COUNTER_TO_CAB]",
    "PnPCabToCounter": "[CAB_TO_COUNTER]",
    "PnPCounterToMicrowave": "[COUNTER_TO_MICROWAVE]",
    "PnPMicrowaveToCounter": "[MICROWAVE_TO_COUNTER]",
    "PnPStoveToCounter": "[STOVE_TO_COUNTER]",
    "PnPCounterToStove": "[COUNTER_TO_STOVE]",
    "PnPCounterToSink": "[COUNTER_TO_SINK]",
    "PnPSinkToCounter": "[SINK_TO_COUNTER]",
    "PnPCoffeeServeMug": "[COFFEE_SERVE_MUG]",
    "PnPRedLegoToBrownBowl": "[PNP_RED_LEGO_TO_BROWN_BOWL]",
    "CoffeeServeMug": "[COFFEE_SERVE_MUG]",
    "PnPCloseDrawer": "[CLOSE_DRAWER]",
    "CoffeeSetupMug": "[COFFEE_SETUP_MUG]",
    "CoffeePressButton": "[COFFEE_PRESS_BUTTON]",
    "CloseDrawer": "[CLOSE_DRAWER]",
    "OpenDrawer": "[OPEN_DRAWER]",
    "CloseSingleDoor": "[CLOSE_SINGLE_DOOR]",
    "CloseDoubleDoor": "[CLOSE_DOUBLE_DOOR]",
    "OpenDoubleDoor": "[OPEN_DOUBLE_DOOR]",
    "OpenSingleDoor": "[OPEN_SINGLE_DOOR]",
    "TurnSinkSpout": "[TURN_SINK_SPOUT]",
    "PutKiwiInCenterOfTable": "[PUT_KIWI_IN_CENTER_OF_TABLE]",
    "PushCoasterToMug": "[PUSH_COASTER_TO_MUG]",
    "TurnMugRightsideUp": "[TURN_MUG_RIGHTSIDE_UP]",
    "BimanualBikeRotorInstall": "[BIKE_ROTOR_INSTALL]",
    "BimanualClearKitchenCounter": "[CLEAR_KITCHEN_COUNTER]",
    "BimanualSetUpBreakfastTable": "[SETUP_BREAKFAST_TABLE]",
    "CleanLitterBox": "[CLEAN_LITTERBOX]",
    "CutAppleIntoSlices": "[CUT_APPLE]",
    "UprightBottle": "[UPRIGHT_BOTTLE]",
    "BagPlate": "[BAG_PLATE]",
    "PushBowl": "[PUSH_BOWL]",
    "Stacking": "[STACKING]",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/proj/vondrick3/sruthi/Appaji/trl/outputs/"
    "PushBowl_20260515_230044/checkpoint-1050"
)

SIGNED_INT_RE = re.compile(r"-?\d+")


def get_frame_at_fraction(path: str, fraction: float = 1.0):
    """Return the frame at floor(last_idx * fraction) of an .mp4 or frame directory.

    fraction=1.0 returns the last frame; fraction=0.5 returns the middle frame.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    if path.endswith(".mp4"):
        vr = _get_video_reader(path)
        n = len(vr)
        if n == 0:
            raise ValueError(f"Video has no frames: {path}")
        idx = int((n - 1) * fraction)
        return extract_frame(path, idx)

    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "frame_*.png"))
            + glob.glob(os.path.join(path, "frame_*.jpg"))
        )
        if not files:
            raise ValueError(f"No frame_*.png/jpg files in directory: {path}")
        target = os.path.basename(files[int((len(files) - 1) * fraction)])
        m = re.search(r"frame_(\d+)\.(?:png|jpg)$", target)
        if not m:
            raise ValueError(f"Could not parse frame index from {target}")
        return extract_frame(path, int(m.group(1)))

    raise FileNotFoundError(f"Not a .mp4 file or frame directory: {path}")


def load_ranker(checkpoint: str, max_pixels_wh: str = "960x540"):
    """Load (processor, model) for the ranker. Pinned to cuda:0 with bf16 + flash-attn-2."""
    w, h = (int(x) for x in max_pixels_wh.split("x"))
    max_pixels = w * h

    logger.info(f"Loading model from {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint, max_pixels=max_pixels)
    # Left-pad so out[:, input_ids.shape[1]:] cleanly extracts new tokens when batching.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()
    return processor, model


def load_ranker_multi(checkpoint: str, max_pixels_wh: str = "960x540", gpu_ids=(0,)):
    """Load one (processor, model) pair per GPU. Returns (processors, models) lists."""
    w, h = (int(x) for x in max_pixels_wh.split("x"))
    max_pixels = w * h

    processors, models = [], []
    for k, gid in enumerate(gpu_ids):
        logger.info(f"Loading model from {checkpoint} onto cuda:{gid} ({k+1}/{len(gpu_ids)})")
        processor = AutoProcessor.from_pretrained(checkpoint, max_pixels=max_pixels)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gid}",
        )
        model.eval()
        processors.append(processor)
        models.append(model)
    return processors, models


def _build_prompt_text(processor, frames, user_text):
    """Render the chat-template text once. Same for every pair (2 image slots + same user text)."""
    mm_template = [
        {"role": "system", "content": [{"type": "text", "text": LEGO_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frames[0]},
                {"type": "image", "image": frames[0]},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return processor.apply_chat_template(mm_template, tokenize=False, add_generation_prompt=True)


def _run_pair_chunk(processor, model, frames, chunk, text):
    """Run generate() on one chunk of pair indices. Returns [(i, j, gen_text), ...]."""
    image_lists = [[frames[i], frames[j]] for i, j in chunk]
    texts = [text] * len(chunk)
    inputs = processor(text=texts, images=image_lists, return_tensors="pt", padding=True)
    inputs = {kk: v.to(model.device) if hasattr(v, "to") else v for kk, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gens = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return [(i, j, g.strip()) for (i, j), g in zip(chunk, gens)]


def _summarize_results(video_paths, task_name, all_results):
    """Tally votes from a list of (i, j, gen_text) tuples and produce the final ranking dict."""
    n = len(video_paths)
    votes = [0] * n
    pairs = []
    all_results.sort(key=lambda r: (r[0], r[1]))
    for i, j, gen in all_results:
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
        pairs.append({"i": i, "j": j, "gen": gen, "pred": pred, "winner": winner_idx})

    logger.info("Vote tally:")
    for k, v in enumerate(video_paths):
        logger.info(f"  [{k}] votes={votes[k]}  {v}")

    max_votes = max(votes)
    tied = [k for k, c in enumerate(votes) if c == max_votes]
    if len(tied) > 1:
        tied_paths = [video_paths[k] for k in tied]
        logger.warning(f"Tie ({max_votes} votes) between: {tied_paths} — picking first")
    winner_idx = tied[0]
    winner = video_paths[winner_idx]

    return {
        "video_paths": list(video_paths),
        "task_name": task_name,
        "votes": votes,
        "pairs": pairs,
        "winner": winner,
        "winner_idx": winner_idx,
        "tied_indices": tied,
    }


def rank_videos(
    video_paths,
    processor,
    model,
    task_name: str = "PnPRedLegoToBrownBowl",
    batch_size: int = 4,
    frame_fraction: float = 1.0,
) -> dict:
    """Run the C(N, 2) pairwise comparisons across N videos and return a structured result dict.

    Comparisons are run in mini-batches of size `batch_size` for throughput.
    `frame_fraction` selects which frame of each video to compare (1.0=last, 0.5=middle).
    """
    n = len(video_paths)
    if n < 2:
        raise ValueError(f"rank_videos requires at least 2 paths, got {n}")
    for v in video_paths:
        if not (os.path.isfile(v) or os.path.isdir(v)):
            raise FileNotFoundError(f"Video path does not exist: {v}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    task_token = TASK_TOKENS[task_name]
    user_text = LEGO_USER_PROMPT_TEMPLATE.format(task_token=task_token)

    logger.info(f"Extracting frame at fraction={frame_fraction} from each of the {n} videos")
    frames = [get_frame_at_fraction(v, frame_fraction) for v in video_paths]

    text = _build_prompt_text(processor, frames, user_text)
    all_pairs = list(combinations(range(n), 2))

    all_results = []
    for start in range(0, len(all_pairs), batch_size):
        chunk = all_pairs[start:start + batch_size]
        all_results.extend(_run_pair_chunk(processor, model, frames, chunk, text))

    return _summarize_results(video_paths, task_name, all_results)


def rank_videos_multi(
    video_paths,
    processors,
    models,
    task_name: str = "PnPRedLegoToBrownBowl",
    batch_size: int = 8,
    frame_fraction: float = 1.0,
) -> dict:
    """Multi-GPU version of `rank_videos`.

    `processors` and `models` are parallel lists of length G (one per GPU). The
    C(N, 2) pairs are round-robin assigned across the G GPUs and each GPU runs
    its share in mini-batches of `batch_size` via a ThreadPoolExecutor (threads
    release the GIL during CUDA kernels, so they overlap on different devices).
    `frame_fraction` selects which frame of each video to compare (1.0=last, 0.5=middle).
    """
    n = len(video_paths)
    if n < 2:
        raise ValueError(f"rank_videos_multi requires at least 2 paths, got {n}")
    for v in video_paths:
        if not (os.path.isfile(v) or os.path.isdir(v)):
            raise FileNotFoundError(f"Video path does not exist: {v}")
    g = len(models)
    if g < 1 or g != len(processors):
        raise ValueError(f"need >=1 (processor, model) pairs of equal length, got {len(processors)}/{g}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    task_token = TASK_TOKENS[task_name]
    user_text = LEGO_USER_PROMPT_TEMPLATE.format(task_token=task_token)

    logger.info(f"Extracting frame at fraction={frame_fraction} from each of the {n} videos")
    frames = [get_frame_at_fraction(v, frame_fraction) for v in video_paths]

    text = _build_prompt_text(processors[0], frames, user_text)

    all_pairs = list(combinations(range(n), 2))
    per_gpu = [[] for _ in range(g)]
    for idx, p in enumerate(all_pairs):
        per_gpu[idx % g].append(p)

    def _run(gpu_idx):
        processor, model = processors[gpu_idx], models[gpu_idx]
        sub = per_gpu[gpu_idx]
        out = []
        for start in range(0, len(sub), batch_size):
            chunk = sub[start:start + batch_size]
            out.extend(_run_pair_chunk(processor, model, frames, chunk, text))
        return out

    all_results = []
    with ThreadPoolExecutor(max_workers=g) as ex:
        futures = [ex.submit(_run, k) for k in range(g)]
        for fut in futures:
            all_results.extend(fut.result())

    return _summarize_results(video_paths, task_name, all_results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+", help="2 or more video paths (.mp4 or frame dirs)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task_name", default="PnPRedLegoToBrownBowl")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--batch_size", type=int, default=4, help="Pairwise comparisons per generate() call")
    p.add_argument("--frame_fraction", type=float, default=1.0,
                   help="Which frame to compare; fraction of the last index (1.0=last, 0.5=middle)")
    p.add_argument("--output_json", default=None,
                   help="If set, write the full ranking dict to this JSON path before printing the winner.")
    args = p.parse_args()

    if len(args.videos) < 2:
        p.error(f"need at least 2 videos, got {len(args.videos)}")

    processor, model = load_ranker(args.checkpoint, args.max_pixels)
    result = rank_videos(
        args.videos, processor, model,
        task_name=args.task_name, batch_size=args.batch_size,
        frame_fraction=args.frame_fraction,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote ranking JSON to {args.output_json}")

    print(result["winner"])


if __name__ == "__main__":
    main()
