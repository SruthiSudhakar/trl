"""Pick the best of N candidate images using a trained BagPlate Qwen2.5-VL checkpoint.

Like rank_videos.py, but the inputs are direct image paths (.jpg/.png) — no
frame extraction. Runs the C(N, 2) unordered pairwise comparisons (one ordering
each) through the checkpoint in batches, tallies which image got the "more task
progress" label most often, and prints that image path on stdout. Per-pair
generations and vote tally go to stderr.

Example — evaluate on exactly 2 images:
CUDA_VISIBLE_DEVICES=0 python examples/scripts/myscripts/rank_images.py \
/path/to/img1.jpg /path/to/img2.jpg
"""

import argparse
import json
import logging
import os
import re
import sys
from itertools import combinations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_vlm_bag_plate import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402
from sft_vlm_overlay_regression_v2 import TASK_TOKENS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/proj/vondrick3/sruthi/Appaji/trl/outputs/"
    "BagPlate_20260514_162759/checkpoint-450"
)

SIGNED_INT_RE = re.compile(r"-?\d+")


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


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


def _build_prompt_text(processor, frames, user_text):
    """Render the chat-template text once. Same for every pair (2 image slots + same user text)."""
    mm_template = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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


def _summarize_results(image_paths, task_name, all_results):
    """Tally votes from a list of (i, j, gen_text) tuples and produce the final ranking dict."""
    n = len(image_paths)
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
    for k, v in enumerate(image_paths):
        logger.info(f"  [{k}] votes={votes[k]}  {v}")

    max_votes = max(votes)
    tied = [k for k, c in enumerate(votes) if c == max_votes]
    if len(tied) > 1:
        tied_paths = [image_paths[k] for k in tied]
        logger.warning(f"Tie ({max_votes} votes) between: {tied_paths} — picking first")
    winner_idx = tied[0]
    winner = image_paths[winner_idx]

    return {
        "image_paths": list(image_paths),
        "task_name": task_name,
        "votes": votes,
        "pairs": pairs,
        "winner": winner,
        "winner_idx": winner_idx,
        "tied_indices": tied,
    }


def rank_images(
    image_paths,
    processor,
    model,
    task_name: str = "BagPlate",
    batch_size: int = 4,
) -> dict:
    """Run the C(N, 2) pairwise comparisons across N images and return a structured result dict."""
    n = len(image_paths)
    if n < 2:
        raise ValueError(f"rank_images requires at least 2 paths, got {n}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    task_token = TASK_TOKENS[task_name]
    user_text = USER_PROMPT_TEMPLATE.format(task_token=task_token)

    logger.info(f"Loading {n} images")
    frames = [load_image(p) for p in image_paths]

    text = _build_prompt_text(processor, frames, user_text)
    all_pairs = list(combinations(range(n), 2))

    all_results = []
    for start in range(0, len(all_pairs), batch_size):
        chunk = all_pairs[start:start + batch_size]
        all_results.extend(_run_pair_chunk(processor, model, frames, chunk, text))

    return _summarize_results(image_paths, task_name, all_results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="2 or more image paths (.jpg/.png)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task_name", default="BagPlate")
    p.add_argument("--max_pixels", default="960x540", help="WxH image budget")
    p.add_argument("--batch_size", type=int, default=4, help="Pairwise comparisons per generate() call")
    p.add_argument("--output_json", default=None,
                   help="If set, write the full ranking dict to this JSON path before printing the winner.")
    args = p.parse_args()

    if len(args.images) < 2:
        p.error(f"need at least 2 images, got {len(args.images)}")

    processor, model = load_ranker(args.checkpoint, args.max_pixels)
    result = rank_images(
        args.images, processor, model,
        task_name=args.task_name, batch_size=args.batch_size,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote ranking JSON to {args.output_json}")

    print(result["winner"])


if __name__ == "__main__":
    main()
