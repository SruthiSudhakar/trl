# RoboCasa VLM Ranker — Training and Serving

This directory holds the training code, ranker server, and launch script used by
the VLM-ranking evaluation mode in
[`robocasa-benchmark/Isaac-GR00T`](https://github.com/robocasa-benchmark/Isaac-GR00T).
See that repo's README for the end-to-end eval pipeline; this one is about how
to (1) train the Qwen2.5-VL ranker on RoboCasa policy rollouts and (2) serve it
as an inbox/poll ranker that GR00T's `run_eval_vlm_ranking.py` talks to.

## Contents

| File | Purpose |
| --- | --- |
| `launch_robocasa.sh` | 8×A6000 DeepSpeed ZeRO-2 SFT launcher. Edit the `TASKS` list and `PRETRAIN_ROOT` at the top, then run. |
| `sft_vlm_robocasa.py` | SFT trainer. Builds pairwise progress-comparison pairs from RoboCasa rollouts (success-vs-success at different sub-frames; last K frames of a failure vs. the aligned success at the same initial condition). |
| `rank_serve_robocasa.py` | Long-running inbox/poll ranker. Loads a trained checkpoint once, then processes ranking jobs written to `inbox_dir/`. |
| `rank_videos.py` | Ranking engine (`load_ranker_multi`, `rank_videos_multi`) reused by the server. |
| `video_frame_utils.py` | Decord-based frame extraction and rollout-directory discovery. |

## Environment

Create a separate conda env from the one GR00T lives in — the ranker needs a
newer `transformers` than GR00T pins:

```bash
conda create -n vlmoverlay python=3.10 -y
conda activate vlmoverlay
pip install -e .                                     # from the trl repo root
pip install flash-attn --no-build-isolation
pip install decord wandb accelerate deepspeed
```

You also need the Qwen2.5-VL-3B-Instruct base weights on disk (or a HF cache):

```bash
export QWEN_BASE=/path/to/Qwen2.5-VL-3B-Instruct
```

## Data layout expected by the trainer

`sft_vlm_robocasa.py` walks the rollout tree produced by GR00T's basic eval
(`run_eval.py` — see the Isaac-GR00T README). One subdir per rollout, each
containing an `eval_log.json` and a `media/` directory:

```
$PRETRAIN_ROOT/
  <TASK>_<...>_seed<seed>/
    eval_log.json                 # per-episode success flags + eval_args.task
    media/seed1000XX.mp4          # one mp4 per episode
    recovered_lang.json           # optional; per-video language description
```

The trainer holds out the last `--num_eval_episodes` initial conditions (that
have at least one success) for eval and trains on the rest.

## Train

```bash
conda activate vlmoverlay
bash examples/scripts/robocasa_vlm/launch_robocasa.sh
```

To retarget to a different set of tasks, either edit the `TASKS` array at the
top of the launch script or override at invocation:

```bash
TASK=CloseFridge bash examples/scripts/robocasa_vlm/launch_robocasa.sh
```

Checkpoints land in `outputs/multitask_<stamp>_robocasa/checkpoint-<step>/`.

## Serve

The ranker is a long-running process that watches a directory for job files
and writes `ranking.json` next to the videos it ranked. This is what
`run_eval_vlm_ranking.py` (in Isaac-GR00T) talks to.

```bash
conda activate vlmoverlay
CUDA_VISIBLE_DEVICES=0 python examples/scripts/robocasa_vlm/rank_serve_robocasa.py \
    --inbox_dir  /shared/rank_dir/inbox \
    --done_dir   /shared/rank_dir/done \
    --error_dir  /shared/rank_dir/error \
    --checkpoint /path/to/outputs/multitask_<stamp>_robocasa/checkpoint-<step> \
    --task_name  PickPlaceDrawerToCounter \
    --gpu_ids 0 --batch_size 10 --max_pixels 960x540
```

Job file format (producer writes atomically, tmp → rename):

```json
{
  "name": "<stem>",
  "output_subdir": "/abs/path/where/ranking.json/should/go",
  "video_paths": ["/abs/.../0.mp4", "/abs/.../1.mp4", "..."],
  "task_name": "CloseToasterOvenDoor"
}
```

Task tokens are resolved the same way at train and serve time: the curated
`TASK_TOKENS` mapping if the task is present, otherwise an auto-derived
`UPPER_SNAKE` bracketed token
(`CloseToasterOvenDoor` → `[CLOSE_TOASTER_OVEN_DOOR]`).

## Wiring into the Isaac-GR00T eval

Point GR00T's eval at this ranker script:

```bash
export VLM_RANK_SERVER=/path/to/trl/examples/scripts/robocasa_vlm/rank_serve_robocasa.py
export VLM_RANK_SERVER_PYTHON=$(conda run -n vlmoverlay which python)
```

Then follow the "Evaluation with VLM ranking" section of the Isaac-GR00T
README.
