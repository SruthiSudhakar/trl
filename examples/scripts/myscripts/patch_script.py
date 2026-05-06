import re

with open("/home/sruthi.sudhakar/hf_trl/trl/examples/scripts/myscripts/evaluate_progressLM.py", "r") as f:
    content = f.read()

# Replace imports
old_imports = """from sft_vlm_overlay_regression_v2 import (
    TASK_TOKENS,
    load_trajectories,
    match_failures_to_successes,
    balance_by_demo_id,
    build_frame_pairs,
    compute_failure_filter_stats,
)"""

new_imports = """from sft_vlm_overlay_regression_realworld_data import (
    TASK_TOKENS,
    is_realworld_format,
    load_realworld_trajectories,
    match_realworld_failures_to_successes,
    compute_realworld_failure_filter_stats,
    build_realworld_frame_pairs,
)"""

content = content.replace(old_imports, new_imports)

# Replace data loading block
# From `print("Loading dataset...")` 
#   until `# Load visual demo frames from first success trajectory`

data_loading_regex = re.compile(r'print\("Loading dataset\.\.\."\).*?# Load visual demo frames from first success trajectory', re.DOTALL)

new_data_loading = """print(f"Loading dataset from {base_dataset_path}...")
        if not is_realworld_format(base_dataset_path):
            print(f"WARNING: {base_dataset_path} does not appear to be a realworld dataset format. Skipping.")
            continue
            
        trajectories = load_realworld_trajectories(base_dataset_path)
        print(f"Found {len(trajectories)} total trajectories")

        if not trajectories:
            continue

        success_trajs = [t for t in trajectories if t["sf"] == "success"]
        failure_trajs = [t for t in trajectories if t["sf"] == "fail"]
        print(f"  {len(success_trajs)} success + {len(failure_trajs)} failure trajectories")

        matched_failures = match_realworld_failures_to_successes(trajectories)
        print(f"  Matched {len(matched_failures)} failures to successes")

        if matched_failures:
            success_mean_diffs_at_idx = compute_realworld_failure_filter_stats(
                trajectories, local_rank=0, world_size=1
            )
        else:
            success_mean_diffs_at_idx = {}

        task_name = Path(base_dataset_path).name
        all_task_pairs = build_realworld_frame_pairs(
            success_data=success_trajs,
            failure_data=matched_failures,
            compare_intervals=compare_intervals,
            train_sample_interval=args.sample_interval,
            success_mean_diffs_at_idx=success_mean_diffs_at_idx,
            task_name=task_name,
            local_rank=0,
            world_size=1,
        )
        print(f"Built {len(all_task_pairs)} evaluation pairs for {task_name}")

        all_demo_ids = sorted(set(item["demo_id"] for item in all_task_pairs))
        rng = random.Random(42)
        rng.shuffle(all_demo_ids)
        n_eval_demos = max(1, int(len(all_demo_ids) * 0.1))
        eval_demo_ids = set(all_demo_ids[:n_eval_demos])
        train_demo_ids = set(all_demo_ids[n_eval_demos:])
        
        if args.split == "val":
            keep_ids = eval_demo_ids
        elif args.split == "train":
            keep_ids = train_demo_ids
        else:
            keep_ids = set(all_demo_ids)

        pairs = [p for p in all_task_pairs if p["demo_id"] in keep_ids]
        print(f"Using {len(pairs)} pairs for '{args.split}' split")

        if args.num_samples and args.num_samples < len(pairs):
            subsample_rng = random.Random(args.seed)
            pairs = subsample_rng.sample(pairs, args.num_samples)
            print(f"Subsampled to {len(pairs)} pairs")

        # Load visual demo frames from first success trajectory"""

content = data_loading_regex.sub(new_data_loading, content)

visual_demo_regex = re.compile(r'# Load visual demo frames from first success trajectory.*?visual_demo_paths = \[\]\n\s*print\("Warning: No success data available for visual demo"\)', re.DOTALL)

new_visual_demo = """# Load visual demo frames from first success trajectory in the split
        num_demo_frames = 5
        valid_success_trajs = [t for t in success_trajs if t["episode_id"] in keep_ids]
        if not valid_success_trajs:
            valid_success_trajs = success_trajs
        
        if valid_success_trajs:
            demo_traj = valid_success_trajs[0]
            demo_frames_dir = demo_traj["frames_dir"]
            demo_last_frame = demo_traj["num_frames"]
            
            frame_indices = [1 + int(i * (demo_last_frame - 1) / (num_demo_frames - 1)) for i in range(num_demo_frames)]
            import tempfile
            import os
            demo_dir = tempfile.mkdtemp(prefix="visual_demo_")
            visual_demo_paths = []
            for idx, fi in enumerate(frame_indices):
                frame = extract_frame(demo_frames_dir, fi).convert("RGB")
                path = os.path.join(demo_dir, f"demo_{idx:03d}.png")
                frame.save(path)
                visual_demo_paths.append(path)
            print(f"Loaded {len(visual_demo_paths)} visual demo frames from {demo_frames_dir} (frames: {frame_indices})")
        else:
            visual_demo_paths = []
            print("Warning: No success data available for visual demo")"""

content = visual_demo_regex.sub(new_visual_demo, content)

with open("/home/sruthi.sudhakar/hf_trl/trl/examples/scripts/myscripts/evaluate_progressLM_realworld.py", "w") as f:
    f.write(content)

print("Done generating evaluate_progressLM_realworld.py")
