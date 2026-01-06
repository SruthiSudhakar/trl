import os
import shutil
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Delete checkpoints except specified ones.")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="outputs/dec19/sft-side_by_side-regression-Qwen2.5-VL-7B-Instruct_20251219_172145",
        help="Base directory containing checkpoints"
    )
    parser.add_argument(
        "--keep", 
        nargs="+", 
        type=int, 
        required=True, 
        help="List of checkpoint numbers to keep (e.g. 1000 2000)"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="If set, only print what would be deleted without actually deleting."
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    keep_numbers = set(args.keep)

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    print(f"Scanning directory: {base_dir}")
    print(f"Keeping checkpoints: {keep_numbers}")

    # Regex to match checkpoint directories
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                checkpoint_num = int(match.group(1))
                
                if checkpoint_num in keep_numbers:
                    print(f"Keeping: {item}")
                else:
                    if args.dry_run:
                        print(f"[DRY RUN] Would delete: {item}")
                    else:
                        print(f"Deleting: {item}")
                        try:
                            shutil.rmtree(item_path)
                        except Exception as e:
                            print(f"Error deleting {item}: {e}")

if __name__ == "__main__":
    main()
