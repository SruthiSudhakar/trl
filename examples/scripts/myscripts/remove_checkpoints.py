import os
import shutil
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Remove checkpoints except specified ones.")
    parser.add_argument("--dir", required=True, help=f"Directory containing checkpoints.")
    parser.add_argument("keep", nargs='*', help="List of checkpoint numbers or names to keep (e.g. 20000 or checkpoint-20000)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()
    
    # allow user to pass relative path
    target_dir = os.path.abspath(args.dir)
    
    if not os.path.exists(target_dir):
        # Try relative to the script location or current working directory?
        # Let's just rely on CWD or absolute path
        print(f"Error: Directory {target_dir} does not exist.")
        sys.exit(1)
        
    print(f"Target directory: {target_dir}")
    
    # Normalize keep list to strings
    keep_list = set(str(k) for k in args.keep)
    
    # helper to check if we should keep
    def should_keep(name):
        if name in keep_list:
            return True
        # also handle number strings if user passed numbers
        # e.g. user passed 2000, dir is checkpoint-2000
        if name.startswith("checkpoint-"):
             num_part = name.split("-")[-1]
             if num_part in keep_list:
                 return True
        return False

    checkpoints = []
    # List all directories starting with 'checkpoint-'
    try:
        items = os.listdir(target_dir)
    except OSError as e:
        print(f"Error accessing directory: {e}")
        sys.exit(1)

    for item in items:
        full_path = os.path.join(target_dir, item)
        if item.startswith("checkpoint-") and os.path.isdir(full_path):
            checkpoints.append(item)
            
    checkpoints.sort() # Sort for display
    
    to_delete = []
    to_keep = []
    
    for cp in checkpoints:
        if should_keep(cp):
            to_keep.append(cp)
        else:
            to_delete.append(cp)
            
    print(f"\nTotal checkpoints found: {len(checkpoints)}")
    print(f"Checkpoints to KEEP ({len(to_keep)}): {', '.join(to_keep)}")
    print(f"Checkpoints to DELETE ({len(to_delete)}): {', '.join(to_delete)}")
    
    if not to_delete:
        print("\nNothing to delete.")
        return

    if args.dry_run:
        print("\nDry run completed. No files were deleted.")
        return

    if not args.yes:
        confirm = input("\nAre you sure you want to delete these checkpoints? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
    print("\nDeleting...")
    deleted_count = 0
    for cp in to_delete:
        full_path = os.path.join(target_dir, cp)
        try:
            shutil.rmtree(full_path)
            # print(f"Deleted: {cp}")
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {cp}: {e}")

    print(f"\nDone. Deleted {deleted_count} checkpoints.")

if __name__ == "__main__":
    main()
