# scripts/train_vlm_overlay.py
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    # Forward all CLI args to the VLM overlay v2 script (on-the-fly video decoding).
    script_path = Path(__file__).parent.parent / "examples" / "scripts" / "myscripts" / "sft_vlm_overlay_regression_v2.py"
    # Ensure video_frame_utils is importable from the same directory
    utils_dir = str(script_path.parent)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    # run_path executes it as __main__, preserving sys.argv
    sys.argv = [str(script_path)] + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")
