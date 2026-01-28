# scripts/train_vlm_overlay.py
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    # Forward all CLI args to the VLM overlay script.
    # This relies on the script living at examples/scripts/myscripts/...
    script_path = Path(__file__).parent.parent / "examples" / "scripts" / "myscripts" / "sft_vlm_overlay_regression_dp_compare_across_sf.py"
    # run_path executes it as __main__, preserving sys.argv
    sys.argv = [str(script_path)] + sys.argv[1:]
    runpy.run_path(str(script_path), run_name="__main__")
