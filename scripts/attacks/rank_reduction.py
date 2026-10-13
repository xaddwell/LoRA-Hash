#!/usr/bin/env python3
"""Rank reduction of LoRA files by shelling out to kohya_ss ``resize_lora.py``.

The original rank is detected from the LoRA tensors (using the largest rank for
SDXL/FLUX files and the first rank otherwise), then one resized copy is produced
per requested factor, named ``<base>_rank_<new_rank>.safetensors``. Accepts a
single ``.safetensors`` file or a directory of them.

Usage:
    python rank_reduction.py --input data/benign --output data/rank_reduced \\
        --kohya-script /path/to/kohya_ss/library/resize_lora.py
"""

import os
import subprocess
import sys
import argparse
from safetensors.torch import load_file
from tqdm import tqdm

def get_lora_rank(lora_path: str, lora_filename: str) -> int:

    try:
        tensors = load_file(lora_path)
        # Look up every possible rank-defining key (supports SD 'lora_down' and FLUX 'lora_A')
        all_rank_keys = [key for key in tensors.keys()
                         if key.endswith(('lora_down.weight', 'lora_A.weight'))]

        if not all_rank_keys:
            print(f"Warning: no 'lora_down.weight' or 'lora_A.weight' found in "
                  f"{lora_filename}; cannot determine the rank.")
            return -1

        filename_lower = lora_filename.lower()

        # For XL or FLUX models, use the largest rank across all LoRA layers as the baseline
        if "xl" in filename_lower or "flux" in filename_lower:
            # Compute and return the maximum rank found
            max_rank = max(tensors[key].shape[0] for key in all_rank_keys)
            return max_rank
        else:
            # Original logic: for SD1.5 and similar models, the first rank found is enough
            first_rank_tensor = tensors[all_rank_keys[0]]
            return first_rank_tensor.shape[0]

    except Exception as e:
        print(f"Error: failed to read file {lora_path}: {e}")
        return -1

def batch_resize_loras(input_path: str, output_dir: str, kohya_script_path: str, factors: list):
    """
    Reduce the rank of a directory of LoRA files or of a single LoRA file.
    """
    if not os.path.isfile(kohya_script_path):
        print(f"Error: kohya_ss resize_lora.py script not found at: {kohya_script_path}")
        print("Please provide the correct path with the --kohya-script argument.")
        return

    # --- Determine whether the input is a file or a directory ---
    lora_paths_to_process = []
    if os.path.isdir(input_path):
        print(f"Input is a directory, scanning '{input_path}' for .safetensors files...")
        filenames = [f for f in os.listdir(input_path) if f.endswith('.safetensors')]
        if not filenames:
            print(f"No .safetensors files found in directory '{input_path}'.")
            return
        lora_paths_to_process = [os.path.join(input_path, f) for f in filenames]
    elif os.path.isfile(input_path):
        if input_path.endswith('.safetensors'):
            print(f"Input is a single file: '{input_path}'")
            lora_paths_to_process = [input_path]
        else:
            print(f"Error: input file '{input_path}' is not a .safetensors file.")
            return
    else:
        print(f"Error: input path '{input_path}' is not a valid file or directory.")
        return
    # --- End of the input checks ---

    print(f"Found {len(lora_paths_to_process)} LoRA files; generating up to "
          f"{len(factors)} rank-reduced versions for each.")
    os.makedirs(output_dir, exist_ok=True)

    total_tasks = len(lora_paths_to_process) * len(factors)
    with tqdm(total=total_tasks, desc="Overall progress") as pbar:
        # The loop variable is named lora_full_path to avoid clashing with input_path
        for lora_full_path in lora_paths_to_process:
            lora_filename = os.path.basename(lora_full_path)

            original_rank = get_lora_rank(lora_full_path, lora_filename)

            if original_rank <= 0:
                pbar.update(len(factors))
                continue

            pbar.set_description(f"Processing: {lora_filename[:20]}...(Rank:{original_rank})")

            processed_ranks = set()

            for factor in sorted(factors, reverse=True):
                new_rank = int(original_rank * factor)
                if new_rank < 1: new_rank = 1

                if new_rank >= original_rank or new_rank in processed_ranks:
                    pbar.update(1)
                    continue

                processed_ranks.add(new_rank)
                base_name = os.path.splitext(lora_filename)[0]
                output_filename = f"{base_name}_rank_{new_rank}.safetensors"
                output_path = os.path.join(output_dir, output_filename)

                absolute_input_path = os.path.abspath(lora_full_path)
                absolute_output_path = os.path.abspath(output_path)

                command = [
                    sys.executable, kohya_script_path,
                    "--model", absolute_input_path,
                    "--save_to", absolute_output_path,
                    "--new_rank", str(new_rank),
                    "--device", "cuda",
                    "--save_precision", "fp16"
                ]

                try:
                    kohya_base_dir = os.path.dirname(os.path.dirname(kohya_script_path))
                    subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=kohya_base_dir
                    )
                except subprocess.CalledProcessError as e:
                    print(f"\nError: resize_lora.py failed (file: {lora_filename}, "
                          f"new rank: {new_rank}).")
                    print(f"Details: {e.stderr}")

                pbar.update(1)

    print("\n--- Batch rank reduction complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptively reduce the rank of LoRA files (supports SD1.5, SDXL, FLUX).")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing the original LoRA files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory where the rank-reduced LoRA files are saved.")
    parser.add_argument("--kohya-script", type=str, required=True,
                        help="Full path to the kohya_ss/library/resize_lora.py script.")
    parser.add_argument("--factors", nargs='+', type=float, default=[0.5, 0.25],
                        help="List of rank-reduction factors (e.g. 0.75 0.5 0.25).")

    args = parser.parse_args()

    # Recommended before using tqdm: pip install tqdm
    # The tqdm fallback is kept so the script also runs where tqdm is missing
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm is not installed, so no progress bar will be shown. "
              "Consider running 'pip install tqdm'")
        class TqdmMock:
            def __init__(self, total, desc): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def update(self, n=1): pass
            def set_description(self, desc): pass
        tqdm = TqdmMock

    batch_resize_loras(args.input, args.output, args.kohya_script, args.factors)
