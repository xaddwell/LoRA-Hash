#!/usr/bin/env python3
"""Sign flip of LoRA ``lora_down.weight`` / ``lora_up.weight`` pairs.

Negating both halves of a LoRA pair leaves the product ``lora_up @ lora_down``
unchanged, so this produces a weight-space variant of the same adapter. Accepts
a single ``.safetensors`` file or a directory of them and writes each flipped
file as ``<base>_flipped.safetensors``.

Usage:
    python sign_flip.py --input data/benign --output-dir data/flip
"""

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import List

def flip_lora_signs(lora_path: str, output_path: str, verbose: bool = True) -> bool:
    try:
        if not os.path.exists(lora_path):
            if verbose:
                print(f"Error: input file does not exist at '{lora_path}'")
            return False

        if verbose:
            print(f"\n--- Processing: {os.path.basename(lora_path)} ---")

        state_dict = load_file(lora_path, device="cpu")
        flipped_state_dict = state_dict.copy()

        down_keys = [key for key in flipped_state_dict.keys() if key.endswith('lora_down.weight')]

        if not down_keys:
            if verbose:
                print(f"Warning: no 'lora_down.weight' key found in file "
                      f"{os.path.basename(lora_path)}.")
            return False

        flipped_pairs_count = 0
        for down_key in down_keys:
            up_key = down_key.replace('lora_down.weight', 'lora_up.weight')
            if up_key in flipped_state_dict:
                flipped_state_dict[down_key] *= -1
                flipped_state_dict[up_key] *= -1
                flipped_pairs_count += 1

        if flipped_pairs_count == 0:
            if verbose:
                print(f"Warning: no flippable LoRA pair found in file "
                      f"{os.path.basename(lora_path)}.")
            return False

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        save_file(flipped_state_dict, output_path)
        if verbose:
            print(f"Successfully saved the flipped LoRA to: {output_path}")
        return True

    except Exception as e:
        if verbose:
            print(f"Error while processing file {os.path.basename(lora_path)}: {e}")
        return False

def process_path(input_path: str, output_dir: str):
    """
    Process the given input path, detecting whether it is a file or a directory,
    and run the corresponding flip operation.
    """
    if not os.path.exists(input_path):
        print(f"Error: input path does not exist: '{input_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    lora_files_to_process: List[str] = []

    # --- Determine whether the input is a file or a directory ---
    if os.path.isfile(input_path):
        if input_path.endswith('.safetensors'):
            lora_files_to_process.append(input_path)
        else:
            print(f"Error: input file '{input_path}' is not a .safetensors file.")
            return
    elif os.path.isdir(input_path):
        print(f"Input detected as a directory, scanning '{input_path}'...")
        files_in_dir = [os.path.join(input_path, f) for f in os.listdir(input_path)
                        if f.endswith('.safetensors')]
        lora_files_to_process.extend(files_in_dir)

    if not lora_files_to_process:
        print("No .safetensors files found to process.")
        return

    print(f"Found {len(lora_files_to_process)} LoRA files in total. Starting processing...")

    # --- Batch processing with tqdm ---
    success_count = 0
    fail_count = 0
    for file_path in tqdm(lora_files_to_process, desc="Flipping LoRA Signs"):
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        output_filename = f"{base_name}_flipped{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # In batch mode, verbose is set to False to keep the progress bar clean
        if flip_lora_signs(file_path, output_path, verbose=False):
            success_count += 1
        else:
            fail_count += 1
            # Record the file that failed
            tqdm.write(f"Failed: {os.path.basename(file_path)}")

    print("\n--- Processing complete ---")
    print(f"Succeeded: {success_count} files")
    print(f"Failed/skipped: {fail_count} files")
    print(f"All flipped files were saved to directory: '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flip the signs of LoRA lora_down/lora_up pairs.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input LoRA file or directory containing LoRA files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where the flipped LoRA files are saved.")
    args = parser.parse_args()

    process_path(args.input, args.output_dir)
