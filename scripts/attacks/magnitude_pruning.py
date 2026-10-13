#!/usr/bin/env python3
"""Magnitude pruning of LoRA weights in ``.safetensors`` files.

All ``lora_down.weight`` / ``lora_up.weight`` tensors are pooled, and the
globally smallest weights (by absolute value) are set to zero so that the
requested percentage of weights is removed. Handles a single file or a whole
directory, producing one pruned file per percentage, named
``<base>_pruned_<pct>pct.safetensors``.

Usage:
    python magnitude_pruning.py --input data/benign --output data/pruned --batch
"""

import torch
from safetensors.torch import load_file, save_file
import argparse
import os

def prune_lora_weights(lora_path: str, output_path: str, pruning_percentage: float):
    """
    Magnitude-prune the weights in a LoRA file.
    (Core processing logic kept unchanged.)
    """
    if not 0.0 <= pruning_percentage <= 100.0:
        raise ValueError("Pruning percentage must be between 0.0 and 100.0.")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"File not found: {lora_path}")

    print(f"\n--- Processing: {os.path.basename(lora_path)} | "
          f"pruning rate: {pruning_percentage}% ---")
    state_dict = load_file(lora_path, device="cpu")

    all_weights = []
    lora_keys = [key for key in state_dict.keys()
                 if 'lora_down.weight' in key or 'lora_up.weight' in key]

    if not lora_keys:
        print("No LoRA weights found, skipping.")
        return

    for key in lora_keys:
        all_weights.append(state_dict[key].flatten())

    flat_weights = torch.cat(all_weights)

    k = int(len(flat_weights) * (pruning_percentage / 100.0))
    if k == 0:
        print("Pruning percentage is 0, nothing to prune.")
        # The file could be copied here, but we return directly to avoid redundancy
        return

    threshold = torch.kthvalue(torch.abs(flat_weights), k).values

    pruned_state_dict = state_dict.copy()

    for key in lora_keys:
        tensor = pruned_state_dict[key]
        mask = torch.abs(tensor) < threshold
        tensor[mask] = 0.0

    # Automatic naming logic
    if os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(lora_path))[0]
        # Integer percentage keeps the name clear
        output_filename = f"{base_name}_pruned_{int(pruning_percentage)}pct.safetensors"
        output_path = os.path.join(output_path, output_filename)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    save_file(pruned_state_dict, output_path)
    print(f"Saved to: {output_path}")

# --- Batch processing function ---
def batch_prune_loras(input_dir, output_dir, percentages):
    """
    Process every LoRA file in a directory and generate several pruning
    levels for each of them.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: input path '{input_dir}' is not a valid directory.")
        return

    lora_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]
    if not lora_files:
        print(f"No .safetensors files found in directory '{input_dir}'.")
        return

    print(f"Found {len(lora_files)} LoRA files in '{input_dir}'; generating "
          f"{len(percentages)} pruning levels for each.")

    for lora_file in lora_files:
        lora_path = os.path.join(input_dir, lora_file)
        for p in percentages:
            # For each file and each percentage, call the core processing function
            # Pass the directory as output_path so the core function names the file
            prune_lora_weights(lora_path, output_dir, p)

    print("\n--- Batch processing complete! ---")

# --- Main entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Magnitude-prune .safetensors LoRA files, single file or batch.")

    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input file or directory.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--batch', action='store_true',
                        help="Enable batch mode. With this flag, --input must be a directory.")

    # Keep --percentage for single-file mode
    parser.add_argument('--percentage', type=float, default=50,
                        help="Percentage of weights to prune in single-file mode (default: 50).")

    # Add --percentages for batch mode
    parser.add_argument('--percentages', nargs='+', type=float, default=[10, 30, 50],
                        help="Percentages of weights to prune in batch mode (default: 10 30 50).")

    args = parser.parse_args()

    if args.batch:
        # In batch mode, call the batch processing function
        batch_prune_loras(args.input, args.output, args.percentages)
    else:
        # Otherwise, process a single file
        if not os.path.isfile(args.input):
            print(f"Error: in single-file mode, input path '{args.input}' must be a file. "
                  f"Use the --batch flag to process a directory.")
        else:
            prune_lora_weights(args.input, args.output, args.percentage)
