#!/usr/bin/env python3
"""Add std-scaled Gaussian noise to LoRA weights.

For every floating-point tensor in a LoRA ``.safetensors`` file, Gaussian noise
is drawn with standard deviation ``weight_std * noise_scale`` and added to the
weights. Scalar tensors and tensors with zero standard deviation are skipped.
Runs on a single file or in batch mode over a directory, and writes one output
file per noise scale, named ``<base>_st_<scale>.safetensors``.

Usage:
    python noise_injection.py --input data/benign --output data/noise --batch
"""

import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file
import argparse

def add_noise_to_lora(input_path, output_path, noise_scale=0.01):
    """
    Add noise based on the weight standard deviation to a single LoRA file.
    """
    tensors = {}
    original_dtypes = {}

    try:
        with safe_open(input_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors[key] = tensor
                original_dtypes[key] = tensor.dtype
    except Exception as e:
        print(f"Error: could not read file {input_path}. Please check that it is valid. "
              f"Details: {e}")
        return

    for key in tensors:
        tensor = tensors[key]

        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            if tensor.numel() == 1:
                # print(f"Skipping scalar tensor: {key} (value: {tensor.item()})")
                continue

            tensor_fp32 = tensor.to(torch.float32)
            weight_std = torch.std(tensor_fp32)

            if weight_std == 0:
                # print(f"Skipping tensor with zero standard deviation: {key}")
                continue

            noise = torch.randn_like(tensor_fp32) * weight_std * noise_scale
            noisy_tensor_fp32 = tensor_fp32 + noise
            tensors[key] = noisy_tensor_fp32.to(original_dtypes[key])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        save_file(tensors, output_path)
        print(f"Generated noisy version: {output_path}")
    except Exception as e:
        print(f"Error: could not save file to {output_path}. Details: {e}")


def batch_add_noise_to_loras(input_dir, output_dir, noise_scales=None):
    """Add noise to multiple LoRA files in batch mode and name the outputs automatically."""
    if noise_scales is None:
        noise_scales = [0.01, 0.05, 0.1, 0.2]

    os.makedirs(output_dir, exist_ok=True)

    lora_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]

    if not lora_files:
        print(f"No .safetensors files found in directory {input_dir}")
        return

    print(f"Found {len(lora_files)} LoRA files, starting batch processing...")

    for lora_file in lora_files:
        input_path = os.path.join(input_dir, lora_file)
        base_name = os.path.splitext(lora_file)[0]

        for noise_scale in noise_scales:
            # --- Automatic output naming ---
            # Replace '.' in the float with '_' to avoid filename issues
            scale_str = str(noise_scale).replace('.', '_')
            output_filename = f"{base_name}_st_{scale_str}.safetensors"
            output_path = os.path.join(output_dir, output_filename)

            add_noise_to_lora(input_path, output_path, noise_scale)

    print(f"\nBatch processing complete: {len(lora_files)} files, each with "
          f"{len(noise_scales)} noisy versions")


def main():
    parser = argparse.ArgumentParser(
        description='Add std-scaled Gaussian noise to LoRA files with automatic naming')

    # --- Unified meaning for --input and --output ---
    parser.add_argument('--input', type=str, required=True,
                        help='Input LoRA file (single-file mode) or directory (batch mode)')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')

    parser.add_argument('--noise-scale', type=float, default=0.1,
                        help='Noise scale factor in single-file mode (default: 0.1)')
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch mode; --input must then be a directory')
    parser.add_argument('--noise-scales', nargs='+', type=float,
                        default=[0.01, 0.05, 0.1, 0.2],
                        help='List of noise scale factors for batch mode')

    args = parser.parse_args()

    if args.batch:
        # Batch processing mode
        if not os.path.isdir(args.input):
            print(f"Error: in batch mode, input path '{args.input}' must be a directory.")
            return
        batch_add_noise_to_loras(args.input, args.output, args.noise_scales)

    else:
        # Single-file processing mode
        if not os.path.isfile(args.input):
            print(f"Error: in single-file mode, input path '{args.input}' must be a file.")
            return

        # --- Automatic output path for a single file ---
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        scale_str = str(args.noise_scale).replace('.', '_')
        output_filename = f"{base_name}_st_{scale_str}.safetensors"
        output_path = os.path.join(args.output, output_filename)

        add_noise_to_lora(args.input, output_path, args.noise_scale)

if __name__ == "__main__":
    main()
