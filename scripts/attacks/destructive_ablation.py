#!/usr/bin/env python3
"""Destructive ablation: does zeroing the extreme weights break the model?

Motivation experiment for the heavy-tailed assumption behind LoRA-Hash.  For a
reference LoRA three variants are produced and the *same* prompts and seeds are
then used to sample images with each of them (see
``scripts/measure_lpips_ssim.py``):

``top``     zero the ``ratio`` largest-magnitude weights of every layer
            (the heavy tail that LoRA-Hash hashes);
``bottom``  zero the ``ratio`` smallest-magnitude weights ("dead zone" noise);
``random``  zero ``ratio`` of the weights uniformly at random (control).

Comparing the resulting LPIPS shows that removing a tiny fraction of the
extreme weights collapses the model's behaviour, while removing a much larger
fraction of the bulk changes nothing perceptible -- i.e. the semantic content is
concentrated in the tail.

Example
-------
    python scripts/attacks/destructive_ablation.py \
        --input /data/hx/lora/benign_SD1.5/animal_SD1.5_cat_20230627113759.safetensors \
        --output-dir /data/hx/lora/ablation --ratio 0.005
"""

from __future__ import annotations

import argparse
import copy
import os

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

MODES = ("top", "bottom", "random")


def ablation_attack(input_path: str, output_dir: str, ratio: float = 0.01) -> None:
    """Write one ablated LoRA per mode into ``output_dir``."""
    if not os.path.exists(input_path):
        print(f"error: file not found: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"loading LoRA: {input_path}")
    original_state_dict = load_file(input_path)

    for mode in MODES:
        print(f"mode {mode.upper()}: zeroing {ratio * 100:.2f}% of each layer")
        new_state_dict = copy.deepcopy(original_state_dict)

        for key in tqdm(new_state_dict.keys(), desc=f"processing {mode}"):
            # Only the LoRA weight factors take part; metadata is skipped.
            if "lora_" not in key or "weight" not in key:
                continue

            tensor = new_state_dict[key]
            flat_tensor = tensor.flatten()
            n_params = flat_tensor.numel()
            n_modify = int(n_params * ratio)
            if n_modify == 0:
                continue

            abs_tensor = torch.abs(flat_tensor)

            if mode == "top":
                # Largest magnitudes: the heavy tail.
                _, indices = torch.topk(abs_tensor, n_modify)
            elif mode == "bottom":
                # Smallest magnitudes: the "dead zone" bulk.
                _, indices = torch.topk(-abs_tensor, n_modify)
            else:
                indices = torch.randperm(n_params)[:n_modify]

            mask = torch.ones_like(flat_tensor)
            mask[indices] = 0
            new_state_dict[key] = (flat_tensor * mask).view(tensor.shape)

        filename = os.path.basename(input_path).replace(".safetensors", "")
        save_name = f"{filename}_ABLA_{mode}_{ratio * 100:.1f}pct.safetensors"
        save_path = os.path.join(output_dir, save_name)
        save_file(new_state_dict, save_path)
        print(f"saved: {save_path}")

    print("\n=== done ===")
    print(
        "Next: measure LPIPS for each ablated LoRA against the reference "
        "(scripts/measure_lpips_ssim.py) using the same prompts and seeds."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="reference LoRA (.safetensors)")
    parser.add_argument("--output-dir", required=True, help="where the ablated LoRAs are written")
    parser.add_argument("--ratio", type=float, default=0.01,
                        help="fraction of weights zeroed per layer (default: 0.01 = 1%%)")
    args = parser.parse_args()
    ablation_attack(args.input, args.output_dir, args.ratio)


if __name__ == "__main__":
    main()
