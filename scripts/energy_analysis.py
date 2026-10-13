#!/usr/bin/env python3
"""Energy distribution of a LoRA update: how concentrated is the heavy tail?

Supporting analysis for the sparsity hypothesis motivating LoRA-Hash.  For a
given LoRA the script reports, over the concatenation of every ``lora_down`` /
``lora_up`` tensor:

* the total L1 energy of the update;
* the share of that energy held by the largest ``--top-ratio`` of the weights;
* the share held by the smallest ``--bottom-ratio``.

The observation that a fraction of a percent of the weights carries a
disproportionate share of the energy -- while the bottom decile carries almost
none -- is why LoRA-Hash hashes the per-segment extrema instead of the bulk.

Example
-------
    python scripts/energy_analysis.py \
        --input /data/hx/lora/benign_SD1.5/animal_SD1.5_cat_20230627113759.safetensors
"""

from __future__ import annotations

import argparse
import os

import torch
from safetensors.torch import load_file


def analyze_energy_distribution(
    input_path: str,
    top_ratio: float = 0.005,
    bottom_ratio: float = 0.10,
) -> None:
    if not os.path.exists(input_path):
        print(f"error: file not found: {input_path}")
        return

    print(f"analysing energy distribution: {os.path.basename(input_path)}")
    state_dict = load_file(input_path)

    weights = [
        tensor.flatten()
        for key, tensor in state_dict.items()
        if "lora_" in key and "weight" in key
    ]
    if not weights:
        print("no LoRA weights found")
        return

    full_vector = torch.cat(weights)
    n_total = full_vector.numel()
    total_energy = torch.sum(torch.abs(full_vector)).item()

    print(f"total parameters : {n_total}")
    print(f"total energy (L1): {total_energy:.4f}")
    print("-" * 40)

    k_top = int(n_total * top_ratio)
    top_energy = torch.sum(torch.topk(torch.abs(full_vector), k_top).values).item()

    k_bottom = int(n_total * bottom_ratio)
    sorted_abs, _ = torch.sort(torch.abs(full_vector))
    bottom_energy = torch.sum(sorted_abs[:k_bottom]).item()

    print(f"[top {top_ratio * 100:g}% of weights]")
    print(f"  count            : {k_top}")
    print(f"  energy           : {top_energy:.4f}")
    print(f"  share of total   : {top_energy / total_energy * 100:.2f}%")

    print(f"\n[bottom {bottom_ratio * 100:g}% of weights]")
    print(f"  count            : {k_bottom}")
    print(f"  energy           : {bottom_energy:.4f}")
    print(f"  share of total   : {bottom_energy / total_energy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="LoRA (.safetensors) to analyse")
    parser.add_argument("--top-ratio", type=float, default=0.005, help="tail fraction (default: 0.005 = 0.5%%)")
    parser.add_argument("--bottom-ratio", type=float, default=0.10, help="bulk fraction (default: 0.10 = 10%%)")
    args = parser.parse_args()
    analyze_energy_distribution(args.input, args.top_ratio, args.bottom_ratio)


if __name__ == "__main__":
    main()
