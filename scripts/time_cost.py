#!/usr/bin/env python3
"""Measure hashing cost (latency and peak GPU memory) for a given method.

Reports the mean wall-clock time per fingerprint over ``--repeats`` runs, plus
the peak CUDA memory allocated since the counters were reset.  The first call is
used as a warm-up so that lazy CUDA initialisation is not counted.

Example
-------
    python scripts/time_cost.py --hasher lora \
        --lora-path /data/hx/lora/benign_SD1.5/animal_SD1.5_cat_20230627113759.safetensors \
        --device cuda:3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_hash import build_hasher  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hasher", default="lora",
                        help="lora, lora-nonorm, sim, nts, cvpr, hos, mean, max, min, abs, std, sha256, md5")
    parser.add_argument("--lora-path", required=True, help="a single .safetensors LoRA to hash repeatedly")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=100, help="number of timed runs")
    args = parser.parse_args()

    if not os.path.exists(args.lora_path):
        raise SystemExit(f"LoRA not found: {args.lora_path}")

    hasher = build_hasher(args.hasher, device=args.device)
    name = type(hasher).__name__
    device = torch.device(args.device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    print(f"[*] hasher = {name}  device = {device}")

    # Warm-up: trigger lazy CUDA/module initialisation.
    hasher.generate_hash(args.lora_path)

    if use_cuda:
        _ = torch.empty(1, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

    start = time.time()
    for _ in range(args.repeats):
        hasher.generate_hash(args.lora_path)

    if use_cuda:
        torch.cuda.synchronize(device=device)

    average_ms = (time.time() - start) / args.repeats * 1000

    peak_memory_mb = 0.0
    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)

    print(f"[{name}] average time : {average_ms:.2f} ms")
    print(f"[{name}] peak GPU mem : {peak_memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
