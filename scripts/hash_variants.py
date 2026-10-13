#!/usr/bin/env python3
"""Hash every benign LoRA and its tampered variants, then report Hamming distances.

This is the script that produces the per-model JSON files consumed by
``attack_breakdown.py`` (decision thresholds, precision/recall/F1) and by the
ROC/AUC plots.

Directory layout expected::

    <lora-root>/
        benign_SD1.5/            <- the reference LoRAs (one file per model)
            animal_....safetensors
        finetune/                <- tampered copies, prefixed with the base name
            animal_..._finetune_light.safetensors
        noise/  prune/  flip/  distill/  extract/  ablation/

Example
-------
    python scripts/hash_variants.py \
        --lora-root /data/hx/lora \
        --benign-dir /data/hx/lora/benign_SD1.5 \
        --hasher lora \
        --output-dir results/hanming_distance/lorahash
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_hash import build_hasher  # noqa: E402

DEFAULT_LORA_ROOT = "/data/hx/lora"
DEFAULT_BENIGN_SUBDIR = "benign_SD1.5"


def find_lora_variants(base_name, benign_lora_dir, lora_root):
    """Return ``(base_path, [variant_paths])`` for ``base_name``.

    A variant is any ``.safetensors`` file outside ``benign_lora_dir`` whose name
    starts with ``base_name``.
    """
    base_lora_path = os.path.join(benign_lora_dir, f"{base_name}.safetensors")
    if not os.path.exists(base_lora_path):
        print(f"[!] base LoRA not found: {base_lora_path}")
        return "", []

    variant_paths = []
    benign_abs = os.path.abspath(benign_lora_dir)
    for subdir in sorted(os.listdir(lora_root)):
        subdir_path = os.path.join(lora_root, subdir)
        if not os.path.isdir(subdir_path) or os.path.abspath(subdir_path) == benign_abs:
            continue
        for filename in sorted(os.listdir(subdir_path)):
            if filename.startswith(base_name) and filename.endswith(".safetensors"):
                variant_paths.append(os.path.join(subdir_path, filename))

    return base_lora_path, sorted(variant_paths)


def process_single_lora(name, benign_dir, lora_root, hasher, output_dir, verbose):
    print("\n" + "=" * 70)
    print(f"base LoRA : {name}")
    print(f"benign dir: {benign_dir}")
    print("=" * 70)

    base_path, variant_paths = find_lora_variants(name, benign_dir, lora_root)
    if not base_path:
        return

    try:
        base_hash = hasher.generate_hash(base_path, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] failed to hash base LoRA: {exc}")
        return

    print(f"base hash : {base_hash}")
    results = {"base_lora_name": name, "base_lora_hash": base_hash, "variants": []}

    for variant_path in variant_paths:
        variant_name = os.path.basename(variant_path)
        try:
            variant_hash = hasher.generate_hash(variant_path, verbose=verbose)
            distance = hasher.calculate_hamming_distance(base_hash, variant_hash)
            print(f"  {variant_name:<60} d = {distance:.4f}")
            results["variants"].append(
                {"variant_name": variant_name, "hamming_distance": round(distance, 4)}
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {variant_name:<60} ERROR: {exc}")
            results["variants"].append({"variant_name": variant_name, "error": str(exc)})

    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_path = os.path.join(output_dir, f"comparison_{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4, ensure_ascii=False)
    print(f"[ok] wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora-root", default=DEFAULT_LORA_ROOT,
                        help="root directory containing the tampered-copy subfolders")
    parser.add_argument("--benign-dir", default=None,
                        help="directory holding the reference LoRAs "
                             f"(default: <lora-root>/{DEFAULT_BENIGN_SUBDIR})")
    parser.add_argument("--input-target", default=None,
                        help="a single LoRA name, or a directory of reference LoRAs; "
                             "defaults to --benign-dir")
    parser.add_argument("--hasher", default="lora",
                        help="lora, lora-nonorm, sim, nts, cvpr, hos, mean, max, min, abs, std, sha256, md5")
    parser.add_argument("--output-dir", default=None,
                        help="where to write comparison_*.json (default: results/hanming_distance/<hasher>hash)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true", help="print intermediate steps")
    args = parser.parse_args()

    benign_dir = args.benign_dir or os.path.join(args.lora_root, DEFAULT_BENIGN_SUBDIR)
    input_target = args.input_target or benign_dir
    output_dir = args.output_dir or os.path.join(
        "results", "hanming_distance", f"{args.hasher.lower()}hash"
    )

    hasher = build_hasher(args.hasher, device=args.device)
    print(f"[*] hasher: {type(hasher).__name__} on {args.device}")

    if os.path.isdir(input_target):
        print(f"[*] batch mode over {input_target}")
        names = [
            os.path.splitext(f)[0]
            for f in sorted(os.listdir(input_target))
            if f.endswith(".safetensors")
        ]
        if not names:
            print("[!] no .safetensors files found")
            return
    else:
        print(f"[*] single mode: {input_target}")
        names = [input_target]

    for index, name in enumerate(names, start=1):
        print(f"\n--- {index}/{len(names)} ---")
        process_single_lora(name, benign_dir, args.lora_root, hasher, output_dir, args.verbose)

    print("\n[*] all done")


if __name__ == "__main__":
    main()
