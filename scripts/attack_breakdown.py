#!/usr/bin/env python3
"""Judge the detection quality of a LoRA hash against perceptual ground truth.

Ground truth comes from LPIPS between images generated with the reference and
the tampered LoRA (see ``measure_lpips_ssim.py``); the prediction comes from the
hash Hamming distance produced by ``hash_variants.py``.

Decision rule
-------------
* a tampered LoRA is a **positive** (a genuine copy) when ``LPIPS <= --lpips``;
* the hash **flags** it when ``Hamming distance <= --threshold``.

so ``TP`` = flagged copy, ``FN`` = missed copy, ``FP`` = distinct model wrongly
flagged.  Precision, recall and F1 are reported overall and per attack family.

Example
-------
    python scripts/attack_breakdown.py \
        --lpips-dir results/lpips_ssim_similarity \
        --dist-dir results/hanming_distance/lorahash \
        --threshold 0.0664 --lpips 0.30

    # sweep the threshold to see the best achievable F1
    python scripts/attack_breakdown.py --dist-dir results/hanming_distance/lorahash --sweep
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DEFAULT_LPIPS_DIR = "results/lpips_ssim_similarity"
DEFAULT_DIST_DIR = "results/hanming_distance/lorahash"


def get_attack_type(filename: str) -> str:
    """Map a variant filename to its attack family (``Ignore`` = not evaluated)."""
    name = filename.lower()

    if "finetune" in name or "ft_" in name:
        # Legitimate re-training on top of the model is not a plagiarism case.
        return "Ignore"
    if "pruned" in name:
        return "pruned"
    if "rank" in name:
        return "rank"
    if "_st_" in name:
        return "Noise"
    if "flipped" in name:
        return "Flip"
    return "Unknown Attack"


def load_data(lpips_dir: str, dist_dir: str) -> pd.DataFrame:
    """Merge the LPIPS ground truth with the hash distances by variant filename."""
    lpips_map = {}
    for filename in sorted(os.listdir(lpips_dir)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(lpips_dir, filename)) as handle:
                content = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        items = content if isinstance(content, list) else content.get("variants", [])
        for item in items:
            fname = (
                item.get("tampered_file")
                or item.get("variant_name")
                or item.get("filename")
            )
            score = item.get("avg_lpips")
            if fname and score is not None:
                lpips_map[os.path.basename(fname)] = float(score)

    hamming_map = {}
    for filename in sorted(os.listdir(dist_dir)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(dist_dir, filename)) as handle:
                content = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        for variant in content.get("variants", []):
            if "hamming_distance" in variant:
                hamming_map[os.path.basename(variant["variant_name"])] = variant[
                    "hamming_distance"
                ]

    rows = []
    for name, lpips in lpips_map.items():
        attack_type = get_attack_type(name)
        if attack_type == "Ignore":
            continue
        distance = hamming_map.get(name, hamming_map.get(name.replace(" ", "_")))
        if distance is None:
            continue
        rows.append(
            {
                "filename": name,
                "attack_type": attack_type,
                "lpips": lpips,
                "distance": distance,
            }
        )
    return pd.DataFrame(rows)


def confusion(df: pd.DataFrame, threshold: float, lpips_boundary: float):
    """Return ``(tp, fp, fn, tn)`` for the given decision threshold."""
    positive = df["lpips"] <= lpips_boundary
    flagged = df["distance"] <= threshold
    tp = int((positive & flagged).sum())
    fn = int((positive & ~flagged).sum())
    fp = int((~positive & flagged).sum())
    tn = int((~positive & ~flagged).sum())
    return tp, fp, fn, tn


def scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def report(df: pd.DataFrame, threshold: float, lpips_boundary: float) -> None:
    print("=" * 88)
    print(f"Detection report  |  Hamming threshold = {threshold}  |  LPIPS copy boundary = {lpips_boundary}")
    print("=" * 88)

    positives = df[df["lpips"] <= lpips_boundary].copy()
    negatives = df[df["lpips"] > lpips_boundary].copy()
    positives["detected"] = positives["distance"] <= threshold
    negatives["false_alarm"] = negatives["distance"] <= threshold

    print(f"\n[Table 1] Recall on genuine copies (LPIPS <= {lpips_boundary}), n = {len(positives)}")
    print(f"{'Attack type':<25} | {'Count':>6} | {'TP':>6} | {'Recall %':>9}")
    print("-" * 60)
    for attack, group in positives.groupby("attack_type"):
        tp = int(group["detected"].sum())
        print(f"{attack:<25} | {len(group):>6} | {tp:>6} | {tp / len(group) * 100:>9.2f}")
    print("-" * 60)
    total_tp = int(positives["detected"].sum()) if len(positives) else 0
    total_recall = total_tp / len(positives) * 100 if len(positives) else 0.0
    print(f"{'OVERALL':<25} | {len(positives):>6} | {total_tp:>6} | {total_recall:>9.2f}")

    print(f"\n[Table 2] False alarms on distinct models (LPIPS > {lpips_boundary}), n = {len(negatives)}")
    print(f"{'Attack type':<25} | {'Count':>6} | {'FP':>6} | {'FP rate %':>9}")
    print("-" * 60)
    for attack, group in negatives.groupby("attack_type"):
        fp = int(group["false_alarm"].sum())
        print(f"{attack:<25} | {len(group):>6} | {fp:>6} | {fp / len(group) * 100:>9.2f}")
    print("-" * 60)
    total_fp = int(negatives["false_alarm"].sum()) if len(negatives) else 0
    total_fpr = total_fp / len(negatives) * 100 if len(negatives) else 0.0
    print(f"{'OVERALL':<25} | {len(negatives):>6} | {total_fp:>6} | {total_fpr:>9.2f}")

    tp, fp, fn, tn = confusion(df, threshold, lpips_boundary)
    precision, recall, f1 = scores(tp, fp, fn)
    print("\n[Summary]")
    print(f"  TP = {tp}   FP = {fp}   FN = {fn}   TN = {tn}")
    print(f"  Precision = {precision * 100:.2f}%   Recall = {recall * 100:.2f}%   F1 = {f1:.4f}")


def sweep(df: pd.DataFrame, lpips_boundary: float) -> None:
    """Print the F1-optimal Hamming threshold."""
    print("=" * 60)
    print(f"Threshold sweep (LPIPS boundary = {lpips_boundary})")
    print("=" * 60)
    best = None
    for threshold in np.arange(0.0, 0.51, 0.002):
        tp, fp, fn, _tn = confusion(df, float(threshold), lpips_boundary)
        precision, recall, f1 = scores(tp, fp, fn)
        if best is None or f1 > best[3]:
            best = (float(threshold), precision, recall, f1)
    threshold, precision, recall, f1 = best
    print(f"  best threshold = {threshold:.3f}   "
          f"precision = {precision * 100:.2f}%   recall = {recall * 100:.2f}%   F1 = {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lpips-dir", default=DEFAULT_LPIPS_DIR,
                        help="directory of *_metric_results.json from measure_lpips_ssim.py")
    parser.add_argument("--dist-dir", default=DEFAULT_DIST_DIR,
                        help="directory of comparison_*.json from hash_variants.py")
    parser.add_argument("--threshold", type=float, default=0.0664,
                        help="Hamming distance decision threshold")
    parser.add_argument("--lpips", type=float, default=0.30,
                        help="LPIPS boundary separating copies from distinct models")
    parser.add_argument("--sweep", action="store_true", help="also report the F1-optimal threshold")
    args = parser.parse_args()

    if not os.path.isdir(args.dist_dir):
        raise SystemExit(f"distance directory not found: {args.dist_dir}")

    df = load_data(args.lpips_dir, args.dist_dir)
    if df.empty:
        raise SystemExit("no matched LPIPS/hash data found; check --lpips-dir and --dist-dir")

    report(df, args.threshold, args.lpips)
    if args.sweep:
        sweep(df, args.lpips)


if __name__ == "__main__":
    main()
