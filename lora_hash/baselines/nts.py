"""NTS baseline: normality-test statistic hashing.

Selects the top ``1/c`` weights by magnitude, standardises them and splits the
result into ``hash_length + 1`` segments.  Each segment is summarised by the
Shapiro-Wilk normality statistic, and neighbouring statistics are compared with
dHash.  The Shapiro-Wilk test runs on CPU (PyTorch has no GPU implementation),
so only the already-selected, segmented weights are transferred.
"""

from __future__ import annotations

import os

import torch
from scipy import stats

from ..common import dhash_bits, hamming_distance, iter_delta_w, load_lora, resolve_device

__all__ = ["NTSHashingBaseline"]


class NTSHashingBaseline:
    """Normality-test-statistic hashing of the flattened weight update."""

    def __init__(
        self,
        hash_length: int = 256,
        selection_ratio_c: int = 32,
        device: str = "cuda",
    ) -> None:
        self.hash_length = hash_length
        self.c = selection_ratio_c
        self.device = resolve_device(device)

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            return "0" * self.hash_length

        try:
            tensors = load_lora(lora_path)
            delta_ws = [
                delta_w.flatten()
                for _, delta_w in iter_delta_w(tensors, self.device, sort=False)
            ]
            del tensors

            if not delta_ws:
                return "0" * self.hash_length

            all_weights = torch.cat(delta_ws)

            # --- magnitude-based weight selection ------------------------- #
            abs_weights = torch.abs(all_weights)
            threshold = torch.quantile(abs_weights, 1.0 - (1.0 / self.c))
            selected = all_weights[abs_weights >= threshold]

            if selected.numel() < self.hash_length + 1:
                return "0" * self.hash_length

            # --- standardise and segment ---------------------------------- #
            mean = torch.mean(selected)
            std = torch.std(selected) + 1e-9
            normalised = (selected - mean) / std

            num_segments = self.hash_length + 1
            segment_length = normalised.numel() // num_segments
            segments = normalised[: num_segments * segment_length].view(num_segments, -1)

            # --- per-segment normality statistic (CPU-bound) --------------- #
            features = []
            for segment in segments.cpu().numpy():
                features.append(
                    float(stats.shapiro(segment)[0]) if len(segment) >= 3 else 0.0
                )

            return dhash_bits(features)
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[NTS] error on {lora_path}: {exc}")
            return "0" * self.hash_length

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        return hamming_distance(h1, h2)
