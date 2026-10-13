"""HOS baseline: higher-order-statistic hashing (Yang & Chen, arXiv 2025).

The largest ``selection_ratio`` fraction of the weights by magnitude is kept,
segmented, and each segment is summarised by its skewness and kurtosis.  The two
statistics are binarised **separately** (a single mixed stream would be
dominated by kurtosis, which has a much larger dynamic range) and concatenated
into a ``2 * hash_length`` bit hash.
"""

from __future__ import annotations

import os

import torch

from ..common import dhash_bits, hamming_distance, iter_delta_w, load_lora, resolve_device

__all__ = ["HOSHasher"]


class HOSHasher:
    """Split-stream skewness/kurtosis hashing."""

    def __init__(
        self,
        hash_length: int = 128,
        selection_ratio: float = 0.0625,
        device: str = "cuda",
    ) -> None:
        self.hash_length = hash_length
        self.selection_ratio = selection_ratio
        self.device = resolve_device(device)

    @staticmethod
    def _compute_hos(tensor: torch.Tensor) -> tuple[float, float]:
        """Return ``(skewness, kurtosis)`` of ``tensor`` via standardised moments."""
        if tensor.numel() == 0:
            return 0.0, 0.0

        mean, std = torch.mean(tensor), torch.std(tensor)
        if std < 1e-9:
            return 0.0, 0.0

        z = (tensor - mean) / std
        return float(torch.mean(z**3)), float(torch.mean(z**4))

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            return "0" * (self.hash_length * 2)

        try:
            tensors = load_lora(lora_path)
            delta_ws = [
                delta_w.flatten()
                for _, delta_w in iter_delta_w(tensors, self.device)
            ]
            del tensors

            if not delta_ws:
                return "0" * (self.hash_length * 2)

            full_vector = torch.cat(delta_ws)

            # --- L1-norm based weight selection ---------------------------- #
            k = int(full_vector.numel() * self.selection_ratio)
            num_segments = self.hash_length + 1
            if k < num_segments:
                k = full_vector.numel()
            # topk on magnitudes, keeping the signed values for the moments
            _, topk_indices = torch.topk(torch.abs(full_vector), k)
            selected = full_vector[topk_indices]

            segment_length = selected.numel() // num_segments
            if segment_length < 1:
                return "0" * (self.hash_length * 2)

            segments = selected[: num_segments * segment_length].view(num_segments, -1)

            # --- split-stream higher-order moments ------------------------- #
            skewness, kurtosis = [], []
            for segment in segments:
                s, ku = self._compute_hos(segment)
                skewness.append(s)
                kurtosis.append(ku)

            return dhash_bits(skewness) + dhash_bits(kurtosis)
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[HOS] error on {lora_path}: {exc}")
            return "0" * (self.hash_length * 2)

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        return hamming_distance(h1, h2)
