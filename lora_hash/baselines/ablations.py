"""Ablation hashers -- feature-channel variants of LoRA-Hash.

These share the exact pipeline of :class:`~lora_hash.hasher.LoraPerceptualHasher`
(reconstruction, per-layer Z-score, segmentation, dHash) and differ **only** in
which scalar statistic summarises each segment.  They isolate the contribution
of the extremum channels against the bulk statistics (mean/std) and against
single-channel variants.
"""

from __future__ import annotations

import os

import torch

from ..common import dhash_bits, hamming_distance, iter_delta_w, load_lora, resolve_device

__all__ = [
    "AblationBaseHasher",
    "MeanHasher",
    "MaxHasher",
    "MinHasher",
    "AbsHasher",
    "StdHasher",
]


class AblationBaseHasher:
    """Common pipeline; subclasses implement :meth:`_extract_features`."""

    def __init__(
        self,
        hash_length: int = 128,
        range_threshold: float = 1e-4,
        min_segment_size: int = 32,
        device: str = "cuda",
    ) -> None:
        self.hash_length = hash_length
        self.range_threshold = range_threshold
        self.min_segment_size = min_segment_size
        self.device = resolve_device(device)

    @property
    def output_length(self) -> int:
        """Ablations emit a single channel of ``hash_length * 2`` bits."""
        return self.hash_length * 2

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        """Map ``(num_segments, segment_size)`` -> ``(num_segments,)``."""
        raise NotImplementedError

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at: {lora_path}")

        try:
            tensors = load_lora(lora_path)
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[{type(self).__name__}] failed to load {lora_path}: {exc}")
            return "0" * self.output_length

        vectors: list[torch.Tensor] = []
        for _, delta_w in iter_delta_w(tensors, self.device):
            if (torch.max(delta_w) - torch.min(delta_w)) < self.range_threshold:
                continue

            vec = delta_w.flatten()
            if vec.numel() == 0:
                continue

            std, mean = torch.std(vec), torch.mean(vec)
            vec = (vec - mean) / std if std > 1e-9 else vec - mean

            if torch.isnan(vec).any():
                continue

            vectors.append(vec)

        del tensors

        if not vectors:
            return "1" * self.output_length

        long_vector = torch.cat(vectors)
        num_segments = self.output_length + 1
        segment_size = long_vector.numel() // num_segments

        if segment_size < self.min_segment_size:
            return "2" * self.output_length

        segments = long_vector[: num_segments * segment_size].view(num_segments, -1)

        try:
            features = self._extract_features(segments).cpu().tolist()
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[{type(self).__name__}] feature extraction failed: {exc}")
            return "3" * self.output_length

        return dhash_bits(features)

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        return hamming_distance(h1, h2)


class MeanHasher(AblationBaseHasher):
    """Bulk statistic: segment mean."""

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        return torch.mean(segments, dim=1)


class StdHasher(AblationBaseHasher):
    """Bulk statistic: segment standard deviation."""

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        return torch.std(segments, dim=1)


class MaxHasher(AblationBaseHasher):
    """Single extremum channel: segment maximum."""

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        return torch.max(segments, dim=1).values


class MinHasher(AblationBaseHasher):
    """Single extremum channel: segment minimum."""

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        return torch.min(segments, dim=1).values


class AbsHasher(AblationBaseHasher):
    """Single channel over magnitudes: segment maximum of ``|dW|``."""

    def _extract_features(self, segments: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(segments), dim=1).values
