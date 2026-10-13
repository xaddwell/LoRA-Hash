"""LoRA-Hash: unsupervised perceptual hashing of LoRA weight updates.

Reference
---------
"LoRA-Hash: Unveiling Model Identity from Heavy-Tailed Weight Distributions"
(ICME 2026).

Method
------
A LoRA adapter is identified by a short binary fingerprint derived only from
its weights -- no training, no reference model and no trigger-word prompt is
required.

1. **Weight reconstruction.**  For every layer the effective update
   ``dW = B @ A`` is reconstructed from the low-rank factors.
2. **Per-layer Z-score standardisation.**  Each ``dW`` is standardised to zero
   mean and unit variance.  LoRA updates are extremely heavy-tailed: >99% of
   the entries sit in a "dead zone" near zero, while a handful of extreme
   entries carry essentially all of the semantic content.  Standardising per
   layer makes the fingerprint invariant to the ``alpha`` scaling factor and
   prevents a single large-magnitude layer from dominating the hash.
3. **Segmented extremum extraction.**  The concatenated, standardised vector
   is cut into ``hash_length + 1`` contiguous segments.  Means and standard
   deviations of a heavy-tailed distribution are dominated by its bulk and are
   therefore noise-sensitive, whereas the extrema are stable and informative.
   Each segment is thus summarised by its **maximum** and its **minimum**.
4. **dHash encoding.**  Each channel is binarised by comparing neighbouring
   segment statistic (``1`` if it exceeds its successor, else ``0``), which is
   robust to global amplitude changes.  The two channels are concatenated into
   a ``2 * hash_length`` bit fingerprint (256 bits by default).

The resulting hash is stable under benign reparameterisations and mild weight
perturbations, yet separates distinct models -- the property exploited for
plagiarism detection and model-identity verification.
"""

from __future__ import annotations

import os

import torch

from .common import (
    dhash_bits,
    hamming_distance,
    iter_delta_w,
    load_lora,
    resolve_device,
)

__all__ = ["LoraPerceptualHasher", "LoRAHash"]


class LoraPerceptualHasher:
    """Unsupervised perceptual hasher for LoRA checkpoints.

    Parameters
    ----------
    hash_length:
        Number of bits produced by each of the two (max/min) channels.  The
        final fingerprint is ``2 * hash_length`` bits long (default: 256).
    range_threshold:
        Layers whose ``dW`` spans a range smaller than this are treated as
        inactive and skipped.
    min_segment_size:
        Minimum number of weights per segment.  If the model has too few
        parameters to fill ``hash_length + 1`` segments of this size the hash
        cannot be computed and a sentinel is returned.
    device:
        ``"cuda"``, ``"cuda:N"`` or ``"cpu"``.  Falls back to CPU when CUDA is
        unavailable.
    normalize:
        Apply per-layer Z-score standardisation (Section 2 of the method).
        Setting it to ``False`` reproduces the "w/o Z-score" ablation.
    """

    def __init__(
        self,
        hash_length: int = 128,
        range_threshold: float = 1e-4,
        min_segment_size: int = 32,
        device: str = "cuda",
        normalize: bool = True,
    ) -> None:
        self.hash_length = hash_length
        self.range_threshold = range_threshold
        self.min_segment_size = min_segment_size
        self.normalize = normalize
        self.device = resolve_device(device)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def _sentinel_length(self) -> int:
        return self.hash_length * 2

    def _collect_layers(self, tensors) -> list[torch.Tensor]:
        """Reconstruct and standardise every active layer's ``dW``."""
        vectors: list[torch.Tensor] = []
        for _, delta_w in iter_delta_w(tensors, self.device):
            if (torch.max(delta_w) - torch.min(delta_w)) < self.range_threshold:
                continue

            vec = delta_w.flatten()
            if vec.numel() == 0:
                continue

            if self.normalize:
                std, mean = torch.std(vec), torch.mean(vec)
                vec = (vec - mean) / std if std > 1e-9 else vec - mean

            if torch.isnan(vec).any():
                continue

            vectors.append(vec)
        return vectors

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        """Return the ``2 * hash_length`` bit fingerprint of ``lora_path``.

        On unrecoverable input errors a sentinel string of the correct length
        is returned (all-``0``/``1``/``2``/``3``) instead of raising, so that a
        batch evaluation can continue past a corrupt checkpoint.
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at: {lora_path}")

        try:
            tensors = load_lora(lora_path)
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[LoRA-Hash] failed to load {lora_path}: {exc}")
            return "0" * self._sentinel_length

        vectors = self._collect_layers(tensors)
        del tensors

        if not vectors:
            return "1" * self._sentinel_length

        long_vector = torch.cat(vectors)
        num_segments = self.hash_length + 1
        segment_size = long_vector.numel() // num_segments

        if segment_size < self.min_segment_size:
            return "2" * self._sentinel_length

        segments = long_vector[: num_segments * segment_size].view(num_segments, -1)

        try:
            max_features = torch.max(segments, dim=1).values.cpu().tolist()
            min_features = torch.min(segments, dim=1).values.cpu().tolist()
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[LoRA-Hash] feature extraction failed: {exc}")
            return "3" * self._sentinel_length

        return dhash_bits(max_features) + dhash_bits(min_features)

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        """Normalised Hamming distance between two fingerprints."""
        return hamming_distance(h1, h2)


#: Convenience alias matching the paper's name for the method.
LoRAHash = LoraPerceptualHasher
