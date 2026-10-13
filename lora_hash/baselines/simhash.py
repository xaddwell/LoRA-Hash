"""SimHash baseline (random-hyperplane LSH over the flattened weight update)."""

from __future__ import annotations

import os

import torch

from ..common import hamming_distance, iter_delta_w, load_lora, resolve_device

__all__ = ["SimHashBaseline"]


class SimHashBaseline:
    """SimHash with fixed random hyperplanes.

    The flattened ``dW`` is projected onto ``hash_length`` random Gaussian
    directions and each projection sign becomes one bit.  The projection matrix
    is regenerated deterministically from ``seed`` in chunks, so the memory cost
    stays bounded even for very large models.
    """

    def __init__(
        self,
        hash_length: int = 256,
        seed: int = 44,
        device: str = "cuda",
        chunk_size: int = 4_000_000,
    ) -> None:
        self.hash_length = hash_length
        self.seed = seed
        self.chunk_size = chunk_size
        self.device = resolve_device(device)
        self.compute_dtype = torch.float32

    def _accumulate_projection(
        self,
        flat_values: torch.Tensor,
        rng: torch.Generator,
        accumulator: torch.Tensor,
    ) -> None:
        """Accumulate ``sum_i v_i * r_i`` where ``r_i`` are per-element random vectors.

        ``rng`` is shared across all layers of a single ``generate_hash`` call so
        that the projection is a deterministic function of ``seed`` alone.
        """
        view = flat_values.view(-1)
        total = view.numel()
        start = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            chunk = view[start:end].to(self.device, dtype=self.compute_dtype)
            proj = torch.randn(
                (end - start, self.hash_length),
                generator=rng,
                device=self.device,
                dtype=self.compute_dtype,
            )
            accumulator += torch.matmul(chunk.unsqueeze(0), proj).squeeze(0)
            start = end

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at: {lora_path}")

        try:
            tensors = load_lora(lora_path)
            accumulator = torch.zeros(
                self.hash_length, device=self.device, dtype=self.compute_dtype
            )
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.seed)

            for _, delta_w in iter_delta_w(tensors, self.device, dtype=self.compute_dtype):
                self._accumulate_projection(delta_w.flatten(), rng, accumulator)
            del tensors

            if torch.all(accumulator == 0):
                return "0" * self.hash_length

            return "".join(
                "1" if val >= 0 else "0" for val in accumulator.cpu().tolist()
            )
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[SimHash] error on {lora_path}: {exc}")
            return "0" * self.hash_length

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        return hamming_distance(h1, h2)
