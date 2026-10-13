"""Liu et al., CVPR 2025 -- learned projection baseline.

A small MLP trained to project a fixed-length, top-k-magnitude subsample of the
flattened weight update onto a ``hash_length``-dimensional embedding, which is
binarised by sign.  Unlike LoRA-Hash this baseline is **supervised**: it needs a
trained checkpoint and a training set of LoRA pairs.

The trained weights (``cvpr_baseline_trained.pth``) are not distributed with
this repository because of their size.  Without the checkpoint the network is
randomly initialised, so the reported numbers are **not** reproducible -- train
it with :mod:`scripts` or drop the checkpoint next to this file and pass
``model_path``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..common import hamming_distance, iter_delta_w, load_lora, resolve_device

__all__ = ["LiuCVPR2025Baseline"]

DEFAULT_CHECKPOINT = "cvpr_baseline_trained.pth"


class LiuCVPR2025Baseline(nn.Module):
    """Supervised projection network baseline."""

    def __init__(
        self,
        model_path: str = DEFAULT_CHECKPOINT,
        hash_length: int = 256,
        input_dim: int = 100_000,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.hash_length = hash_length
        self.input_dim = input_dim
        self.device = resolve_device(device)

        self.projection_net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, hash_length),
            nn.Tanh(),
        ).to(self.device)

        if os.path.exists(model_path):
            print(f"[LiuCVPR2025] loading trained weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            if any(k.startswith("projection_net.") for k in state_dict):
                state_dict = {
                    k.replace("projection_net.", ""): v for k, v in state_dict.items()
                }
            self.projection_net.load_state_dict(state_dict)
        else:
            print(
                f"[LiuCVPR2025] WARNING: checkpoint '{model_path}' not found; "
                "using random weights (results will not be meaningful)."
            )

        self.projection_net.eval()
        for param in self.projection_net.parameters():
            param.requires_grad = False

    def _preprocess_weights(self, tensors) -> torch.Tensor | None:
        """Flatten ``dW``, keep the top-``input_dim`` magnitudes and standardise."""
        delta_ws = [
            delta_w.flatten() for _, delta_w in iter_delta_w(tensors, self.device)
        ]
        if not delta_ws:
            return None

        full_vector = torch.cat(delta_ws)
        current_len = full_vector.numel()

        if current_len > self.input_dim:
            _, indices = torch.topk(torch.abs(full_vector), self.input_dim)
            values = full_vector[indices]
            return (values - values.mean()) / (values.std() + 1e-9)

        padding = torch.zeros(self.input_dim - current_len, device=self.device)
        return torch.cat([full_vector, padding])

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            return "0" * self.hash_length

        try:
            tensors = load_lora(lora_path)
            input_tensor = self._preprocess_weights(tensors)
            del tensors

            if input_tensor is None:
                return "0" * self.hash_length

            with torch.no_grad():
                features = self.projection_net(input_tensor.unsqueeze(0))

            return "".join("1" if v > 0 else "0" for v in features.squeeze().tolist())
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[LiuCVPR2025] error on {lora_path}: {exc}")
            return "0" * self.hash_length

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        return hamming_distance(h1, h2)
