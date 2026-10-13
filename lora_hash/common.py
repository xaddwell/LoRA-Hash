"""Shared utilities for LoRA-Hash and the baseline hashers.

All hashers in this package follow the same three-stage pipeline:

1. **Reparameterization** -- a LoRA layer stores ``lora_down`` (A) and
   ``lora_up`` (B); the weight update it applies is ``dW = B @ A``.  Every
   method therefore starts by materialising ``dW`` for each layer.
2. **Feature extraction** -- turn the (very large) set of ``dW`` entries into
   a small fixed-length vector of scalar features.
3. **Binarisation** -- turn the feature vector into a bit string, either by
   dHash (comparing neighbouring features) or by sign thresholding.

This module implements stage 1 and the shared binarisation helpers so that the
individual hashers only contain the part that actually distinguishes them.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
from safetensors.torch import load_file

__all__ = [
    "resolve_device",
    "load_lora",
    "iter_delta_w",
    "dhash_bits",
    "sign_bits",
    "hamming_distance",
    "DOWN_SUFFIX",
    "UP_SUFFIX",
]

DOWN_SUFFIX = "lora_down.weight"
UP_SUFFIX = "lora_up.weight"


def resolve_device(device: str | torch.device) -> torch.device:
    """Return a usable :class:`torch.device`, falling back to CPU.

    ``device`` may be ``"cuda"``, ``"cuda:N"``, ``"cpu"`` or a device object.
    If CUDA is requested but unavailable the CPU is used instead.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def load_lora(lora_path: str) -> Dict[str, torch.Tensor]:
    """Load a ``.safetensors`` LoRA checkpoint into CPU tensors."""
    return load_file(lora_path, device="cpu")


def iter_delta_w(
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    sort: bool = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(key, dW)`` for every LoRA layer in ``tensors``.

    With ``sort=True`` (the default) layers are visited in sorted key order so
    that hashes are deterministic across runs, machines and checkpoint file
    layouts.  ``sort=False`` preserves the checkpoint's own key order, which is
    what the NTS baseline was evaluated with.

    ``dW`` is materialised on ``device`` as

    .. math:: \\Delta W = B A

    where ``A`` is ``lora_down.weight`` and ``B`` is ``lora_up.weight``.
    Convolutional LoRA layers (4-D tensors) are squeezed to 2-D first.
    """
    down_keys = [k for k in tensors if k.endswith(DOWN_SUFFIX)]
    if sort:
        down_keys.sort()
    for down_key in down_keys:
        up_key = down_key.replace(DOWN_SUFFIX, UP_SUFFIX)
        if up_key not in tensors:
            continue
        a = tensors[down_key].to(device, dtype=dtype, non_blocking=True)
        b = tensors[up_key].to(device, dtype=dtype, non_blocking=True)
        if a.dim() == 4:
            a = a.squeeze()
        if b.dim() == 4:
            b = b.squeeze()
        yield down_key, b @ a


def dhash_bits(values) -> str:
    """Difference hash: ``"1"`` where a value exceeds its successor.

    A sequence of ``n`` values yields ``n - 1`` bits.
    """
    values = list(values)
    return "".join(
        "1" if values[i] > values[i + 1] else "0" for i in range(len(values) - 1)
    )


def sign_bits(values) -> str:
    """Sign hash: ``"1"`` for non-negative values, ``"0"`` otherwise."""
    return "".join("1" if float(v) >= 0.0 else "0" for v in values)


def hamming_distance(h1: str, h2: str) -> float:
    """Normalised Hamming distance between two equal-length hash strings."""
    if len(h1) != len(h2):
        raise ValueError(
            f"Hash length mismatch: {len(h1)} != {len(h2)}. "
            "Both hashes must be produced by the same configuration."
        )
    return sum(c1 != c2 for c1, c2 in zip(h1, h2)) / len(h1)
