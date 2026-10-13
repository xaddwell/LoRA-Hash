#!/usr/bin/env python3
"""Smoke tests for LoRA-Hash that need no external dataset.

Builds synthetic kohya-style LoRA checkpoints in a temporary directory and
checks the contract of the public API: fingerprint length, determinism,
identical-file distance, and separation between unrelated weights.

Run directly (``python tests/test_hasher.py``) or with pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_hash import LoraPerceptualHasher, hamming_distance  # noqa: E402


def _write_synthetic_lora(path: str, seed: int, layers: int = 3, rank: int = 8, dim: int = 64) -> None:
    """Write a LoRA whose ``dW`` blocks are large enough to be hashable."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for index in range(layers):
        down = torch.randn(rank, dim, generator=generator) * 0.1
        up = torch.randn(dim, rank, generator=generator) * 0.1
        tensors[f"layer{index}.lora_down.weight"] = down
        tensors[f"layer{index}.lora_up.weight"] = up
    save_file(tensors, path)


def test_hash_length_and_determinism():
    hasher = LoraPerceptualHasher(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.safetensors")
        _write_synthetic_lora(path, seed=0)

        h1 = hasher.generate_hash(path)
        h2 = hasher.generate_hash(path)

        assert len(h1) == 256, f"expected 256 bits, got {len(h1)}"
        assert set(h1) <= {"0", "1"}, "fingerprint must be binary"
        assert h1 == h2, "hashing the same file twice must be deterministic"


def test_identical_copy_has_zero_distance():
    hasher = LoraPerceptualHasher(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.safetensors")
        b = os.path.join(tmp, "b.safetensors")
        _write_synthetic_lora(a, seed=1)
        _write_synthetic_lora(b, seed=1)  # identical content, different filename

        assert hasher.calculate_hamming_distance(
            hasher.generate_hash(a), hasher.generate_hash(b)
        ) == 0.0


def test_unrelated_weights_are_separated():
    hasher = LoraPerceptualHasher(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.safetensors")
        b = os.path.join(tmp, "b.safetensors")
        _write_synthetic_lora(a, seed=2)
        _write_synthetic_lora(b, seed=3)

        distance = hasher.calculate_hamming_distance(
            hasher.generate_hash(a), hasher.generate_hash(b)
        )
        assert distance > 0.0, "two unrelated adapters must not collide exactly"


def test_no_normalization_ablation_runs():
    hasher = LoraPerceptualHasher(device="cpu", normalize=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.safetensors")
        _write_synthetic_lora(path, seed=4)
        assert len(hasher.generate_hash(path)) == 256


def test_hamming_distance_rejects_length_mismatch():
    try:
        hamming_distance("0101", "01")
    except ValueError:
        return
    raise AssertionError("mismatched hash lengths should raise ValueError")


def main() -> int:
    tests = [
        test_hash_length_and_determinism,
        test_identical_copy_has_zero_distance,
        test_unrelated_weights_are_separated,
        test_no_normalization_ablation_runs,
        test_hamming_distance_rejects_length_mismatch,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
