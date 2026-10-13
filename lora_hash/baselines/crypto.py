"""Cryptographic hashing baselines (SHA-256 / MD5).

These operate on the raw *file bytes* rather than on the weights, which is the
standard integrity-checking approach.  They are included to show why such
hashes are unsuitable for plagiarism detection: they are perfectly sensitive to
a single flipped bit (avalanche effect), so any benign reparameterisation or
mild fine-tuning produces a completely unrelated digest.
"""

from __future__ import annotations

import hashlib
import os

__all__ = ["CryptoBaseHasher", "SHA256Hasher", "MD5Hasher"]


class CryptoBaseHasher:
    """File-digest baseline exposing the same interface as the perceptual hashers."""

    def __init__(self, device=None, **kwargs) -> None:  # noqa: ARG002 - interface parity
        pass

    def _new_hasher(self):
        raise NotImplementedError

    def generate_hash(self, lora_path: str, verbose: bool = False) -> str:
        if not os.path.exists(lora_path):
            if verbose:
                print(f"[{type(self).__name__}] file not found: {lora_path}")
            return "File Not Found"

        try:
            hasher = self._new_hasher()
            with open(lora_path, "rb") as handle:
                for block in iter(lambda: handle.read(4096), b""):
                    hasher.update(block)
            return hasher.hexdigest()
        except Exception as exc:  # noqa: BLE001 - batch robustness
            if verbose:
                print(f"[{type(self).__name__}] error: {exc}")
            return "Error"

    @staticmethod
    def calculate_hamming_distance(h1: str, h2: str) -> float:
        """Normalised bit-level Hamming distance between two hex digests.

        Because of the avalanche effect the result is ~0.5 for any two
        different files -- i.e. the baseline carries no graded similarity
        signal at all.
        """
        if h1 == h2:
            return 0.0
        if len(h1) != len(h2):
            return 1.0
        distance = bin(int(h1, 16) ^ int(h2, 16)).count("1")
        return distance / (len(h1) * 4)


class SHA256Hasher(CryptoBaseHasher):
    def _new_hasher(self):
        return hashlib.sha256()


class MD5Hasher(CryptoBaseHasher):
    def _new_hasher(self):
        return hashlib.md5()
