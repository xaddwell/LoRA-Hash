"""LoRA-Hash: unsupervised perceptual hashing for LoRA model identity.

Quick start
-----------
>>> from lora_hash import LoraPerceptualHasher
>>> hasher = LoraPerceptualHasher(device="cuda")          # 256-bit fingerprint
>>> h1 = hasher.generate_hash("benign.safetensors")
>>> h2 = hasher.generate_hash("suspect.safetensors")
>>> hasher.calculate_hamming_distance(h1, h2)             # in [0, 1]
0.0352

See ``README.md`` for the full reproduction pipeline.
"""

from .baselines import (
    REGISTRY as _BASELINE_REGISTRY,
    AbsHasher,
    HOSHasher,
    LiuCVPR2025Baseline,
    MD5Hasher,
    MaxHasher,
    MeanHasher,
    MinHasher,
    NTSHashingBaseline,
    SHA256Hasher,
    SimHashBaseline,
    StdHasher,
)
from .common import hamming_distance
from .hasher import LoRAHash, LoraPerceptualHasher

__version__ = "1.0.0"

__all__ = [
    "AbsHasher",
    "HOSHasher",
    "LiuCVPR2025Baseline",
    "LoRAHash",
    "LoraPerceptualHasher",
    "MD5Hasher",
    "MaxHasher",
    "MeanHasher",
    "MinHasher",
    "NTSHashingBaseline",
    "REGISTRY",
    "SHA256Hasher",
    "SimHashBaseline",
    "StdHasher",
    "build_hasher",
    "hamming_distance",
]

#: ``name -> hasher class`` for LoRA-Hash and every baseline.
REGISTRY = {"lora": LoraPerceptualHasher, **_BASELINE_REGISTRY}


def build_hasher(name: str, device: str = "cuda", **kwargs):
    """Instantiate the hasher registered under ``name``.

    ``"lora"`` is LoRA-Hash itself; other names are the baselines listed in
    :data:`REGISTRY`.  ``"lora-nonorm"`` builds LoRA-Hash without per-layer
    Z-score standardisation (the corresponding ablation).
    """
    key = name.lower()
    if key == "lora-nonorm":
        return LoraPerceptualHasher(device=device, normalize=False, **kwargs)
    if key not in REGISTRY:
        raise ValueError(
            f"unknown hasher '{name}'. Available: "
            f"{', '.join(sorted(REGISTRY) + ['lora-nonorm'])}"
        )
    return REGISTRY[key](device=device, **kwargs)
