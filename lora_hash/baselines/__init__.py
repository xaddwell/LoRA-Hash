"""Baseline hashers used in the LoRA-Hash comparison.

Every class exposes the same minimal interface::

    hasher = SomeHasher(device="cuda")
    h1 = hasher.generate_hash("a.safetensors")
    h2 = hasher.generate_hash("b.safetensors")
    d  = hasher.calculate_hamming_distance(h1, h2)   # in [0, 1]

``REGISTRY`` maps the short names used on the command line of
``scripts/hash_variants.py`` to the corresponding classes.
"""

from .ablations import (
    AbsHasher,
    MaxHasher,
    MeanHasher,
    MinHasher,
    StdHasher,
)
from .crypto import MD5Hasher, SHA256Hasher
from .hos import HOSHasher
from .liu_cvpr2025 import LiuCVPR2025Baseline
from .nts import NTSHashingBaseline
from .simhash import SimHashBaseline

__all__ = [
    "AbsHasher",
    "HOSHasher",
    "LiuCVPR2025Baseline",
    "MD5Hasher",
    "MaxHasher",
    "MeanHasher",
    "MinHasher",
    "NTSHashingBaseline",
    "SHA256Hasher",
    "SimHashBaseline",
    "StdHasher",
    "REGISTRY",
]

#: ``name -> hasher class`` for every baseline plus LoRA-Hash itself.
REGISTRY = {
    "sim": SimHashBaseline,
    "nts": NTSHashingBaseline,
    "cvpr": LiuCVPR2025Baseline,
    "hos": HOSHasher,
    "mean": MeanHasher,
    "max": MaxHasher,
    "min": MinHasher,
    "abs": AbsHasher,
    "std": StdHasher,
    "sha256": SHA256Hasher,
    "md5": MD5Hasher,
}
