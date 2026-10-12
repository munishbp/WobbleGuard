"""
Hash-based green-list computation and sequence encoding utilities.
"""

import hashlib
import struct
import math
from typing import Optional

import numpy as np

from .config import WatermarkConfig, DNA_CHAR_TO_IDX, IDX_TO_DNA_CHAR, VOCAB_SIZE


def _context_hash(context_ids: list[int], key: bytes) -> int:
    """
    Hash preceding k token IDs with the secret key to produce a 64-bit seed.

    Uses SHA-256(key || context_bytes), truncated to the first 8 bytes
    and interpreted as a little-endian uint64. SHA-256 modeled as a random
    oracle ensures the green/red partition is uniform and unpredictable
    without the key.
    """
    payload = key + bytes(context_ids)
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<Q", digest[:8])[0]


def get_green_list(context_ids: list[int], config: WatermarkConfig) -> set[int]:
    """
    Compute the green-list token indices for the current position.

    Given the preceding k token IDs and the secret key, deterministically
    partitions {0, 1, 2, 3} into green (favored) and red (disfavored) sets.

    The hash seeds a PRNG that generates a permutation of the vocabulary;
    the first ceil(gamma * V) elements of the permutation form the green list.

    Parameters
    ----------
    context_ids : list[int]
        The preceding k token IDs. Must have exactly config.context_width elements.
    config : WatermarkConfig
        Watermark configuration (key, gamma, vocab_size).

    Returns
    -------
    set[int]
        Token indices in the green list.
    """
    seed = _context_hash(context_ids, config.key)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(config.vocab_size)
    return set(perm[: config.n_green].tolist())


def sequence_to_ids(seq: str) -> list[int]:
    """Convert a DNA string (e.g. 'ACGT') to a list of integer token IDs."""
    return [DNA_CHAR_TO_IDX[c] for c in seq.upper()]


def ids_to_sequence(ids: list[int]) -> str:
    """Convert a list of integer token IDs back to a DNA string."""
    return "".join(IDX_TO_DNA_CHAR[i] for i in ids)
