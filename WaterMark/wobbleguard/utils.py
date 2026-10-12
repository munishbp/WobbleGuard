# hash-based green-list computation plus small sequence encoding helpers.

import hashlib
import struct
import math
from typing import Optional

import numpy as np

from .config import WatermarkConfig, DNA_CHAR_TO_IDX, IDX_TO_DNA_CHAR, VOCAB_SIZE


def _context_hash(context_ids: list[int], key: bytes) -> int:
    # hash the preceding k token ids with the secret key, get a 64-bit seed.
    # it's sha-256(key || context_bytes), we take the first 8 bytes and read
    # them as a little-endian uint64. treating sha-256 as a random oracle
    # is what makes the green/red split uniform and unguessable without the key.
    payload = key + bytes(context_ids)
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<Q", digest[:8])[0]


def get_green_list(context_ids: list[int], config: WatermarkConfig) -> set[int]:
    # work out the green-list token ids for the current position.
    # given the previous k token ids and the secret key, split {0,1,2,3}
    # into green (favored) and red (disfavored) deterministically.
    # the hash seeds a prng that shuffles the vocab, and the first
    # ceil(gamma * V) elements of that shuffle are the green list.
    # context_ids must have exactly config.context_width elements.
    seed = _context_hash(context_ids, config.key)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(config.vocab_size)
    return set(perm[: config.n_green].tolist())


def sequence_to_ids(seq: str) -> list[int]:
    # turn a dna string like 'acgt' into a list of integer token ids
    return [DNA_CHAR_TO_IDX[c] for c in seq.upper()]


def ids_to_sequence(ids: list[int]) -> str:
    # turn a list of integer token ids back into a dna string
    return "".join(IDX_TO_DNA_CHAR[i] for i in ids)
