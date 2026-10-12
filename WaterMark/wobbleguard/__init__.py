"""
DNA Watermarking for Bio-Foundation Models
==========================================

Zero-training, hash-based green/red list watermark for DNA sequences.
Adapted from Kirchenbauer et al. (ICML 2023) for the 4-token DNA alphabet.

Part of: Persaud et al., "On The Dangers of Open-Source Bio-Foundation Models"
"""

from .config import WatermarkConfig, DNA_VOCAB, DNA_CHAR_TO_IDX, IDX_TO_DNA_CHAR, VOCAB_SIZE
from .core import WatermarkLogitsProcessor, WatermarkDetector, WatermarkResult
from .utils import get_green_list, sequence_to_ids, ids_to_sequence

__all__ = [
    "WatermarkConfig",
    "WatermarkLogitsProcessor",
    "WatermarkDetector",
    "WatermarkResult",
    "get_green_list",
    "sequence_to_ids",
    "ids_to_sequence",
    "DNA_VOCAB",
    "DNA_CHAR_TO_IDX",
    "IDX_TO_DNA_CHAR",
    "VOCAB_SIZE",
]
