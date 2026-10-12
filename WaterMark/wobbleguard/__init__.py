# dna watermarking for bio-foundation models.
# zero-training, hash-based green/red list watermark for dna sequences.
# adapted from kirchenbauer et al. (icml 2023) for the 4-token dna alphabet.
# part of: persaud et al., "on the dangers of open-source bio-foundation models"

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
