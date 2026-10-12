"""
Watermark configuration and DNA alphabet constants.
"""

from dataclasses import dataclass

# DNA alphabet — canonical ordering used throughout
DNA_VOCAB: list[str] = ["A", "C", "G", "T"]
DNA_CHAR_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_DNA_CHAR: dict[int, str] = {0: "A", 1: "C", 2: "G", 3: "T"}
VOCAB_SIZE: int = 4


@dataclass(frozen=True)
class WatermarkConfig:
    """
    Immutable configuration for DNA watermark generation and detection.

    Parameters
    ----------
    key : bytes
        Secret key (>= 16 bytes). Determines the green/red partition at each
        position. Detection requires the same key used during generation.
    context_width : int
        Number of preceding nucleotides (k) used to seed the hash for
        green-list computation. k=4 gives 256 possible contexts.
    gamma : float
        Fraction of the vocabulary placed in the green list at each position.
        With vocab_size=4, gamma=0.5 means 2 green / 2 red tokens per step.
    delta : float
        Logit bias added to green-list tokens before sampling. Controls the
        strength of the watermark signal vs. quality degradation tradeoff.
    vocab_size : int
        Size of the token alphabet. Always 4 for DNA {A, C, G, T}.
    """

    key: bytes
    context_width: int = 4
    gamma: float = 0.5
    delta: float = 1.5
    vocab_size: int = 4

    def __post_init__(self) -> None:
        if len(self.key) < 16:
            raise ValueError(f"Key must be >= 16 bytes, got {len(self.key)}")
        if not (0 < self.gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.delta <= 0:
            raise ValueError(f"delta must be > 0, got {self.delta}")
        if self.context_width < 1:
            raise ValueError(f"context_width must be >= 1, got {self.context_width}")

    @property
    def n_green(self) -> int:
        """Number of green-list tokens per position."""
        return max(1, round(self.gamma * self.vocab_size))

    @property
    def n_red(self) -> int:
        """Number of red-list tokens per position."""
        return self.vocab_size - self.n_green
