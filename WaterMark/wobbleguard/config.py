# watermark config and the dna alphabet constants.

from dataclasses import dataclass

# dna alphabet — this exact ordering is used everywhere
DNA_VOCAB: list[str] = ["A", "C", "G", "T"]
DNA_CHAR_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_DNA_CHAR: dict[int, str] = {0: "A", 1: "C", 2: "G", 3: "T"}
VOCAB_SIZE: int = 4


@dataclass(frozen=True)
class WatermarkConfig:
    # config for watermark generation and detection. immutable on purpose
    # so a detector can't accidentally drift from what the embedder used.
    #
    # key: secret key (>= 16 bytes). decides the green/red split at every
    # position — you need the same key to detect that you used to embed.
    # context_width: how many nucleotides (k) before the current position
    # feed the hash for the green list. k=4 means 256 possible contexts.
    # gamma: fraction of the vocab that lands in the green list at each
    # position. with vocab_size=4, gamma=0.5 is 2 green / 2 red per step.
    # delta: logit bias added to green tokens before sampling. bigger =
    # stronger watermark but worse quality, so it's the main tradeoff knob.
    # vocab_size: token alphabet size. always 4 for dna {A, C, G, T}.

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
        # how many green-list tokens per position
        return max(1, round(self.gamma * self.vocab_size))

    @property
    def n_red(self) -> int:
        # how many red-list tokens per position
        return self.vocab_size - self.n_green
