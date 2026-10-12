"""
Core watermark engine: logits processor (embedding) and detector (post-hoc).

Implements the green/red list watermarking scheme of Kirchenbauer et al.
(ICML 2023), adapted for the 4-token DNA alphabet {A, C, G, T}.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .config import WatermarkConfig
from .utils import get_green_list, sequence_to_ids


class WatermarkLogitsProcessor:
    """
    Zero-training logits processor that embeds a watermark during generation.

    Drop-in wrapper for any DNA language model's sampling loop. At each
    autoregressive step, adds a logit bias delta to green-list tokens
    (determined by a hash of the secret key and the preceding k nucleotides).

    Usage::

        processor = WatermarkLogitsProcessor(config)
        for step in range(seq_length):
            logits = model.forward(input_ids)[-1]           # shape (4,)
            logits = processor(input_ids.tolist(), logits)   # apply watermark
            token = sample(logits)
            input_ids.append(token)
    """

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def __call__(
        self,
        input_ids: list[int],
        logits: np.ndarray,
    ) -> np.ndarray:
        """
        Apply watermark bias to logits at the current generation step.

        Parameters
        ----------
        input_ids : list[int]
            All token IDs generated so far (prompt + generated). Must have at
            least ``config.context_width`` elements for the watermark to engage.
        logits : np.ndarray
            Raw model logits, shape ``(vocab_size,)`` = ``(4,)``.
            Modified **in-place** and returned.

        Returns
        -------
        np.ndarray
            Modified logits with delta added to green-list positions.
            Returns unmodified logits if context is too short.
        """
        k = self.config.context_width
        if len(input_ids) < k:
            return logits

        context = input_ids[-k:]
        green_set = get_green_list(context, self.config)

        for token_id in green_set:
            logits[token_id] += self.config.delta

        return logits


@dataclass
class WatermarkResult:
    """Detection result for a single DNA sequence."""

    num_green: int
    """Count of scored positions where the token was in the green list."""

    total_scored: int
    """Total positions scored (sequence length minus context width k)."""

    z_score: float
    """
    Standard z-score under H0: P(green) = gamma.
    z = (num_green - T*gamma) / sqrt(T*gamma*(1-gamma))
    """

    p_value: float
    """One-sided p-value: P(Z >= z_score | H0)."""

    is_watermarked: bool
    """Verdict: z_score > z_threshold at the configured alpha level."""

    green_fraction: float
    """Observed fraction of green tokens: num_green / total_scored."""

    threshold_z: float
    """The z-score threshold used for the verdict (determined by alpha)."""


class WatermarkDetector:
    """
    Statistical detector for the green/red list DNA watermark.

    Given a DNA sequence and the secret key, counts how many positions
    fall on their context-dependent green list, then performs a one-sided
    one-proportion z-test against H0: P(green) = gamma.

    Detection requires **only the sequence and the secret key** — no model
    access, no logits. A DNA synthesis provider could run this detector
    without Evo2.

    Parameters
    ----------
    config : WatermarkConfig
        Must use the same key, context_width, and gamma as the processor.
    alpha : float
        Significance level for the hypothesis test. Default 0.01 gives
        FPR <= 1% on natural (unwatermarked) sequences.
    """

    def __init__(self, config: WatermarkConfig, alpha: float = 0.01) -> None:
        self.config = config
        self.alpha = alpha
        self.z_threshold = norm.ppf(1 - alpha)

    def detect(self, sequence: str) -> WatermarkResult:
        """
        Run watermark detection on a DNA sequence.

        Parameters
        ----------
        sequence : str
            DNA string of any length (must be > context_width).

        Returns
        -------
        WatermarkResult
            Contains z-score, p-value, and binary verdict.
        """
        ids = sequence_to_ids(sequence)
        k = self.config.context_width
        gamma = self.config.gamma

        if len(ids) <= k:
            return WatermarkResult(
                num_green=0,
                total_scored=0,
                z_score=0.0,
                p_value=1.0,
                is_watermarked=False,
                green_fraction=0.0,
                threshold_z=self.z_threshold,
            )

        num_green = 0
        total_scored = 0

        for t in range(k, len(ids)):
            context = ids[t - k : t]
            green_set = get_green_list(context, self.config)
            if ids[t] in green_set:
                num_green += 1
            total_scored += 1

        # One-proportion z-test
        # H0: P(green) = gamma   H1: P(green) > gamma
        expected = total_scored * gamma
        std = (total_scored * gamma * (1 - gamma)) ** 0.5

        if std > 0:
            z_score = (num_green - expected) / std
        else:
            z_score = 0.0

        p_value = 1.0 - norm.cdf(z_score)
        green_fraction = num_green / total_scored if total_scored > 0 else 0.0

        return WatermarkResult(
            num_green=num_green,
            total_scored=total_scored,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=(z_score > self.z_threshold),
            green_fraction=green_fraction,
            threshold_z=self.z_threshold,
        )

    def detect_batch(self, sequences: list[str]) -> list[WatermarkResult]:
        """Run detection on multiple sequences."""
        return [self.detect(seq) for seq in sequences]
