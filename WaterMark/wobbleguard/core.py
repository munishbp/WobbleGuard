# core watermark engine: logits processor (embedding) and detector (post-hoc).
# green/red list scheme from kirchenbauer et al. (icml 2023), adapted for
# the 4-token dna alphabet {A, C, G, T}.

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .config import WatermarkConfig
from .utils import get_green_list, sequence_to_ids


class WatermarkLogitsProcessor:
    # zero-training logits processor that stamps the watermark in during
    # generation. it's a drop-in wrapper for any dna model's sampling loop:
    # at each step it adds a logit bias delta to the green-list tokens
    # (picked by hashing the secret key with the previous k nucleotides).
    #
    # usage:
    #     processor = WatermarkLogitsProcessor(config)
    #     for step in range(seq_length):
    #         logits = model.forward(input_ids)[-1]           # shape (4,)
    #         logits = processor(input_ids.tolist(), logits)  # apply watermark
    #         token = sample(logits)
    #         input_ids.append(token)

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def __call__(
        self,
        input_ids: list[int],
        logits: np.ndarray,
    ) -> np.ndarray:
        # apply the watermark bias to this step's logits.
        # input_ids is every token id so far (prompt + generated) and needs
        # at least config.context_width elements before the watermark kicks in.
        # logits is the raw model output, shape (vocab_size,) = (4,); it's
        # modified in-place and returned.
        # if the context is too short, the logits come back untouched.
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
    # detection result for one dna sequence.

    num_green: int
    # how many scored positions had a green-list token

    total_scored: int
    # positions scored overall (length minus the k context)

    z_score: float
    # standard z-score under h0: P(green) = gamma.
    # z = (num_green - T*gamma) / sqrt(T*gamma*(1-gamma))

    p_value: float
    # one-sided p-value: P(Z >= z_score | H0)

    is_watermarked: bool
    # verdict: z_score > z_threshold at the configured alpha

    green_fraction: float
    # observed fraction of green tokens: num_green / total_scored

    threshold_z: float
    # the z-score cutoff the verdict used (set by alpha)


class WatermarkDetector:
    # statistical detector for the green/red list dna watermark.
    # given a dna sequence and the secret key, count how many positions land
    # on their context-dependent green list, then run a one-sided
    # one-proportion z-test against h0: P(green) = gamma.
    # you only need the sequence and the key — no model access, no logits.
    # a dna synthesis provider could run this without evo2.
    # config must use the same key, context_width, and gamma as the processor
    # used to embed. alpha is the significance level; the default 0.01 gives
    # FPR <= 1% on natural (unwatermarked) sequences.

    def __init__(self, config: WatermarkConfig, alpha: float = 0.01) -> None:
        self.config = config
        self.alpha = alpha
        self.z_threshold = norm.ppf(1 - alpha)

    def detect(self, sequence: str) -> WatermarkResult:
        # run watermark detection on a dna sequence of any length
        # (it just has to be longer than context_width). returns a
        # WatermarkResult with the z-score, p-value, and the verdict.
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

        # one-proportion z-test
        # h0: P(green) = gamma   h1: P(green) > gamma
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
        # run detection on a bunch of sequences
        return [self.detect(seq) for seq in sequences]
