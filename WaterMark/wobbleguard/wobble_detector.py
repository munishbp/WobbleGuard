# wobble-aware watermark detector: a detect-time defense against synonymous
# codon substitution.
# the embedder is plain vanilla kgw-dna (see core.py), unchanged. at
# detection, for each scored position we enumerate the small number of
# plausible context de-swap patterns the genetic code allows (only
# third-position wobble substitutions get enumerated), take the union of the
# resulting candidate green lists, and flag the position if the observed
# nucleotide or any of its wobble-synonymous alternates lands in that union.
# the null rate at each position comes analytically from that position's own
# enumeration, so the z-test stays well-calibrated.
# non-coding sequences (no long orf in any frame) just fall back to the
# vanilla detector.

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from scipy.stats import norm

from .config import DNA_CHAR_TO_IDX, WatermarkConfig
from .core import WatermarkDetector, WatermarkResult
from .genetic_code import (
    NUCLEOTIDES,
    detect_reading_frame,
    wobble_alternates_at_position,
)
from .utils import get_green_list


@dataclass
class WobbleDetectionResult:
    # detection result from the wobble-aware detector.

    num_flagged: int
    total_scored: int
    null_expected: float
    z_score: float
    p_value: float
    is_watermarked: bool
    flag_fraction: float
    threshold_z: float
    frame_used: int | None
    fell_back_to_vanilla: bool

    @classmethod
    def from_vanilla(cls, r: WatermarkResult) -> "WobbleDetectionResult":
        return cls(
            num_flagged=r.num_green,
            total_scored=r.total_scored,
            null_expected=r.total_scored * 0.5,
            z_score=r.z_score,
            p_value=r.p_value,
            is_watermarked=r.is_watermarked,
            flag_fraction=r.green_fraction,
            threshold_z=r.threshold_z,
            frame_used=None,
            fell_back_to_vanilla=True,
        )


def _enumerate_context_candidates(
    seq: str, context_start: int, k: int, frame: int
) -> list[str]:
    # every plausible context string the embedder could have used, given that
    # the adversary may have wobble-swapped zero or more positions in the
    # context window. non-wobble positions each contribute a single candidate.
    per_position: list[list[str]] = []
    for i in range(k):
        pos = context_start + i
        alts = wobble_alternates_at_position(seq, pos, frame)
        per_position.append(sorted(alts))
    return ["".join(p) for p in itertools.product(*per_position)]


def _wobble_class_ids(seq: str, t: int, frame: int, obs: str) -> frozenset[int]:
    # the wobble equivalence class at position `t` if we pretend the observed
    # base were `obs`. returns nucleotide indices, not characters.
    hypothetical = seq[:t] + obs + seq[t + 1 :]
    alts = wobble_alternates_at_position(hypothetical, t, frame)
    return frozenset(DNA_CHAR_TO_IDX[c] for c in alts)


class WobbleAwareDetector:
    # detect-time defense that shrugs off the synonymous-substitution attack
    # by enumerating context de-swap candidates and widening the acceptance
    # set to wobble-equivalent observations.
    # it reuses watermark.utils.get_green_list — the hashing and green-list
    # split are exactly the same as vanilla kgw-dna.

    def __init__(self, config: WatermarkConfig, alpha: float = 0.01) -> None:
        self.config = config
        self.alpha = alpha
        self.z_threshold = norm.ppf(1 - alpha)
        self._vanilla = WatermarkDetector(config, alpha=alpha)

    def detect(self, sequence: str) -> WobbleDetectionResult:
        seq = sequence.upper()
        frame = detect_reading_frame(seq)
        if frame is None:
            return WobbleDetectionResult.from_vanilla(self._vanilla.detect(seq))

        k = self.config.context_width
        V = self.config.vocab_size
        if len(seq) <= k:
            return WobbleDetectionResult(
                num_flagged=0, total_scored=0, null_expected=0.0,
                z_score=0.0, p_value=1.0, is_watermarked=False,
                flag_fraction=0.0, threshold_z=self.z_threshold,
                frame_used=frame, fell_back_to_vanilla=False,
            )

        flagged = 0
        null_sum = 0.0
        null_var_sum = 0.0
        total = 0

        # per-context memoization: candidate-context strings -> G* as a frozenset of ids
        g_union_cache: dict[tuple[str, ...], frozenset[int]] = {}

        for t in range(k, len(seq)):
            ctx_start = t - k
            candidates = _enumerate_context_candidates(seq, ctx_start, k, frame)
            cache_key = tuple(candidates)

            g_union = g_union_cache.get(cache_key)
            if g_union is None:
                accum: set[int] = set()
                for cand in candidates:
                    cand_ids = [DNA_CHAR_TO_IDX[c] for c in cand]
                    accum |= get_green_list(cand_ids, self.config)
                g_union = frozenset(accum)
                g_union_cache[cache_key] = g_union

            # observed flag: does obs[t]'s wobble class intersect G*?
            obs_class = _wobble_class_ids(seq, t, frame, seq[t])
            is_flagged = bool(obs_class & g_union)

            # position-specific null rate: average intersection indicator over x in V
            gamma_t = 0.0
            for hypo in NUCLEOTIDES:
                x_class = _wobble_class_ids(seq, t, frame, hypo)
                if x_class & g_union:
                    gamma_t += 1.0
            gamma_t /= V

            if is_flagged:
                flagged += 1
            null_sum += gamma_t
            null_var_sum += gamma_t * (1.0 - gamma_t)
            total += 1

        z_score = (flagged - null_sum) / math.sqrt(null_var_sum) if null_var_sum > 0 else 0.0
        p_value = 1.0 - norm.cdf(z_score)

        return WobbleDetectionResult(
            num_flagged=flagged,
            total_scored=total,
            null_expected=null_sum,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=(z_score > self.z_threshold),
            flag_fraction=flagged / total if total > 0 else 0.0,
            threshold_z=self.z_threshold,
            frame_used=frame,
            fell_back_to_vanilla=False,
        )

    def detect_batch(self, sequences: list[str]) -> list[WobbleDetectionResult]:
        return [self.detect(s) for s in sequences]
