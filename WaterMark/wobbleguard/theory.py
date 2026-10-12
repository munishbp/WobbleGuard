"""
Closed-form theoretical analysis of DNA watermark detectability.

Provides power analysis, minimum sequence length derivations, and
quality-cost bounds for the green/red list watermarking scheme on
the 4-token DNA alphabet.

All results assume the base distribution is approximately uniform
or known. For non-uniform distributions (e.g., AT-biased viral genomes),
the functions provide conservative approximations.
"""

import math
from typing import Optional

import numpy as np
from scipy.stats import norm


def p_green_under_watermark(
    delta: float,
    gamma: float,
    vocab_size: int = 4,
) -> float:
    """
    Probability that a sampled token is green under the watermark.

    Under a uniform base distribution with logit bias delta on green tokens:

        P(green) = n_G * exp(delta) / (n_G * exp(delta) + n_R)

    where n_G = round(gamma * V), n_R = V - n_G.

    Parameters
    ----------
    delta : float
        Logit bias added to green tokens.
    gamma : float
        Fraction of vocabulary in green list.
    vocab_size : int
        Vocabulary size (4 for DNA).

    Returns
    -------
    float
        P(green | watermark, uniform base).
    """
    n_green = max(1, round(gamma * vocab_size))
    n_red = vocab_size - n_green
    exp_delta = math.exp(delta)
    return (n_green * exp_delta) / (n_green * exp_delta + n_red)


def expected_z_score(
    L: int,
    delta: float,
    gamma: float,
    vocab_size: int = 4,
) -> float:
    """
    Expected z-score for a watermarked sequence of scored length L.

    z_expected = sqrt(L) * (p - gamma) / sqrt(gamma * (1 - gamma))

    where p = P(green | watermark).
    """
    p = p_green_under_watermark(delta, gamma, vocab_size)
    return math.sqrt(L) * (p - gamma) / math.sqrt(gamma * (1 - gamma))


def detection_power(
    L: int,
    delta: float,
    gamma: float,
    alpha: float = 0.01,
    vocab_size: int = 4,
) -> float:
    """
    Statistical power (TPR) for detecting the watermark.

    Power = Phi( (sqrt(L) * (p - gamma) - z_alpha * sqrt(gamma*(1-gamma)))
                 / sqrt(p * (1 - p)) )

    Parameters
    ----------
    L : int
        Number of scored positions (sequence length minus context width).
    delta : float
        Logit bias.
    gamma : float
        Green-list fraction.
    alpha : float
        Significance level (FPR).
    vocab_size : int
        Vocabulary size.

    Returns
    -------
    float
        Detection power (probability of correctly flagging a watermarked sequence).
    """
    z_alpha = norm.ppf(1 - alpha)
    p = p_green_under_watermark(delta, gamma, vocab_size)

    if p <= gamma:
        return 0.0

    numerator = math.sqrt(L) * (p - gamma) - z_alpha * math.sqrt(gamma * (1 - gamma))
    denominator = math.sqrt(p * (1 - p))

    return float(norm.cdf(numerator / denominator))


def min_sequence_length(
    delta: float,
    gamma: float,
    alpha: float = 0.01,
    power: float = 0.95,
    vocab_size: int = 4,
) -> int:
    """
    Minimum scored sequence length for target detection power at given FPR.

    Derived from the power equation:

        L_min = ceil( ((z_alpha * sqrt(gamma*(1-gamma)) + z_beta * sqrt(p*(1-p)))
                       / (p - gamma))^2 )

    where z_alpha = Phi^{-1}(1 - alpha), z_beta = Phi^{-1}(power).

    Returns
    -------
    int
        Minimum L. Returns a large sentinel (999999) if delta is too small.
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    p = p_green_under_watermark(delta, gamma, vocab_size)

    if p <= gamma:
        return 999999

    numerator = z_alpha * math.sqrt(gamma * (1 - gamma)) + z_beta * math.sqrt(p * (1 - p))
    denominator = p - gamma

    return math.ceil((numerator / denominator) ** 2)


def entropy_loss_per_position(
    delta: float,
    gamma: float,
    vocab_size: int = 4,
) -> float:
    """
    KL divergence from watermarked to uniform distribution (bits per position).

    KL(P_wm || P_uniform) = sum_i P_wm(i) * log2(P_wm(i) / P_uniform(i))

    This is the worst-case information-theoretic quality cost of the watermark
    assuming a uniform base distribution. Real model distributions have lower
    KL because the watermark bias is masked by the model's own preferences.

    Returns
    -------
    float
        KL divergence in bits.
    """
    n_green = max(1, round(gamma * vocab_size))
    n_red = vocab_size - n_green
    p = p_green_under_watermark(delta, gamma, vocab_size)

    p_each_green = p / n_green
    p_each_red = (1 - p) / n_red if n_red > 0 else 0.0
    p_uniform = 1.0 / vocab_size

    kl = 0.0
    kl += n_green * p_each_green * math.log2(p_each_green / p_uniform)
    if n_red > 0 and p_each_red > 0:
        kl += n_red * p_each_red * math.log2(p_each_red / p_uniform)

    return kl


def power_after_mutations(
    L: int,
    delta: float,
    gamma: float,
    mutation_rate: float,
    alpha: float = 0.01,
    vocab_size: int = 4,
) -> float:
    """
    Approximate detection power after random point mutations.

    A mutation at position t replaces the token with a uniform random
    nucleotide. This affects detection in two ways:
    1. The mutated position is now random (P(green) = gamma, not p).
    2. Positions whose context window includes the mutation have a
       corrupted hash (effectively random green/red assignment).

    For point mutations at rate r with context width k, approximately
    L * (1 - r) * (1 - r)^k positions retain their original watermark signal.
    The remaining positions contribute noise (P(green) = gamma).

    This is a conservative (pessimistic) approximation.
    """
    p = p_green_under_watermark(delta, gamma, vocab_size)
    # Fraction of positions with intact watermark signal
    # A position is intact if: (1) not mutated AND (2) none of its k context positions mutated
    # Approximation: assume independence
    # k is not passed here; use a reasonable default of 4
    k = 4
    intact_frac = (1 - mutation_rate) ** (k + 1)

    # Effective P(green) across all positions
    p_effective = intact_frac * p + (1 - intact_frac) * gamma

    if p_effective <= gamma:
        return 0.0

    z_alpha = norm.ppf(1 - alpha)
    numerator = math.sqrt(L) * (p_effective - gamma) - z_alpha * math.sqrt(gamma * (1 - gamma))
    denominator = math.sqrt(p_effective * (1 - p_effective))

    return float(norm.cdf(numerator / denominator))


def generate_theory_table(
    deltas: Optional[list[float]] = None,
    gammas: Optional[list[float]] = None,
    L: int = 636,
    alpha: float = 0.01,
) -> list[dict]:
    """
    Generate a summary table of theoretical predictions.

    Returns a list of dicts suitable for CSV export.
    """
    if deltas is None:
        deltas = [0.5, 1.0, 1.5, 2.0]
    if gammas is None:
        gammas = [0.25, 0.5]

    rows = []
    for delta in deltas:
        for gamma in gammas:
            p = p_green_under_watermark(delta, gamma)
            rows.append({
                "delta": delta,
                "gamma": gamma,
                "p_green": round(p, 4),
                "expected_z_at_L": round(expected_z_score(L, delta, gamma), 2),
                "power_at_L": round(detection_power(L, delta, gamma, alpha), 4),
                "L_min_95": min_sequence_length(delta, gamma, alpha, 0.95),
                "kl_bits_per_pos": round(entropy_loss_per_position(delta, gamma), 4),
            })
    return rows
