# closed-form theory for how detectable the dna watermark is.
# power analysis, minimum sequence length derivations, and quality-cost
# bounds for the green/red list scheme on the 4-token dna alphabet.
# everything here assumes the base distribution is roughly uniform or known.
# for non-uniform distributions (say at-biased viral genomes) these
# functions give conservative approximations.

import math
from typing import Optional

import numpy as np
from scipy.stats import norm


def p_green_under_watermark(
    delta: float,
    gamma: float,
    vocab_size: int = 4,
) -> float:
    # probability that a sampled token is green under the watermark.
    # with a uniform base distribution and logit bias delta on green tokens:
    #     P(green) = n_G * exp(delta) / (n_G * exp(delta) + n_R)
    # where n_G = round(gamma * V) and n_R = V - n_G.
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
    # expected z-score for a watermarked sequence with scored length L.
    #     z_expected = sqrt(L) * (p - gamma) / sqrt(gamma * (1 - gamma))
    # where p = P(green | watermark).
    p = p_green_under_watermark(delta, gamma, vocab_size)
    return math.sqrt(L) * (p - gamma) / math.sqrt(gamma * (1 - gamma))


def detection_power(
    L: int,
    delta: float,
    gamma: float,
    alpha: float = 0.01,
    vocab_size: int = 4,
) -> float:
    # statistical power (tpr) for detecting the watermark:
    #     power = Phi( (sqrt(L) * (p - gamma) - z_alpha * sqrt(gamma*(1-gamma)))
    #                 / sqrt(p * (1 - p)) )
    # L is the number of scored positions (length minus context width),
    # alpha is the significance level (fpr). returns the chance of correctly
    # flagging a watermarked sequence.
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
    # smallest scored length that hits the target power at the given fpr.
    # from the power equation:
    #     L_min = ceil( ((z_alpha * sqrt(gamma*(1-gamma)) + z_beta * sqrt(p*(1-p)))
    #                    / (p - gamma))^2 )
    # where z_alpha = Phi^{-1}(1 - alpha) and z_beta = Phi^{-1}(power).
    # if delta is too small it returns the big 999999 sentinel instead.
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
    # kl divergence from the watermarked distribution to uniform, in bits
    # per position:
    #     KL(P_wm || P_uniform) = sum_i P_wm(i) * log2(P_wm(i) / P_uniform(i))
    # this is the worst-case info-theoretic quality cost of the watermark
    # assuming a uniform base. real model distributions lose less because
    # the model's own preferences mask the watermark bias.
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
    # rough detection power after random point mutations.
    # a mutation at position t swaps the token for a uniform random
    # nucleotide, which hurts twice:
    # 1. the mutated position is now random (P(green) = gamma, not p).
    # 2. positions whose context window includes the mutation get a
    #    corrupted hash, so their green/red assignment is effectively random.
    # at point-mutation rate r with context width k, roughly
    # L * (1 - r) * (1 - r)^k positions keep their original watermark signal
    # and the rest are just noise (P(green) = gamma).
    # so this is a conservative (pessimistic) approximation.
    p = p_green_under_watermark(delta, gamma, vocab_size)
    # fraction of positions with intact watermark signal.
    # a position is intact if (1) it wasn't mutated AND (2) none of its k
    # context positions were. we assume independence to keep it simple.
    # k isn't passed in, so we just use 4.
    k = 4
    intact_frac = (1 - mutation_rate) ** (k + 1)

    # effective P(green) across all positions
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
    # build the summary table of theoretical predictions.
    # returns a list of dicts ready for csv export.
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
