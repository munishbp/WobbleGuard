# 5-tier evaluation engine for dna watermark detection.
# tier 1: synthetic validation (monte carlo)
# tier 2: negative control on existing evo2 output
# tier 3: negative control on natural ncbi pathogens
# tier 4: positive control on watermarked evo2 output
# tier 5: stress tests (mutations, truncation, chimeric, indels, synonymous subs)

import json
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

from .config import WatermarkConfig, DNA_CHAR_TO_IDX, IDX_TO_DNA_CHAR, VOCAB_SIZE
from .core import WatermarkDetector, WatermarkResult
from .genetic_code import WOBBLE_ALTERNATES, detect_reading_frame
from .utils import sequence_to_ids, ids_to_sequence


# -------------------------------------------------------------------------
# perturbation functions (tier 5)
# -------------------------------------------------------------------------

def apply_random_mutations(
    sequence: str,
    mutation_rate: float,
    rng: np.random.Generator,
) -> str:
    # apply uniform random point mutations: each position flips to a
    # uniformly random *different* nucleotide with probability mutation_rate.
    ids = sequence_to_ids(sequence)
    for i in range(len(ids)):
        if rng.random() < mutation_rate:
            # pick a different nucleotide
            alternatives = [x for x in range(VOCAB_SIZE) if x != ids[i]]
            ids[i] = int(rng.choice(alternatives))
    return ids_to_sequence(ids)


def apply_truncation(sequence: str, keep_fraction: float) -> str:
    # chop the sequence down to the first keep_fraction of its length
    keep_len = max(1, int(len(sequence) * keep_fraction))
    return sequence[:keep_len]


def apply_indels(
    sequence: str,
    indel_rate: float,
    rng: np.random.Generator,
) -> str:
    # random insertions and deletions at the given rate.
    # per position: with probability indel_rate/2 delete it, with
    # probability indel_rate/2 insert a random nucleotide before it.
    result = []
    for c in sequence:
        r = rng.random()
        if r < indel_rate / 2:
            # deletion: skip this character
            continue
        elif r < indel_rate:
            # insertion: add a random nucleotide before this one
            result.append(IDX_TO_DNA_CHAR[int(rng.integers(0, VOCAB_SIZE))])
        result.append(c)
    return "".join(result)


def apply_synonymous_substitutions(
    sequence: str,
    sub_rate: float,
    rng: np.random.Generator,
) -> str:
    # weaker-than-realistic wobble attack: mutate the 3rd nucleotide of each
    # codon to any uniformly-random different base. not guaranteed to be
    # synonymous. kept for backward compatibility with the existing tier 5
    # results; new work should use apply_true_synonymous_substitutions.
    ids = sequence_to_ids(sequence)
    # process in codons (triplets)
    for codon_start in range(0, len(ids) - 2, 3):
        wobble_pos = codon_start + 2
        if rng.random() < sub_rate:
            alternatives = [x for x in range(VOCAB_SIZE) if x != ids[wobble_pos]]
            ids[wobble_pos] = int(rng.choice(alternatives))
    return ids_to_sequence(ids)


def apply_true_synonymous_substitutions(
    sequence: str,
    sub_rate: float,
    rng: np.random.Generator,
    frame: int | None = None,
) -> str:
    # realistic wobble attack: swap only the third codon position to a
    # synonymous alternate from the genetic-code lookup, so the amino acid
    # sequence comes out exactly the same.
    # if frame is None we infer it with detect_reading_frame; sequences with
    # no long orf come back unchanged (no frame means the attacker can't do
    # a synonymous edit).
    # codons with a single synonym (Met AUG, Trp UGG, and any stops) never
    # get touched.
    seq = sequence.upper()
    if frame is None:
        frame = detect_reading_frame(seq)
        if frame is None:
            return seq

    out = list(seq)
    for cs in range(frame, len(seq) - 2, 3):
        if rng.random() >= sub_rate:
            continue
        codon = seq[cs : cs + 3]
        alternates = WOBBLE_ALTERNATES.get(codon)
        if not alternates or len(alternates) <= 1:
            continue
        choices = sorted(a for a in alternates if a != codon)
        if not choices:
            continue
        replacement = choices[int(rng.integers(0, len(choices)))]
        out[cs + 2] = replacement[2]
    return "".join(out)


def apply_chimeric_splice(
    seq_watermarked: str,
    seq_natural: str,
    splice_fraction: float = 0.5,
) -> str:
    # make a chimera: first chunk from the watermarked sequence, rest from
    # the natural one. splice_fraction = 0.5 means the first half is
    # watermarked and the second half is natural.
    splice_point = int(len(seq_watermarked) * splice_fraction)
    return seq_watermarked[:splice_point] + seq_natural[splice_point:min(len(seq_natural), len(seq_watermarked))]


# -------------------------------------------------------------------------
# evaluation functions
# -------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    # detection results after applying some perturbation.
    perturbation_type: str
    perturbation_param: float
    n_sequences: int
    n_trials: int
    detection_rate: float
    mean_z_score: float
    std_z_score: float


def evaluate_robustness_mutations(
    sequences: list[str],
    config: WatermarkConfig,
    mutation_rates: Optional[list[float]] = None,
    n_trials: int = 100,
    seed: int = 42,
    alpha: float = 0.01,
) -> list[RobustnessResult]:
    # detection rate after random point mutations
    if mutation_rates is None:
        mutation_rates = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    detector = WatermarkDetector(config, alpha=alpha)
    rng = np.random.default_rng(seed)
    results = []

    for rate in mutation_rates:
        z_scores = []
        detected = 0
        total = 0

        for seq in sequences:
            for _ in range(n_trials):
                if rate == 0.0:
                    mutated = seq
                else:
                    mutated = apply_random_mutations(seq, rate, rng)
                result = detector.detect(mutated)
                z_scores.append(result.z_score)
                if result.is_watermarked:
                    detected += 1
                total += 1

        z_arr = np.array(z_scores)
        results.append(RobustnessResult(
            perturbation_type="point_mutation",
            perturbation_param=rate,
            n_sequences=len(sequences),
            n_trials=n_trials,
            detection_rate=detected / total if total > 0 else 0.0,
            mean_z_score=float(z_arr.mean()),
            std_z_score=float(z_arr.std()),
        ))

    return results


def evaluate_robustness_truncation(
    sequences: list[str],
    config: WatermarkConfig,
    keep_fractions: Optional[list[float]] = None,
    alpha: float = 0.01,
) -> list[RobustnessResult]:
    # detection rate on truncated sequences
    if keep_fractions is None:
        keep_fractions = [1.0, 0.75, 0.50, 0.33, 0.25]

    detector = WatermarkDetector(config, alpha=alpha)
    results = []

    for frac in keep_fractions:
        z_scores = []
        detected = 0

        for seq in sequences:
            truncated = apply_truncation(seq, frac)
            if len(truncated) <= config.context_width:
                continue
            result = detector.detect(truncated)
            z_scores.append(result.z_score)
            if result.is_watermarked:
                detected += 1

        z_arr = np.array(z_scores) if z_scores else np.array([0.0])
        results.append(RobustnessResult(
            perturbation_type="truncation",
            perturbation_param=frac,
            n_sequences=len(sequences),
            n_trials=1,
            detection_rate=detected / len(z_scores) if z_scores else 0.0,
            mean_z_score=float(z_arr.mean()),
            std_z_score=float(z_arr.std()),
        ))

    return results


def evaluate_ncbi_negatives(
    ncbi_json_path: Path,
    config: WatermarkConfig,
    alpha: float = 0.01,
) -> dict:
    # run the detector on every ncbi natural pathogen fragment.
    # returns a dict with the overall fpr, per-virus fpr, max z-score,
    # and z-score stats.
    with open(ncbi_json_path) as f:
        fragments = json.load(f)

    detector = WatermarkDetector(config, alpha=alpha)

    all_z = []
    per_virus: dict[str, list[float]] = {}
    fp_count = 0

    for frag in fragments:
        seq = frag["sequence"]
        virus = frag["virus"]
        result = detector.detect(seq)

        all_z.append(result.z_score)
        per_virus.setdefault(virus, []).append(result.z_score)

        if result.is_watermarked:
            fp_count += 1

    n = len(all_z)
    z_arr = np.array(all_z)
    fpr = fp_count / n if n > 0 else 0.0

    # wilson confidence interval for the fpr
    if n > 0:
        z95 = 1.96
        p_hat = fpr
        denom = 1 + z95**2 / n
        center = (p_hat + z95**2 / (2 * n)) / denom
        spread = z95 * math.sqrt(p_hat * (1 - p_hat) / n + z95**2 / (4 * n**2)) / denom
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)
    else:
        ci_low = ci_high = 0.0

    # fpr per virus
    virus_fpr = {}
    for virus, z_list in per_virus.items():
        z_virus = np.array(z_list)
        threshold = float(norm.ppf(1 - alpha))
        virus_fp = int(np.sum(z_virus > threshold))
        virus_fpr[virus] = {
            "n_fragments": len(z_list),
            "false_positives": virus_fp,
            "fpr": virus_fp / len(z_list) if z_list else 0.0,
            "mean_z": float(z_virus.mean()),
            "max_z": float(z_virus.max()),
        }

    return {
        "total_fragments": n,
        "total_false_positives": fp_count,
        "overall_fpr": fpr,
        "fpr_95ci_low": ci_low,
        "fpr_95ci_high": ci_high,
        "mean_z": float(z_arr.mean()) if n > 0 else 0.0,
        "max_z": float(z_arr.max()) if n > 0 else 0.0,
        "std_z": float(z_arr.std()) if n > 0 else 0.0,
        "per_virus": virus_fpr,
        "all_z_scores": z_arr.tolist(),
    }


def evaluate_evo2_unwatermarked(
    json_path: Path,
    config: WatermarkConfig,
    alpha: float = 0.01,
) -> dict:
    # run the detector on all evo2 generations from a results json file
    with open(json_path) as f:
        data = json.load(f)

    detector = WatermarkDetector(config, alpha=alpha)

    z_scores = []
    fp_count = 0
    total = 0

    for entry in data:
        for gen in entry.get("generations", []):
            seq = gen.get("generated_dna", "")
            if len(seq) > config.context_width + 1:
                result = detector.detect(seq)
                z_scores.append(result.z_score)
                if result.is_watermarked:
                    fp_count += 1
                total += 1

    z_arr = np.array(z_scores) if z_scores else np.array([0.0])
    return {
        "total_sequences": total,
        "false_positives": fp_count,
        "fpr": fp_count / total if total > 0 else 0.0,
        "mean_z": float(z_arr.mean()),
        "std_z": float(z_arr.std()),
        "max_z": float(z_arr.max()),
        "z_scores": z_arr.tolist(),
    }
