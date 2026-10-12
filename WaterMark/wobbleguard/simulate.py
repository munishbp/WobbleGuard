# monte carlo simulation engine for validating the watermark detection stats.
# makes synthetic watermarked and unwatermarked dna sequences, runs the
# detector on both, then compares the empirical tpr, fpr, auc, and z-score
# distributions against the theoretical predictions.

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import WatermarkConfig, VOCAB_SIZE
from .core import WatermarkLogitsProcessor, WatermarkDetector, WatermarkResult
from .utils import ids_to_sequence


@dataclass
class SimulationResult:
    # results from one monte carlo detection experiment.

    config_label: str
    n_watermarked: int
    n_unwatermarked: int
    true_positive_rate: float
    false_positive_rate: float
    auc: float
    z_scores_watermarked: np.ndarray
    z_scores_unwatermarked: np.ndarray
    mean_z_watermarked: float
    mean_z_unwatermarked: float


# observed evo2 nucleotide frequencies (from 7B verified jailbreak sequences)
EVO2_BASE_LOGITS = np.log(np.array([0.289, 0.206, 0.220, 0.285]))  # A, C, G, T


def generate_synthetic_sequence(
    config: Optional[WatermarkConfig],
    length: int = 640,
    base_logits: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> str:
    # generate one synthetic dna sequence.
    # if config is given, the watermark processor runs during generation;
    # if it's None, you get an unwatermarked sequence (the null hypothesis).
    # length is the number of nucleotides. base_logits is the base
    # log-probs for {A, C, G, T} — defaults to uniform if None. rng is
    # the random generator; we make one if you don't pass it.
    if rng is None:
        rng = np.random.default_rng()
    if base_logits is None:
        base_logits = np.log(np.ones(VOCAB_SIZE) / VOCAB_SIZE)

    processor = WatermarkLogitsProcessor(config) if config is not None else None
    k = config.context_width if config is not None else 4

    # random seed prefix (not part of the output sequence)
    input_ids = rng.integers(0, VOCAB_SIZE, size=k).tolist()

    for _ in range(length):
        logits = base_logits.copy()
        if processor is not None:
            logits = processor(input_ids, logits)

        # softmax sampling
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        token = int(rng.choice(VOCAB_SIZE, p=probs))
        input_ids.append(token)

    return ids_to_sequence(input_ids[k:])


def run_monte_carlo(
    config: WatermarkConfig,
    n_watermarked: int = 10_000,
    n_unwatermarked: int = 10_000,
    seq_length: int = 640,
    base_logits: Optional[np.ndarray] = None,
    seed: int = 42,
    alpha: float = 0.01,
) -> SimulationResult:
    # full monte carlo validation of the detection stats: generate
    # watermarked and unwatermarked sequences, run the detector on all of
    # them, and return the z-score distributions, tpr, fpr, and auc.
    rng = np.random.default_rng(seed)
    detector = WatermarkDetector(config, alpha=alpha)

    z_wm = np.zeros(n_watermarked)
    z_null = np.zeros(n_unwatermarked)

    for i in range(n_watermarked):
        seq = generate_synthetic_sequence(
            config, seq_length, base_logits,
            np.random.default_rng(rng.integers(0, 2**32)),
        )
        z_wm[i] = detector.detect(seq).z_score

    for i in range(n_unwatermarked):
        seq = generate_synthetic_sequence(
            None, seq_length, base_logits,
            np.random.default_rng(rng.integers(0, 2**32)),
        )
        z_null[i] = detector.detect(seq).z_score

    threshold = detector.z_threshold
    tpr = float(np.mean(z_wm > threshold))
    fpr = float(np.mean(z_null > threshold))

    # auc
    labels = np.concatenate([np.ones(n_watermarked), np.zeros(n_unwatermarked)])
    scores = np.concatenate([z_wm, z_null])
    auc = float(roc_auc_score(labels, scores))

    label = f"delta={config.delta},gamma={config.gamma},k={config.context_width}"

    return SimulationResult(
        config_label=label,
        n_watermarked=n_watermarked,
        n_unwatermarked=n_unwatermarked,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        auc=auc,
        z_scores_watermarked=z_wm,
        z_scores_unwatermarked=z_null,
        mean_z_watermarked=float(z_wm.mean()),
        mean_z_unwatermarked=float(z_null.mean()),
    )


def sweep_parameters(
    key: bytes,
    deltas: Optional[list[float]] = None,
    gammas: Optional[list[float]] = None,
    context_widths: Optional[list[int]] = None,
    n_samples: int = 2_000,
    seq_length: int = 640,
    base_logits: Optional[np.ndarray] = None,
    seed: int = 42,
) -> dict[str, SimulationResult]:
    # sweep over (delta, gamma, k) and collect detection stats for each combo.
    # returns a dict keyed by the config label string -> SimulationResult.
    if deltas is None:
        deltas = [0.5, 1.0, 1.5, 2.0]
    if gammas is None:
        gammas = [0.25, 0.5]
    if context_widths is None:
        context_widths = [1, 4]

    results = {}
    for delta in deltas:
        for gamma in gammas:
            for k in context_widths:
                config = WatermarkConfig(
                    key=key, context_width=k, gamma=gamma, delta=delta,
                )
                label = f"delta={delta},gamma={gamma},k={k}"
                print(f"  Simulating {label} ({n_samples} per class)...")
                result = run_monte_carlo(
                    config, n_samples, n_samples, seq_length, base_logits, seed,
                )
                results[label] = result
    return results
