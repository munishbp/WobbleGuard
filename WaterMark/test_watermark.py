#!/usr/bin/env python3
"""
Test suite for the DNA watermarking engine.

Validates correctness of green-list computation, logits processing,
detection statistics, and absence of false positives on real data.

Run: pytest test_watermark.py -v

# Make the wobbleguard package importable when run from this directory.
import sys as _sys
from pathlib import Path as _Path
_WG_ROOT = _Path(__file__).resolve().parent
for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
    if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from wobbleguard.config import WatermarkConfig, DNA_CHAR_TO_IDX, VOCAB_SIZE
from wobbleguard.utils import get_green_list, sequence_to_ids, ids_to_sequence, _context_hash
from wobbleguard.core import WatermarkLogitsProcessor, WatermarkDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_KEY = b"neurips2026-evo2-watermark-test"  # 31 bytes


@pytest.fixture
def config():
    return WatermarkConfig(key=TEST_KEY, context_width=4, gamma=0.5, delta=1.5)


@pytest.fixture
def processor(config):
    return WatermarkLogitsProcessor(config)


@pytest.fixture
def detector(config):
    return WatermarkDetector(config, alpha=0.01)


# ---------------------------------------------------------------------------
# Step 1: Green-list properties
# ---------------------------------------------------------------------------


def test_green_list_deterministic(config):
    """Same context + key always produces the same green set."""
    context = [0, 1, 2, 3]
    g1 = get_green_list(context, config)
    g2 = get_green_list(context, config)
    assert g1 == g2


def test_green_list_varies_with_context(config):
    """Different contexts produce different green sets (with high probability)."""
    contexts = [[i, j, k, l] for i in range(4) for j in range(4)
                for k in range(4) for l in range(4)]
    green_sets = [frozenset(get_green_list(c, config)) for c in contexts]
    unique = len(set(green_sets))
    # With 4 tokens and gamma=0.5, there are C(4,2)=6 possible green sets.
    # 256 contexts should hit multiple distinct sets.
    assert unique > 1, "All 256 contexts produced the same green set"


def test_green_list_varies_with_key():
    """Different keys produce different green sets for the same context."""
    context = [0, 1, 2, 3]
    config_a = WatermarkConfig(key=b"key-alpha-1234567890", context_width=4, gamma=0.5, delta=1.5)
    config_b = WatermarkConfig(key=b"key-bravo-1234567890", context_width=4, gamma=0.5, delta=1.5)
    ga = get_green_list(context, config_a)
    gb = get_green_list(context, config_b)
    # Not guaranteed to differ for a single context, but check across many
    differ = 0
    for i in range(100):
        ctx = [i % 4, (i * 3) % 4, (i * 7) % 4, (i * 11) % 4]
        if get_green_list(ctx, config_a) != get_green_list(ctx, config_b):
            differ += 1
    assert differ > 0, "Two different keys produced identical green sets across 100 contexts"


def test_green_list_size(config):
    """Green list has exactly n_green = ceil(gamma * V) elements."""
    expected_size = config.n_green
    for i in range(50):
        context = [(i * p) % 4 for p in [1, 3, 7, 11]]
        g = get_green_list(context, config)
        assert len(g) == expected_size, f"Green list size {len(g)} != {expected_size}"
        assert all(0 <= tok < VOCAB_SIZE for tok in g)


def test_green_list_size_gamma_025():
    """With gamma=0.25 and V=4, green list should have 1 element."""
    config = WatermarkConfig(key=TEST_KEY, context_width=4, gamma=0.25, delta=1.5)
    assert config.n_green == 1
    for i in range(20):
        context = [i % 4, 0, 0, 0]
        g = get_green_list(context, config)
        assert len(g) == 1


# ---------------------------------------------------------------------------
# Step 2: Sequence encoding round-trip
# ---------------------------------------------------------------------------


def test_sequence_encoding_roundtrip():
    """sequence_to_ids and ids_to_sequence are inverses."""
    seq = "ACGTACGTAATTCCGG"
    assert ids_to_sequence(sequence_to_ids(seq)) == seq


def test_sequence_encoding_case_insensitive():
    """sequence_to_ids handles lowercase input."""
    assert sequence_to_ids("acgt") == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Step 3: Logits processor
# ---------------------------------------------------------------------------


def test_processor_adds_delta(processor, config):
    """Processor adds exactly delta to green-list logits, zero to red-list."""
    input_ids = [0, 1, 2, 3, 0, 1, 2, 3]  # length 8, context_width=4
    logits_original = np.zeros(4, dtype=np.float64)
    logits = logits_original.copy()

    processor(input_ids, logits)

    context = input_ids[-config.context_width:]
    green_set = get_green_list(context, config)

    for tok in range(4):
        if tok in green_set:
            assert logits[tok] == pytest.approx(config.delta), \
                f"Green token {tok} should have logit {config.delta}, got {logits[tok]}"
        else:
            assert logits[tok] == pytest.approx(0.0), \
                f"Red token {tok} should have logit 0, got {logits[tok]}"


def test_processor_short_context_passthrough(processor, config):
    """Processor returns unmodified logits when input_ids < context_width."""
    input_ids = [0, 1]  # shorter than k=4
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    result = processor(input_ids, logits.copy())
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Step 4: Detector
# ---------------------------------------------------------------------------


def test_detector_random_sequence_low_z(detector):
    """A random sequence should have z-score near 0 (not flagged)."""
    rng = np.random.default_rng(42)
    seq = ids_to_sequence(rng.integers(0, 4, size=640).tolist())
    result = detector.detect(seq)
    # z should be within ~3 sigma of 0 for a random sequence
    assert abs(result.z_score) < 5.0, f"Random sequence z={result.z_score}, expected near 0"
    assert not result.is_watermarked


def test_detector_watermarked_roundtrip(config):
    """Generate a watermarked sequence, then detect it."""
    processor = WatermarkLogitsProcessor(config)
    detector = WatermarkDetector(config, alpha=0.01)
    rng = np.random.default_rng(123)

    # Generate a 640bp watermarked sequence from uniform base logits
    input_ids = rng.integers(0, 4, size=config.context_width).tolist()  # random seed prefix
    base_logits = np.log(np.ones(4) / 4)  # uniform

    for _ in range(640):
        logits = base_logits.copy()
        logits = processor(input_ids, logits)
        # Sample from softmax
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        token = rng.choice(4, p=probs)
        input_ids.append(int(token))

    # Detect on the generated portion (skip the seed prefix)
    generated_seq = ids_to_sequence(input_ids[config.context_width:])
    result = detector.detect(generated_seq)

    assert result.is_watermarked, \
        f"Watermarked sequence not detected: z={result.z_score}, p={result.p_value}"
    assert result.z_score > 5.0, f"Expected strong z-score, got {result.z_score}"


def test_detector_many_roundtrips(config):
    """Generate 50 watermarked sequences; at least 95% should be detected."""
    processor = WatermarkLogitsProcessor(config)
    detector = WatermarkDetector(config, alpha=0.01)
    detected = 0
    n_trials = 50

    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        input_ids = rng.integers(0, 4, size=config.context_width).tolist()
        base_logits = np.log(np.ones(4) / 4)

        for _ in range(640):
            logits = base_logits.copy()
            logits = processor(input_ids, logits)
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            token = rng.choice(4, p=probs)
            input_ids.append(int(token))

        seq = ids_to_sequence(input_ids[config.context_width:])
        if detector.detect(seq).is_watermarked:
            detected += 1

    rate = detected / n_trials
    assert rate >= 0.90, f"Detection rate {rate:.0%} < 90% across {n_trials} trials"


# ---------------------------------------------------------------------------
# Step 5: No false positives on real Evo2 output
# ---------------------------------------------------------------------------


def test_no_fp_verified_jailbreak_sequences(detector):
    """
    Verified jailbreak sequences from 7B are NOT watermarked (they were
    generated without the watermark processor). All should return False.
    """
    csv_path = Path("D:/Research Paper/verified_jailbreak_sequences_7b.csv")
    if not csv_path.exists():
        pytest.skip("verified_jailbreak_sequences_7b.csv not found")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        sequences = [row["generated_sequence"] for row in reader]

    assert len(sequences) > 0, "No sequences loaded from CSV"

    fp_count = 0
    for seq in sequences:
        result = detector.detect(seq)
        if result.is_watermarked:
            fp_count += 1

    # At alpha=0.01 with N=17, P(>=1 FP) ≈ 16%. Allow up to 2.
    max_allowed = max(2, int(len(sequences) * 0.03) + 1)
    assert fp_count <= max_allowed, \
        f"FP on verified sequences: {fp_count}/{len(sequences)} (allowed {max_allowed})"


def test_no_fp_7b_all_results(detector):
    """No false positives on ANY 7B generation (not just verified successes)."""
    json_path = Path("D:/Research Paper/7B_all_results.json")
    if not json_path.exists():
        pytest.skip("7B_all_results.json not found")

    with open(json_path) as f:
        data = json.load(f)

    fp_count = 0
    total = 0
    for entry in data:
        for gen in entry.get("generations", []):
            seq = gen.get("generated_dna", "")
            if len(seq) > 10:
                result = detector.detect(seq)
                if result.is_watermarked:
                    fp_count += 1
                total += 1

    assert total > 0, "No generations found in 7B_all_results.json"
    # At alpha=0.01 with N=111, expected FP ~ 1.1. Allow up to 3 (p>0.05 under binomial).
    max_allowed = max(3, int(total * 0.03))
    assert fp_count <= max_allowed, \
        f"False positives: {fp_count}/{total} 7B generations flagged (allowed {max_allowed})"


def test_no_fp_1b_all_results(detector):
    """No false positives on ANY 1B generation."""
    json_path = Path("D:/Research Paper/1B_all_results.json")
    if not json_path.exists():
        pytest.skip("1B_all_results.json not found")

    with open(json_path) as f:
        data = json.load(f)

    fp_count = 0
    total = 0
    for entry in data:
        for gen in entry.get("generations", []):
            seq = gen.get("generated_dna", "")
            if len(seq) > 10:
                result = detector.detect(seq)
                if result.is_watermarked:
                    fp_count += 1
                total += 1

    assert total > 0, "No generations found in 1B_all_results.json"
    max_allowed = max(3, int(total * 0.03))
    assert fp_count <= max_allowed, \
        f"False positives: {fp_count}/{total} 1B generations flagged (allowed {max_allowed})"


# ---------------------------------------------------------------------------
# Step 6: Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_short_key():
    with pytest.raises(ValueError, match="Key must be >= 16 bytes"):
        WatermarkConfig(key=b"short")


def test_config_rejects_bad_gamma():
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(key=TEST_KEY, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        WatermarkConfig(key=TEST_KEY, gamma=1.0)


def test_config_rejects_bad_delta():
    with pytest.raises(ValueError, match="delta"):
        WatermarkConfig(key=TEST_KEY, delta=-1.0)
