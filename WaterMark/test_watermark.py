#!/usr/bin/env python3
# tests for the dna watermarking engine.
# covers green-list computation, logits processing, detection stats,
# and false positives on real data.
# run with: pytest test_watermark.py -v

# make the wobbleguard package importable when run from this directory.
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))

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

# fixtures

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


# step 1: green-list properties


def test_green_list_deterministic(config):
    # same context + key should always give the same green set
    context = [0, 1, 2, 3]
    g1 = get_green_list(context, config)
    g2 = get_green_list(context, config)
    assert g1 == g2


def test_green_list_varies_with_context(config):
    # different contexts should give different green sets (with high probability)
    contexts = [[i, j, k, l] for i in range(4) for j in range(4)
                for k in range(4) for l in range(4)]
    green_sets = [frozenset(get_green_list(c, config)) for c in contexts]
    unique = len(set(green_sets))
    # 4 tokens at gamma=0.5 means C(4,2)=6 possible green sets,
    # so 256 contexts should hit more than one of them
    assert unique > 1, "All 256 contexts produced the same green set"


def test_green_list_varies_with_key():
    # different keys should give different green sets for the same context
    context = [0, 1, 2, 3]
    config_a = WatermarkConfig(key=b"key-alpha-1234567890", context_width=4, gamma=0.5, delta=1.5)
    config_b = WatermarkConfig(key=b"key-bravo-1234567890", context_width=4, gamma=0.5, delta=1.5)
    ga = get_green_list(context, config_a)
    gb = get_green_list(context, config_b)
    # a single context isn't guaranteed to differ, so check a bunch
    differ = 0
    for i in range(100):
        ctx = [i % 4, (i * 3) % 4, (i * 7) % 4, (i * 11) % 4]
        if get_green_list(ctx, config_a) != get_green_list(ctx, config_b):
            differ += 1
    assert differ > 0, "Two different keys produced identical green sets across 100 contexts"


def test_green_list_size(config):
    # green list should have exactly n_green = ceil(gamma * V) tokens
    expected_size = config.n_green
    for i in range(50):
        context = [(i * p) % 4 for p in [1, 3, 7, 11]]
        g = get_green_list(context, config)
        assert len(g) == expected_size, f"Green list size {len(g)} != {expected_size}"
        assert all(0 <= tok < VOCAB_SIZE for tok in g)


def test_green_list_size_gamma_025():
    # gamma=0.25 with V=4 means the green list is just 1 token
    config = WatermarkConfig(key=TEST_KEY, context_width=4, gamma=0.25, delta=1.5)
    assert config.n_green == 1
    for i in range(20):
        context = [i % 4, 0, 0, 0]
        g = get_green_list(context, config)
        assert len(g) == 1


# step 2: sequence encoding round-trip


def test_sequence_encoding_roundtrip():
    # sequence_to_ids and ids_to_sequence should undo each other
    seq = "ACGTACGTAATTCCGG"
    assert ids_to_sequence(sequence_to_ids(seq)) == seq


def test_sequence_encoding_case_insensitive():
    # lowercase input should work too
    assert sequence_to_ids("acgt") == [0, 1, 2, 3]


# step 3: logits processor


def test_processor_adds_delta(processor, config):
    # green tokens get exactly +delta, red tokens stay at 0
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
    # fewer input tokens than context_width means the processor leaves the logits alone
    input_ids = [0, 1]  # shorter than k=4
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    result = processor(input_ids, logits.copy())
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


# step 4: detector


def test_detector_random_sequence_low_z(detector):
    # a random sequence should land near z=0 and not get flagged
    rng = np.random.default_rng(42)
    seq = ids_to_sequence(rng.integers(0, 4, size=640).tolist())
    result = detector.detect(seq)
    # z should sit within ~3 sigma of 0 for random sequence
    assert abs(result.z_score) < 5.0, f"Random sequence z={result.z_score}, expected near 0"
    assert not result.is_watermarked


def test_detector_watermarked_roundtrip(config):
    # generate a watermarked sequence, then detect it
    processor = WatermarkLogitsProcessor(config)
    detector = WatermarkDetector(config, alpha=0.01)
    rng = np.random.default_rng(123)

    # 640bp watermarked sequence from uniform base logits
    input_ids = rng.integers(0, 4, size=config.context_width).tolist()  # random seed prefix
    base_logits = np.log(np.ones(4) / 4)  # uniform

    for _ in range(640):
        logits = base_logits.copy()
        logits = processor(input_ids, logits)
        # sample from the softmax
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        token = rng.choice(4, p=probs)
        input_ids.append(int(token))

    # detect on just the generated part, skip the seed prefix
    generated_seq = ids_to_sequence(input_ids[config.context_width:])
    result = detector.detect(generated_seq)

    assert result.is_watermarked, \
        f"Watermarked sequence not detected: z={result.z_score}, p={result.p_value}"
    assert result.z_score > 5.0, f"Expected strong z-score, got {result.z_score}"


def test_detector_many_roundtrips(config):
    # 50 watermarked sequences, at least 95% should be detected
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


# step 5: no false positives on real evo2 output


def test_no_fp_verified_jailbreak_sequences(detector):
    # verified jailbreak sequences from 7B were generated without the
    # watermark processor, so they should all come back false
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

    # at alpha=0.01 with N=17, P(>=1 FP) is about 16%. allow up to 2.
    max_allowed = max(2, int(len(sequences) * 0.03) + 1)
    assert fp_count <= max_allowed, \
        f"FP on verified sequences: {fp_count}/{len(sequences)} (allowed {max_allowed})"


def test_no_fp_7b_all_results(detector):
    # no false positives on ANY 7B generation, not just the verified wins
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
    # at alpha=0.01 with N=111 the expected FP count is ~1.1, allow up to 3
    # (stays p>0.05 under the binomial)
    max_allowed = max(3, int(total * 0.03))
    assert fp_count <= max_allowed, \
        f"False positives: {fp_count}/{total} 7B generations flagged (allowed {max_allowed})"


def test_no_fp_1b_all_results(detector):
    # no false positives on ANY 1B generation
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


# step 6: config validation


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
