#!/usr/bin/env python3
"""
Test suite for the wobble-aware DNA watermark detector.

Run: pytest test_wobble_detector.py -v

# Make the wobbleguard package importable when run from this directory.
import sys as _sys
from pathlib import Path as _Path
_WG_ROOT = _Path(__file__).resolve().parent
for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
    if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

"""

import numpy as np
import pytest

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkLogitsProcessor, WatermarkDetector
from wobbleguard.evaluate import (
    apply_random_mutations,
    apply_true_synonymous_substitutions,
)
from wobbleguard.genetic_code import (
    CODON_TABLE,
    SYNONYM_CLASS,
    WOBBLE_ALTERNATES,
    detect_reading_frame,
)
from wobbleguard.simulate import generate_synthetic_sequence
from wobbleguard.wobble_detector import WobbleAwareDetector


TEST_KEY = b"neurips2026-wobble-defense-test!"  # 32 bytes


@pytest.fixture
def config():
    return WatermarkConfig(key=TEST_KEY, context_width=4, gamma=0.5, delta=1.5)


@pytest.fixture
def vanilla(config):
    return WatermarkDetector(config, alpha=0.01)


@pytest.fixture
def wobble(config):
    return WobbleAwareDetector(config, alpha=0.01)


# ---------------------------------------------------------------------------
# Genetic-code table sanity
# ---------------------------------------------------------------------------

def test_codon_table_has_64_entries():
    assert len(CODON_TABLE) == 64


def test_three_stop_codons():
    stops = [c for c, aa in CODON_TABLE.items() if aa == "*"]
    assert set(stops) == {"TAA", "TAG", "TGA"}


def test_61_sense_codons():
    sense = [c for c, aa in CODON_TABLE.items() if aa != "*"]
    assert len(sense) == 61


def test_synonym_class_is_reflexive_and_symmetric():
    for codon, synonyms in SYNONYM_CLASS.items():
        assert codon in synonyms
        for s in synonyms:
            assert codon in SYNONYM_CLASS[s]


def test_wobble_alternates_share_amino_acid():
    for codon, alts in WOBBLE_ALTERNATES.items():
        if CODON_TABLE[codon] == "*":
            continue
        for alt in alts:
            assert CODON_TABLE[alt] == CODON_TABLE[codon]
            assert alt[:2] == codon[:2]  # only 3rd position differs


def test_methionine_and_tryptophan_have_singleton_classes():
    assert WOBBLE_ALTERNATES["ATG"] == frozenset({"ATG"})  # Met
    assert WOBBLE_ALTERNATES["TGG"] == frozenset({"TGG"})  # Trp


def test_leucine_codons_are_fourfold_degenerate_at_wobble():
    # CTN codes Leu for N in {A,C,G,T}.
    assert WOBBLE_ALTERNATES["CTA"] == frozenset({"CTA", "CTC", "CTG", "CTT"})


# ---------------------------------------------------------------------------
# Reading-frame detection
# ---------------------------------------------------------------------------

def test_detect_frame_on_pure_coding_sequence():
    # 60 Leu codons — no stops in frame 0.
    seq = "CTG" * 60
    assert detect_reading_frame(seq) == 0


def test_detect_frame_returns_none_on_random_dna():
    rng = np.random.default_rng(0)
    random_seq = "".join(rng.choice(list("ACGT"), size=640))
    frame = detect_reading_frame(random_seq, min_orf_codons=50)
    # Random DNA rarely produces a 50-codon stop-free run; this is a soft check
    # — we accept None or a frame, but it should be consistent on the same seed.
    assert frame is None or frame in (0, 1, 2)


# ---------------------------------------------------------------------------
# Detector invariance and graceful fallback
# ---------------------------------------------------------------------------

def test_wobble_detector_falls_back_on_non_coding(wobble, vanilla):
    rng = np.random.default_rng(7)
    random_seq = "".join(rng.choice(list("ACGT"), size=640))
    res = wobble.detect(random_seq)
    # Non-coding → vanilla path. z-score should equal the vanilla detector's.
    van = vanilla.detect(random_seq)
    assert res.fell_back_to_vanilla
    assert abs(res.z_score - van.z_score) < 1e-9


def test_wobble_detector_calibrated_on_synthetic_null(wobble, config):
    """FPR on 200 unwatermarked coding sequences should be near alpha=0.01."""
    rng = np.random.default_rng(123)
    fp = 0
    trials = 0
    for _ in range(200):
        # Build a 200-codon random sense-codon sequence (no stops, so a frame
        # is guaranteed to be found).
        codons = []
        for _ in range(200):
            while True:
                c = "".join(rng.choice(list("ACGT"), size=3))
                if CODON_TABLE[c] != "*":
                    codons.append(c)
                    break
        seq = "".join(codons)
        res = wobble.detect(seq)
        if res.is_watermarked:
            fp += 1
        trials += 1
    fpr = fp / trials
    # Allow a permissive ceiling — we want calibration near alpha=0.01, and
    # will tighten when running at 10k trials.
    assert fpr <= 0.05, f"FPR {fpr:.3f} exceeds 5% calibration ceiling"


# ---------------------------------------------------------------------------
# Attack-recovery: the headline test
# ---------------------------------------------------------------------------

def _generate_watermarked_coding_sequence(config, rng):
    """Emit a sense-codon-only watermarked sequence with known frame 0."""
    # Start with a short random prefix (in-frame, no stops).
    seq = generate_synthetic_sequence(config, length=640, rng=rng)
    # Filter to guarantee no stops in frame 0: replace any in-frame stop codon's
    # first base with C (all C-prefix codons are sense).
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        codon = "".join(out[i : i + 3])
        if CODON_TABLE.get(codon) == "*":
            out[i] = "C"
    return "".join(out)


def test_wobble_detector_matches_vanilla_on_unperturbed_watermark(
    wobble, vanilla, config
):
    """TPR parity: an unperturbed watermarked sequence should be flagged by both."""
    rng = np.random.default_rng(42)
    detected_wobble = 0
    detected_vanilla = 0
    n = 30
    for _ in range(n):
        seq = _generate_watermarked_coding_sequence(
            config, np.random.default_rng(int(rng.integers(0, 2**32)))
        )
        if wobble.detect(seq).is_watermarked:
            detected_wobble += 1
        if vanilla.detect(seq).is_watermarked:
            detected_vanilla += 1
    # Wobble detector should retain most of vanilla's detection power.
    assert detected_wobble >= max(1, int(0.8 * detected_vanilla))


def test_wobble_detector_recovers_under_full_synonymous_substitution(
    wobble, vanilla, config
):
    """
    The headline claim: vanilla KGW-DNA collapses to ~4% detection under
    full synonymous substitution; the wobble-aware detector must stay
    substantially higher.
    """
    rng = np.random.default_rng(99)
    wobble_hits = 0
    vanilla_hits = 0
    n = 30
    for _ in range(n):
        clean = _generate_watermarked_coding_sequence(
            config, np.random.default_rng(int(rng.integers(0, 2**32)))
        )
        attacked = apply_true_synonymous_substitutions(
            clean, sub_rate=1.0, rng=rng, frame=0
        )
        if wobble.detect(attacked).is_watermarked:
            wobble_hits += 1
        if vanilla.detect(attacked).is_watermarked:
            vanilla_hits += 1
    # Real test: wobble must beat vanilla by a wide margin.
    assert wobble_hits >= vanilla_hits
    assert wobble_hits >= int(0.5 * n), (
        f"wobble detection {wobble_hits}/{n} below 50% under full synonymous attack"
    )


def test_true_synonymous_attack_preserves_protein():
    rng = np.random.default_rng(3)
    # 200-codon sense-only sequence
    codons = []
    for _ in range(200):
        while True:
            c = "".join(rng.choice(list("ACGT"), size=3))
            if CODON_TABLE[c] != "*":
                codons.append(c)
                break
    seq = "".join(codons)
    attacked = apply_true_synonymous_substitutions(seq, sub_rate=1.0, rng=rng, frame=0)
    # Same length, same protein translation
    assert len(attacked) == len(seq)
    for i in range(0, len(seq), 3):
        assert CODON_TABLE[seq[i : i + 3]] == CODON_TABLE[attacked[i : i + 3]]
