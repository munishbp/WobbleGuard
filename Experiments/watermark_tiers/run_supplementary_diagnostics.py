#!/usr/bin/env python3
"""
Skepticism-check diagnostics for the wobble-aware detector (supplementary).

Test A: Wrong-key cross-validation.
        The detector must report low z on the same watermarked sequence when
        probed with a wrong key. Gold-standard sanity for any watermark.

Test B: delta=0 null on in-frame coding.
        With no watermark bias, the detector should report FPR ~= alpha.
        Catches constitutional positive bias on coding sequences.

Test C: delta-monotonicity sweep.
        Generate at delta in {0.1, 0.25, 0.5, 1.0, 1.5, 2.0}, with and
        without full synonymous substitution. TPR and mean-z must grow
        monotonically. Gives a supplementary-ready TPR-vs-delta curve.

Outputs JSON + a CSV per test, suitable to drop into the supplement.

# Make the wobbleguard package importable when run from this directory.
import sys as _sys
from pathlib import Path as _Path
_WG_ROOT = _Path(__file__).resolve().parent
for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
    if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.evaluate import apply_true_synonymous_substitutions
from wobbleguard.genetic_code import CODON_TABLE
from wobbleguard.simulate import generate_synthetic_sequence
from wobbleguard.wobble_detector import WobbleAwareDetector


CORRECT_KEY = b"neurips2026-wobble-diagnostics01"
ALPHA = 0.01
SEQ_LEN = 640


def sense_filter(seq: str) -> str:
    """Replace in-frame stops with C so detect_reading_frame() engages."""
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        if CODON_TABLE.get("".join(out[i : i + 3])) == "*":
            out[i] = "C"
    return "".join(out)


def make_config(delta: float, key: bytes = CORRECT_KEY) -> WatermarkConfig:
    return WatermarkConfig(key=key, context_width=4, gamma=0.5, delta=delta)


# ---------------------------------------------------------------------------
# Test A
# ---------------------------------------------------------------------------

def test_a_wrong_key(n_sequences: int = 50, n_wrong_keys: int = 50) -> dict:
    print(f"\n=== Test A: Wrong-key cross-validation "
          f"(n_seq={n_sequences}, n_wrong_keys={n_wrong_keys}) ===")
    cfg_correct = make_config(1.5)
    det_correct = WobbleAwareDetector(cfg_correct, alpha=ALPHA)

    rng = np.random.default_rng(101)
    correct_zs, wrong_zs = [], []
    wrong_hits = 0
    wrong_total = 0

    t0 = time.time()
    for s_i in range(n_sequences):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        seq = sense_filter(generate_synthetic_sequence(cfg_correct, SEQ_LEN, rng=seq_rng))
        correct_zs.append(det_correct.detect(seq).z_score)

        for _ in range(n_wrong_keys):
            wrong_key = bytes(int(b) for b in rng.integers(0, 256, size=32))
            det_wrong = WobbleAwareDetector(make_config(1.5, wrong_key), alpha=ALPHA)
            r = det_wrong.detect(seq)
            wrong_zs.append(r.z_score)
            wrong_total += 1
            if r.is_watermarked:
                wrong_hits += 1

    correct_arr = np.array(correct_zs)
    wrong_arr = np.array(wrong_zs)
    out = {
        "n_sequences": n_sequences,
        "n_wrong_keys_per_seq": n_wrong_keys,
        "correct_mean_z": float(correct_arr.mean()),
        "correct_std_z": float(correct_arr.std()),
        "correct_min_z": float(correct_arr.min()),
        "wrong_mean_z": float(wrong_arr.mean()),
        "wrong_std_z": float(wrong_arr.std()),
        "wrong_max_z": float(wrong_arr.max()),
        "wrong_99th_pctile_z": float(np.percentile(wrong_arr, 99)),
        "wrong_key_fpr": wrong_hits / wrong_total,
        "separation_margin": float(correct_arr.mean() - wrong_arr.mean()),
        "elapsed_sec": time.time() - t0,
    }
    print(f"  correct key z = {out['correct_mean_z']:.2f} +/- {out['correct_std_z']:.2f} "
          f"(min {out['correct_min_z']:.2f})")
    print(f"  wrong   key z = {out['wrong_mean_z']:.2f} +/- {out['wrong_std_z']:.2f} "
          f"(max {out['wrong_max_z']:.2f}, 99th pct {out['wrong_99th_pctile_z']:.2f})")
    print(f"  wrong-key FPR = {out['wrong_key_fpr']*100:.2f}% "
          f"(threshold FPR target = {ALPHA*100:.1f}%)")
    print(f"  separation    = {out['separation_margin']:.2f}  ({out['elapsed_sec']:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Test B
# ---------------------------------------------------------------------------

def test_b_delta_zero_null(n: int = 2000) -> dict:
    print(f"\n=== Test B: delta=0 null on in-frame coding (n={n}) ===")
    # "delta=0" is implemented by passing config=None to the generator.
    # The detector still uses the normal config for hashing/threshold.
    cfg = make_config(1.5)  # detector config: delta irrelevant at detect time
    det = WobbleAwareDetector(cfg, alpha=ALPHA)

    rng = np.random.default_rng(202)
    zs = []
    fp = 0
    t0 = time.time()
    for _ in range(n):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        # config=None => uniform unbiased sampling
        seq = sense_filter(generate_synthetic_sequence(None, SEQ_LEN, rng=seq_rng))
        r = det.detect(seq)
        zs.append(r.z_score)
        if r.is_watermarked:
            fp += 1
    arr = np.array(zs)
    out = {
        "n": n,
        "empirical_fpr": fp / n,
        "target_fpr": ALPHA,
        "mean_z": float(arr.mean()),
        "std_z": float(arr.std()),
        "max_z": float(arr.max()),
        "pct_above_threshold": float(np.mean(arr > det.z_threshold)),
        "elapsed_sec": time.time() - t0,
    }
    wilson_low, wilson_high = _wilson_ci(fp, n)
    out["fpr_wilson_95ci_low"] = wilson_low
    out["fpr_wilson_95ci_high"] = wilson_high
    print(f"  empirical FPR = {out['empirical_fpr']*100:.2f}% "
          f"(Wilson 95% CI: {wilson_low*100:.2f}% - {wilson_high*100:.2f}%)")
    print(f"  target FPR    = {ALPHA*100:.2f}%")
    print(f"  mean z        = {out['mean_z']:.3f} +/- {out['std_z']:.3f} "
          f"(max {out['max_z']:.2f})  ({out['elapsed_sec']:.1f}s)")
    return out


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


# ---------------------------------------------------------------------------
# Test C
# ---------------------------------------------------------------------------

def test_c_delta_sweep(n_per_delta: int = 200) -> dict:
    print(f"\n=== Test C: delta-monotonicity sweep (n={n_per_delta} per delta) ===")
    deltas = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    rng = np.random.default_rng(303)
    results = []

    for delta in deltas:
        cfg = make_config(delta)
        det_v = WatermarkDetector(cfg, alpha=ALPHA)
        det_w = WobbleAwareDetector(cfg, alpha=ALPHA)

        # Generate n watermarked sequences
        seqs = []
        for _ in range(n_per_delta):
            seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
            seqs.append(sense_filter(generate_synthetic_sequence(cfg, SEQ_LEN, rng=seq_rng)))

        for attack_rate, attack_name in [(0.0, "no_attack"), (1.0, "full_syn_sub")]:
            vz, wz = [], []
            vh, wh = 0, 0
            for s in seqs:
                if attack_rate > 0:
                    a = apply_true_synonymous_substitutions(s, attack_rate, rng, frame=0)
                else:
                    a = s
                vr = det_v.detect(a); wr = det_w.detect(a)
                vz.append(vr.z_score); wz.append(wr.z_score)
                vh += vr.is_watermarked; wh += wr.is_watermarked
            results.append({
                "delta": delta,
                "attack": attack_name,
                "vanilla_tpr": vh / n_per_delta,
                "wobble_tpr": wh / n_per_delta,
                "vanilla_mean_z": float(np.mean(vz)),
                "wobble_mean_z": float(np.mean(wz)),
            })
            r = results[-1]
            print(f"  delta={delta:<5} {attack_name:<14}: "
                  f"v_tpr={r['vanilla_tpr']:.2f} v_z={r['vanilla_mean_z']:>6.2f}  |  "
                  f"w_tpr={r['wobble_tpr']:.2f} w_z={r['wobble_mean_z']:>5.2f}")

    return {"n_per_delta": n_per_delta, "rows": results}


# ---------------------------------------------------------------------------
# Test D
# ---------------------------------------------------------------------------

def test_d_tier5_scaled(n: int = 1000) -> dict:
    """High-N synonymous substitution sweep for tight Wilson CIs on the
    paper's headline robustness claim."""
    print(f"\n=== Test D: Scaled Tier 5 synonymous sub (n={n} per rate) ===")
    cfg = make_config(1.5)
    det_v = WatermarkDetector(cfg, alpha=ALPHA)
    det_w = WobbleAwareDetector(cfg, alpha=ALPHA)

    rng = np.random.default_rng(404)
    print(f"  generating {n} watermarked sequences...")
    t0 = time.time()
    seqs = []
    for _ in range(n):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        seqs.append(sense_filter(generate_synthetic_sequence(cfg, SEQ_LEN, rng=seq_rng)))
    print(f"  generation done ({time.time()-t0:.0f}s)")

    rows = []
    for rate in [0.0, 0.25, 0.50, 0.75, 1.00]:
        v_hits = w_hits = 0
        vz, wz = [], []
        for s in seqs:
            if rate > 0:
                a = apply_true_synonymous_substitutions(s, rate, rng, frame=0)
            else:
                a = s
            vr = det_v.detect(a); wr = det_w.detect(a)
            vz.append(vr.z_score); wz.append(wr.z_score)
            v_hits += vr.is_watermarked; w_hits += wr.is_watermarked

        v_ci = _wilson_ci(v_hits, n)
        w_ci = _wilson_ci(w_hits, n)
        row = {
            "rate": rate,
            "n": n,
            "vanilla_tpr": v_hits / n,
            "vanilla_tpr_95ci_low": v_ci[0],
            "vanilla_tpr_95ci_high": v_ci[1],
            "vanilla_mean_z": float(np.mean(vz)),
            "wobble_tpr": w_hits / n,
            "wobble_tpr_95ci_low": w_ci[0],
            "wobble_tpr_95ci_high": w_ci[1],
            "wobble_mean_z": float(np.mean(wz)),
        }
        rows.append(row)
        print(f"  rate={rate:.2f}: vanilla {v_hits}/{n} = {row['vanilla_tpr']:.3f} "
              f"[{v_ci[0]:.3f}, {v_ci[1]:.3f}] | "
              f"wobble {w_hits}/{n} = {row['wobble_tpr']:.3f} "
              f"[{w_ci[0]:.3f}, {w_ci[1]:.3f}]")

    return {"n": n, "rows": rows, "elapsed_sec": time.time() - t0}


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict], keys: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    results = {}
    results["test_a"] = test_a_wrong_key(n_sequences=50, n_wrong_keys=50)
    results["test_b"] = test_b_delta_zero_null(n=2000)
    results["test_c"] = test_c_delta_sweep(n_per_delta=200)
    results["test_d"] = test_d_tier5_scaled(n=1000)

    out_json = Path("wobble_diagnostics_results.json")
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    # Supplementary-ready CSVs
    write_csv(
        Path("supp_test_a_wrong_key.csv"),
        [{
            "metric": k,
            "value": v,
        } for k, v in results["test_a"].items()],
        keys=["metric", "value"],
    )
    write_csv(
        Path("supp_test_b_null.csv"),
        [{
            "metric": k,
            "value": v,
        } for k, v in results["test_b"].items()],
        keys=["metric", "value"],
    )
    write_csv(
        Path("supp_test_c_delta_sweep.csv"),
        results["test_c"]["rows"],
        keys=["delta", "attack", "vanilla_tpr", "wobble_tpr",
              "vanilla_mean_z", "wobble_mean_z"],
    )
    write_csv(
        Path("supp_test_d_tier5_scaled.csv"),
        results["test_d"]["rows"],
        keys=["rate", "n", "vanilla_tpr", "vanilla_tpr_95ci_low",
              "vanilla_tpr_95ci_high", "vanilla_mean_z",
              "wobble_tpr", "wobble_tpr_95ci_low",
              "wobble_tpr_95ci_high", "wobble_mean_z"],
    )
    print(f"\nResults -> {out_json}")
    print("Supplementary CSVs -> supp_test_a_wrong_key.csv, "
          "supp_test_b_null.csv, supp_test_c_delta_sweep.csv")


if __name__ == "__main__":
    main()
