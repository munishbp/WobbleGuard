#!/usr/bin/env python3
"""
Test E: Adaptive synonymous-substitution attack.

Instead of picking a uniformly random synonym at each wobble position, the
attacker selects the synonym that MINIMIZES the detector's flag at the
affected positions. This is the realistic worst-case attack a sophisticated
adversary would use.

Two variants implemented:
  E.1 'flag_greedy'  — local greedy: pick synonym that minimizes sum of
                       flag_t over the affected window [t_w, t_w+k].
  E.2 'g_union_max'  — pick synonym that maximizes |G_union| at the
                       affected positions (drives gamma_t -> 1 and
                       silently reduces effective T).

Runs at N=1000 to get Wilson CIs comparable with Test D.

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

from wobbleguard.config import DNA_CHAR_TO_IDX, WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.genetic_code import (
    CODON_TABLE,
    WOBBLE_ALTERNATES,
    wobble_alternates_at_position,
)
from wobbleguard.simulate import generate_synthetic_sequence
from wobbleguard.utils import get_green_list
from wobbleguard.wobble_detector import WobbleAwareDetector, _enumerate_context_candidates


KEY = b"neurips2026-wobble-adaptive-key!"
CFG = WatermarkConfig(key=KEY, context_width=4, gamma=0.5, delta=1.5)
ALPHA = 0.01
SEQ_LEN = 640
N = 1000


def sense_filter(seq: str) -> str:
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        if CODON_TABLE.get("".join(out[i : i + 3])) == "*":
            out[i] = "C"
    return "".join(out)


def _g_union_at(seq: str, t: int, frame: int) -> frozenset[int]:
    k = CFG.context_width
    if t < k:
        return frozenset()
    candidates = _enumerate_context_candidates(seq, t - k, k, frame)
    accum: set[int] = set()
    for cand in candidates:
        accum |= get_green_list([DNA_CHAR_TO_IDX[c] for c in cand], CFG)
    return frozenset(accum)


def _flag_at(seq: str, t: int, frame: int, g_union: frozenset[int]) -> int:
    """1 if position t would flag under the wobble-aware rule with this g_union."""
    alts = wobble_alternates_at_position(seq, t, frame)
    obs_ids = {DNA_CHAR_TO_IDX[c] for c in alts}
    return 1 if (obs_ids & g_union) else 0


def apply_adaptive_flag_greedy(sequence: str, frame: int = 0) -> str:
    """Per-wobble-position greedy: choose synonym minimizing sum of flags at [t_w .. t_w+k]."""
    seq = list(sequence.upper())
    k = CFG.context_width
    L = len(seq)

    for cs in range(frame, L - 2, 3):
        wobble_pos = cs + 2
        codon = "".join(seq[cs : cs + 3])
        alts = WOBBLE_ALTERNATES.get(codon, frozenset())
        if len(alts) <= 1:
            continue
        candidates_nt = sorted({alt[2] for alt in alts})

        best_nt = seq[wobble_pos]
        best_score = None
        for nt in candidates_nt:
            seq[wobble_pos] = nt
            # Count flags in the affected window.
            score = 0
            for t in range(wobble_pos, min(wobble_pos + k + 1, L)):
                if t < k:
                    continue
                g = _g_union_at("".join(seq), t, frame)
                score += _flag_at("".join(seq), t, frame, g)
            if best_score is None or score < best_score:
                best_score = score
                best_nt = nt
        seq[wobble_pos] = best_nt

    return "".join(seq)


def apply_adaptive_g_union_max(sequence: str, frame: int = 0) -> str:
    """Per-wobble-position greedy: choose synonym maximizing |G_union| in the affected window.
    Larger G_union pushes per-position gamma_t toward 1 and drops positions from the z-test."""
    seq = list(sequence.upper())
    k = CFG.context_width
    L = len(seq)

    for cs in range(frame, L - 2, 3):
        wobble_pos = cs + 2
        codon = "".join(seq[cs : cs + 3])
        alts = WOBBLE_ALTERNATES.get(codon, frozenset())
        if len(alts) <= 1:
            continue
        candidates_nt = sorted({alt[2] for alt in alts})

        best_nt = seq[wobble_pos]
        best_union_size = -1
        for nt in candidates_nt:
            seq[wobble_pos] = nt
            union_sum = 0
            for t in range(wobble_pos, min(wobble_pos + k + 1, L)):
                if t < k:
                    continue
                union_sum += len(_g_union_at("".join(seq), t, frame))
            if union_sum > best_union_size:
                best_union_size = union_sum
                best_nt = nt
        seq[wobble_pos] = best_nt

    return "".join(seq)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def run() -> dict:
    print(f"=== Test E: Adaptive attacker at N={N} ===")
    det_v = WatermarkDetector(CFG, alpha=ALPHA)
    det_w = WobbleAwareDetector(CFG, alpha=ALPHA)

    rng = np.random.default_rng(505)
    print(f"  generating {N} watermarked sequences...")
    t0 = time.time()
    seqs = []
    for _ in range(N):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        seqs.append(sense_filter(generate_synthetic_sequence(CFG, SEQ_LEN, rng=seq_rng)))
    print(f"  generation done ({time.time()-t0:.0f}s)")

    rows = []
    for attack_name, attack_fn in [
        ("random_syn_sub_1.0", None),  # baseline for comparison (random synonym)
        ("adaptive_flag_greedy", apply_adaptive_flag_greedy),
        ("adaptive_g_union_max", apply_adaptive_g_union_max),
    ]:
        print(f"\n  attack = {attack_name}")
        t_start = time.time()
        v_hits = w_hits = 0
        vz, wz = [], []
        for i, s in enumerate(seqs):
            if attack_fn is None:
                # random synonym baseline
                from wobbleguard.evaluate import apply_true_synonymous_substitutions
                a = apply_true_synonymous_substitutions(s, 1.0, rng, frame=0)
            else:
                a = attack_fn(s, frame=0)
            vr = det_v.detect(a); wr = det_w.detect(a)
            vz.append(vr.z_score); wz.append(wr.z_score)
            v_hits += vr.is_watermarked; w_hits += wr.is_watermarked
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{N} ({time.time()-t_start:.0f}s)")

        v_ci = _wilson_ci(v_hits, N)
        w_ci = _wilson_ci(w_hits, N)
        row = {
            "attack": attack_name,
            "n": N,
            "vanilla_tpr": v_hits / N,
            "vanilla_tpr_95ci_low": v_ci[0],
            "vanilla_tpr_95ci_high": v_ci[1],
            "vanilla_mean_z": float(np.mean(vz)),
            "wobble_tpr": w_hits / N,
            "wobble_tpr_95ci_low": w_ci[0],
            "wobble_tpr_95ci_high": w_ci[1],
            "wobble_mean_z": float(np.mean(wz)),
            "elapsed_sec": time.time() - t_start,
        }
        rows.append(row)
        print(f"  -> vanilla {v_hits}/{N} = {row['vanilla_tpr']:.3f} "
              f"[{v_ci[0]:.3f}, {v_ci[1]:.3f}] (mean_z={row['vanilla_mean_z']:.2f})")
        print(f"  -> wobble  {w_hits}/{N} = {row['wobble_tpr']:.3f} "
              f"[{w_ci[0]:.3f}, {w_ci[1]:.3f}] (mean_z={row['wobble_mean_z']:.2f})")

    return {"n": N, "rows": rows}


def main() -> None:
    out = run()
    Path("wobble_adaptive_results.json").write_text(json.dumps(out, indent=2))
    with Path("supp_test_e_adaptive.csv").open("w", newline="") as f:
        keys = ["attack", "n", "vanilla_tpr", "vanilla_tpr_95ci_low",
                "vanilla_tpr_95ci_high", "vanilla_mean_z",
                "wobble_tpr", "wobble_tpr_95ci_low", "wobble_tpr_95ci_high",
                "wobble_mean_z", "elapsed_sec"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in out["rows"]:
            w.writerow(row)
    print("\nResults -> wobble_adaptive_results.json, supp_test_e_adaptive.csv")


if __name__ == "__main__":
    main()
