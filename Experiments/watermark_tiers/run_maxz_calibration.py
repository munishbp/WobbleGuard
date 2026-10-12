#!/usr/bin/env python3
# empirical calibration of the max-z detection threshold.
#
# the drill:
#   1. run both detectors on the full NCBI corpus (22,982 unwatermarked
#      fragments).
#   2. compute max(z_v, z_w) per sequence.
#   3. pick the threshold at the (1 - alpha) quantile. that's the threshold
#      that gives FPR = alpha by construction.
#   4. apply the same threshold to watermarked sequences under the tier 5
#      attack panel and report TPR.
#
# saves raw z-score arrays to json for any downstream analysis.
#
# the bootstrap snippet below was never live code (it just sat inside the old
# docstring), so it stays commented out:
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.evaluate import (
    apply_random_mutations,
    apply_true_synonymous_substitutions,
    apply_truncation,
)
from wobbleguard.genetic_code import CODON_TABLE
from wobbleguard.simulate import generate_synthetic_sequence
from wobbleguard.wobble_detector import WobbleAwareDetector

KEY = b"neurips2026-maxz-calibration-key"
CFG = WatermarkConfig(key=KEY, context_width=4, gamma=0.5, delta=1.5)
ALPHA = 0.01
SEQ_LEN = 640
N_WM = 1000  # watermarked seqs for the attack tpr


def sense_filter(seq: str) -> str:
    # swap any in-frame stop codon for C so the reading-frame detector engages
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        if CODON_TABLE.get("".join(out[i : i + 3])) == "*":
            out[i] = "C"
    return "".join(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # wilson score interval, clipped to [0, 1]
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def run() -> dict:
    vanilla = WatermarkDetector(CFG, alpha=ALPHA)
    wobble = WobbleAwareDetector(CFG, alpha=ALPHA)
    z_ana = float(np.round(vanilla.z_threshold, 3))

    # --- step 1: ncbi corpus ---
    print("Step 1: detecting on 22,982 NCBI fragments...")
    with Path("ncbi_negative_sequences.json").open() as f:
        ncbi = json.load(f)

    zv_ncbi, zw_ncbi = [], []
    t0 = time.time()
    for j, frag in enumerate(ncbi):
        seq = frag["sequence"]
        if len(seq) <= CFG.context_width + 1:
            continue
        zv_ncbi.append(vanilla.detect(seq).z_score)
        zw_ncbi.append(wobble.detect(seq).z_score)
        if (j + 1) % 5000 == 0:
            print(f"  {j+1}/{len(ncbi)} ({time.time()-t0:.0f}s)")
    zv_ncbi = np.array(zv_ncbi)
    zw_ncbi = np.array(zw_ncbi)
    maxz_ncbi = np.maximum(zv_ncbi, zw_ncbi)
    print(f"  done ({time.time()-t0:.0f}s)")

    # --- step 2: empirical thresholds ---
    n = len(maxz_ncbi)
    thr_v = float(np.quantile(zv_ncbi, 1 - ALPHA))
    thr_w = float(np.quantile(zw_ncbi, 1 - ALPHA))
    thr_max = float(np.quantile(maxz_ncbi, 1 - ALPHA))
    fpr_v_ana = int(np.sum(zv_ncbi > z_ana)) / n
    fpr_w_ana = int(np.sum(zw_ncbi > z_ana)) / n
    fpr_max_ana = int(np.sum(maxz_ncbi > z_ana)) / n

    print(f"\nAnalytical threshold z* = {z_ana} at alpha={ALPHA}:")
    print(f"  vanilla FPR = {fpr_v_ana*100:.2f}%  (CI {tuple(round(x*100,2) for x in wilson(int(fpr_v_ana*n), n))})")
    print(f"  wobble  FPR = {fpr_w_ana*100:.2f}%  (CI {tuple(round(x*100,2) for x in wilson(int(fpr_w_ana*n), n))})")
    print(f"  max-z   FPR = {fpr_max_ana*100:.2f}%")

    print(f"\nEmpirical thresholds at alpha={ALPHA} (99th pctile of the NCBI null):")
    print(f"  vanilla threshold : z_v > {thr_v:.3f}")
    print(f"  wobble threshold  : z_w > {thr_w:.3f}")
    print(f"  max-z threshold   : max(z_v, z_w) > {thr_max:.3f}")

    # --- step 3: tpr at the max-z threshold on the watermarked attack panel ---
    print(f"\nStep 3: TPR on {N_WM} watermarked sequences under attack panel...")
    rng = np.random.default_rng(606)
    seqs = []
    for _ in range(N_WM):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        seqs.append(sense_filter(generate_synthetic_sequence(CFG, SEQ_LEN, rng=seq_rng)))

    attacks = [
        ("no_attack", lambda s: s),
        ("syn_sub_1.0", lambda s: apply_true_synonymous_substitutions(s, 1.0, rng, frame=0)),
        ("mutation_0.20", lambda s: apply_random_mutations(s, 0.20, rng)),
        ("truncation_0.25", lambda s: apply_truncation(s, 0.25)),
    ]

    tpr_table = []
    for name, fn in attacks:
        zv, zw = [], []
        for s in seqs:
            a = fn(s)
            if len(a) <= CFG.context_width + 1:
                continue
            zv.append(vanilla.detect(a).z_score)
            zw.append(wobble.detect(a).z_score)
        zv = np.array(zv); zw = np.array(zw)
        maxz = np.maximum(zv, zw)

        tpr_v_ana = int(np.sum(zv > z_ana)) / len(zv)
        tpr_w_ana = int(np.sum(zw > z_ana)) / len(zw)
        tpr_max_emp = int(np.sum(maxz > thr_max)) / len(maxz)
        tpr_max_ana = int(np.sum(maxz > z_ana)) / len(maxz)

        tpr_table.append({
            "attack": name,
            "n": len(zv),
            "vanilla_tpr_analytical": tpr_v_ana,
            "wobble_tpr_analytical": tpr_w_ana,
            "maxz_tpr_analytical": tpr_max_ana,
            "maxz_tpr_empirical": tpr_max_emp,
        })
        print(f"  {name:<18}: v={tpr_v_ana:.3f}  w={tpr_w_ana:.3f}  "
              f"max(z>{z_ana})={tpr_max_ana:.3f}  "
              f"max(z>{thr_max:.2f})={tpr_max_emp:.3f}")

    out = {
        "alpha": ALPHA,
        "n_ncbi": n,
        "analytical_threshold": z_ana,
        "empirical_threshold_vanilla": thr_v,
        "empirical_threshold_wobble": thr_w,
        "empirical_threshold_maxz": thr_max,
        "ncbi_fpr_analytical": {
            "vanilla": fpr_v_ana,
            "wobble": fpr_w_ana,
            "maxz": fpr_max_ana,
        },
        "watermarked_tpr": tpr_table,
        "zv_ncbi_stats": {
            "mean": float(zv_ncbi.mean()),
            "std": float(zv_ncbi.std()),
            "max": float(zv_ncbi.max()),
            "99th_pctile": thr_v,
        },
        "zw_ncbi_stats": {
            "mean": float(zw_ncbi.mean()),
            "std": float(zw_ncbi.std()),
            "max": float(zw_ncbi.max()),
            "99th_pctile": thr_w,
        },
    }

    Path("maxz_calibration_results.json").write_text(json.dumps(out, indent=2))
    # also save the raw z arrays in case anyone wants them later
    np.savez("maxz_ncbi_zscores.npz", zv=zv_ncbi, zw=zw_ncbi, maxz=maxz_ncbi)
    print("\nSaved -> maxz_calibration_results.json, maxz_ncbi_zscores.npz")
    return out


if __name__ == "__main__":
    run()
