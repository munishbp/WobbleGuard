"""Compute max(z_v, z_w) TPR at the empirical threshold z*=3.065 across the
synonymous substitution rate sweep used in Tier 5 / Table 5.

Mirrors run_wobble_gauntlet.py's synonymous block but adds the composite

# Make the wobbleguard package importable when run from this directory.
import sys as _sys
from pathlib import Path as _Path
_WG_ROOT = _Path(__file__).resolve().parent
for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
    if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

max-z TPR at the empirical threshold from app:maxz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.evaluate import apply_true_synonymous_substitutions
from wobbleguard.genetic_code import CODON_TABLE
from wobbleguard.simulate import generate_synthetic_sequence
from wobbleguard.wobble_detector import WobbleAwareDetector

KEY = b"neurips2026-maxz-calibration-key"
CFG = WatermarkConfig(key=KEY, context_width=4, gamma=0.5, delta=1.5)
ALPHA = 0.01
SEQ_LEN = 640
N_WM = 1000  # match the Tier 5 / maxz N=1,000 reported in app:maxz
Z_STAR_EMP = 3.065  # from app:maxz, Table tab:supp-f


def sense_filter(seq: str) -> str:
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        if CODON_TABLE.get("".join(out[i : i + 3])) == "*":
            out[i] = "C"
    return "".join(out)


def run() -> dict:
    vanilla = WatermarkDetector(CFG, alpha=ALPHA)
    wobble = WobbleAwareDetector(CFG, alpha=ALPHA)

    rng = np.random.default_rng(606)
    print(f"Generating {N_WM} watermarked sequences (seed=606)...")
    seqs = []
    for _ in range(N_WM):
        seq_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        seqs.append(sense_filter(generate_synthetic_sequence(CFG, SEQ_LEN, rng=seq_rng)))

    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    table = []
    for rate in rates:
        zv_list, zw_list = [], []
        for s in seqs:
            a = apply_true_synonymous_substitutions(s, rate, rng, frame=0) if rate > 0 else s
            zv_list.append(vanilla.detect(a).z_score)
            zw_list.append(wobble.detect(a).z_score)
        zv = np.array(zv_list)
        zw = np.array(zw_list)
        maxz = np.maximum(zv, zw)

        # TPR at analytical threshold (z=2.326) and empirical threshold (3.065)
        n = len(zv)
        tpr_v_ana = int(np.sum(zv > 2.326)) / n
        tpr_w_ana = int(np.sum(zw > 2.326)) / n
        tpr_max_emp = int(np.sum(maxz > Z_STAR_EMP)) / n

        table.append({
            "rate": rate,
            "n": n,
            "vanilla_tpr_analytical": tpr_v_ana,
            "wobble_tpr_analytical": tpr_w_ana,
            "maxz_tpr_empirical": tpr_max_emp,
        })
        print(
            f"  r={rate}: zv@2.326 TPR={tpr_v_ana:.4f}  "
            f"zw@2.326 TPR={tpr_w_ana:.4f}  "
            f"max@3.065 TPR={tpr_max_emp:.4f}"
        )

    Path("maxz_synonymous_sweep.json").write_text(json.dumps(table, indent=2))
    print("\nSaved -> maxz_synonymous_sweep.json")
    return {"table": table}


if __name__ == "__main__":
    run()
