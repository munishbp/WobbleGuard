#!/usr/bin/env python3
# key-variance study (#13).
#
# for K independently-sampled secret keys, run both the vanilla and wobble
# detectors over the 22,982-fragment NCBI corpus and report per-key FPR at:
#   * the analytical z = 2.326 threshold (alpha = 0.01)
#   * the empirical z* = 3.065 threshold (max-z, from appendix F)
#
# reports mean +/- std FPR across keys plus the per-key spread. this tells us
# how much the single-key tier 3 numbers wobble at deployment time.
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
import os
import secrets
import time
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.wobble_detector import WobbleAwareDetector


N_KEYS = 8
ALPHA = 0.01
ANALYTICAL_THR = 2.326
EMPIRICAL_MAXZ_THR = 3.065
SEED_BASE = 909


def make_keys(n: int) -> list[bytes]:
    # n cryptographically random 32-byte secret keys
    return [secrets.token_bytes(32) for _ in range(n)]


def per_key_ncbi_sweep(key: bytes, fragments: list) -> dict:
    cfg = WatermarkConfig(key=key, context_width=4, gamma=0.5, delta=1.5)
    vanilla = WatermarkDetector(cfg, alpha=ALPHA)
    wobble = WobbleAwareDetector(cfg, alpha=ALPHA)

    zv, zw = [], []
    for frag in fragments:
        seq = frag["sequence"]
        if len(seq) <= cfg.context_width + 1:
            continue
        zv.append(vanilla.detect(seq).z_score)
        zw.append(wobble.detect(seq).z_score)
    zv = np.array(zv)
    zw = np.array(zw)
    maxz = np.maximum(zv, zw)
    n = len(zv)

    return {
        "n": n,
        "fpr_vanilla_analytical": float(np.mean(zv > ANALYTICAL_THR)),
        "fpr_wobble_analytical": float(np.mean(zw > ANALYTICAL_THR)),
        "fpr_maxz_analytical": float(np.mean(maxz > ANALYTICAL_THR)),
        "fpr_maxz_empirical": float(np.mean(maxz > EMPIRICAL_MAXZ_THR)),
        "zv_max": float(zv.max()),
        "zw_max": float(zw.max()),
        "maxz_max": float(maxz.max()),
        "zv_99pct": float(np.quantile(zv, 0.99)),
        "zw_99pct": float(np.quantile(zw, 0.99)),
        "maxz_99pct": float(np.quantile(maxz, 0.99)),
    }


def main():
    print(f"Loading NCBI fragments...")
    with Path("ncbi_negative_sequences.json").open() as f:
        fragments = json.load(f)
    print(f"  {len(fragments)} fragments")

    keys = make_keys(N_KEYS)

    per_key = []
    t0 = time.time()
    for i, key in enumerate(keys):
        ti = time.time()
        result = per_key_ncbi_sweep(key, fragments)
        elapsed = time.time() - ti
        result["key_index"] = i
        result["key_hex_prefix"] = key[:6].hex()
        per_key.append(result)
        print(f"key {i+1}/{N_KEYS} [{result['key_hex_prefix']}...] in {elapsed:.0f}s: "
              f"vanilla FPR(z=2.326)={result['fpr_vanilla_analytical']*100:.2f}%, "
              f"wobble FPR(z=2.326)={result['fpr_wobble_analytical']*100:.2f}%, "
              f"maxz FPR(z=3.065)={result['fpr_maxz_empirical']*100:.2f}%")

    # aggregate the per-key numbers into mean/std/min/max
    def stats(field: str) -> dict:
        arr = np.array([k[field] for k in per_key])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "n_keys": N_KEYS,
        "n_fragments": per_key[0]["n"],
        "thresholds": {
            "analytical": ANALYTICAL_THR,
            "empirical_maxz": EMPIRICAL_MAXZ_THR,
        },
        "fpr_vanilla_analytical": stats("fpr_vanilla_analytical"),
        "fpr_wobble_analytical": stats("fpr_wobble_analytical"),
        "fpr_maxz_analytical": stats("fpr_maxz_analytical"),
        "fpr_maxz_empirical": stats("fpr_maxz_empirical"),
        "per_key": per_key,
        "wall_time_seconds": time.time() - t0,
    }

    Path("key_variance_results.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print(f"KEY-VARIANCE SUMMARY ({N_KEYS} independent keys, "
          f"n = {summary['n_fragments']} NCBI fragments each)")
    print("=" * 70)
    for label, key in [
        ("Vanilla FPR @ z=2.326", "fpr_vanilla_analytical"),
        ("Wobble  FPR @ z=2.326", "fpr_wobble_analytical"),
        ("Max-z   FPR @ z=2.326", "fpr_maxz_analytical"),
        ("Max-z   FPR @ z=3.065 (empirical)", "fpr_maxz_empirical"),
    ]:
        s = summary[key]
        print(f"  {label:38s}  mean={s['mean']*100:5.2f}%  std={s['std']*100:5.2f}pp  "
              f"min={s['min']*100:5.2f}%  max={s['max']*100:5.2f}%")
    print(f"\nTotal wall time: {summary['wall_time_seconds']:.0f}s")
    print(f"Saved -> key_variance_results.json")


if __name__ == "__main__":
    main()
