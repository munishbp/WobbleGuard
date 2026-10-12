#!/usr/bin/env python3
# end-to-end gauntlet: wobble-aware detector vs vanilla KGW-DNA.
#
# tier 1: monte carlo (watermarked vs unwatermarked synthetic)  -> tpr/fpr/auc
# tier 2: evo2 unwatermarked generations                        -> fpr in the wild
# tier 3: ncbi natural pathogen fragments                       -> fpr in the wild
# tier 5: robustness (synonymous sub, random mut, truncation)  -> tpr under attack
#
# writes one summary csv and prints side-by-side numbers.
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

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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


KEY = b"neurips2026-wobble-gauntlet-key!"
CFG = WatermarkConfig(key=KEY, context_width=4, gamma=0.5, delta=1.5)
ALPHA = 0.01
SEQ_LEN = 640
NCBI_SAMPLE_SIZE = 22982  # full corpus for the paper headline
MC_PER_CLASS = 2000       # per-class monte carlo sample

vanilla = WatermarkDetector(CFG, alpha=ALPHA)
wobble = WobbleAwareDetector(CFG, alpha=ALPHA)


def sense_filter(seq: str) -> str:
    # swap in-frame stops for a sense codon so reading-frame detection engages
    out = list(seq)
    for i in range(0, len(out) - 2, 3):
        if CODON_TABLE.get("".join(out[i : i + 3])) == "*":
            out[i] = "C"
    return "".join(out)


def run_tier1_monte_carlo() -> dict:
    print("\n=== Tier 1: Monte Carlo ===")
    rng = np.random.default_rng(42)
    z_wm_v, z_wm_w = [], []
    z_null_v, z_null_w = [], []

    t0 = time.time()
    for i in range(MC_PER_CLASS):
        seq = generate_synthetic_sequence(
            CFG, SEQ_LEN, rng=np.random.default_rng(int(rng.integers(0, 2**32)))
        )
        seq = sense_filter(seq)
        z_wm_v.append(vanilla.detect(seq).z_score)
        z_wm_w.append(wobble.detect(seq).z_score)

    for i in range(MC_PER_CLASS):
        # config=None => unwatermarked, unbiased null sequences
        seq = generate_synthetic_sequence(
            None, SEQ_LEN, rng=np.random.default_rng(int(rng.integers(0, 2**32)))
        )
        seq = sense_filter(seq)
        z_null_v.append(vanilla.detect(seq).z_score)
        z_null_w.append(wobble.detect(seq).z_score)
    elapsed = time.time() - t0

    zt_v = vanilla.z_threshold
    zt_w = wobble.z_threshold
    res = {
        "n_per_class": MC_PER_CLASS,
        "vanilla_tpr": float(np.mean(np.array(z_wm_v) > zt_v)),
        "vanilla_fpr": float(np.mean(np.array(z_null_v) > zt_v)),
        "vanilla_auc": float(roc_auc_score(
            [1]*MC_PER_CLASS + [0]*MC_PER_CLASS, z_wm_v + z_null_v
        )),
        "vanilla_mean_z_wm": float(np.mean(z_wm_v)),
        "vanilla_mean_z_null": float(np.mean(z_null_v)),
        "wobble_tpr": float(np.mean(np.array(z_wm_w) > zt_w)),
        "wobble_fpr": float(np.mean(np.array(z_null_w) > zt_w)),
        "wobble_auc": float(roc_auc_score(
            [1]*MC_PER_CLASS + [0]*MC_PER_CLASS, z_wm_w + z_null_w
        )),
        "wobble_mean_z_wm": float(np.mean(z_wm_w)),
        "wobble_mean_z_null": float(np.mean(z_null_w)),
        "elapsed_sec": elapsed,
    }
    print(f"  Vanilla: TPR={res['vanilla_tpr']:.3f}, FPR={res['vanilla_fpr']:.3f}, "
          f"AUC={res['vanilla_auc']:.3f}, mean_z_wm={res['vanilla_mean_z_wm']:.2f}")
    print(f"  Wobble : TPR={res['wobble_tpr']:.3f}, FPR={res['wobble_fpr']:.3f}, "
          f"AUC={res['wobble_auc']:.3f}, mean_z_wm={res['wobble_mean_z_wm']:.2f}")
    print(f"  ({elapsed:.1f}s)")
    return res


def run_tier2_evo2_unwatermarked() -> dict:
    print("\n=== Tier 2: Evo2 unwatermarked generations ===")
    out = {}
    for path, label in [
        (Path("7B_all_results.json"), "7B"),
        (Path("1B_all_results.json"), "1B"),
    ]:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        with path.open() as f:
            data = json.load(f)

        vz, wz = [], []
        fp_v = fp_w = 0
        total = 0
        for entry in data:
            for gen in entry.get("generations", []):
                seq = gen.get("generated_dna", "")
                if len(seq) <= CFG.context_width + 1:
                    continue
                vr = vanilla.detect(seq)
                wr = wobble.detect(seq)
                vz.append(vr.z_score)
                wz.append(wr.z_score)
                fp_v += vr.is_watermarked
                fp_w += wr.is_watermarked
                total += 1
        if total == 0:
            continue
        out[label] = {
            "n": total,
            "vanilla_fpr": fp_v / total,
            "wobble_fpr": fp_w / total,
            "vanilla_mean_z": float(np.mean(vz)),
            "wobble_mean_z": float(np.mean(wz)),
            "vanilla_max_z": float(np.max(vz)),
            "wobble_max_z": float(np.max(wz)),
        }
        r = out[label]
        print(f"  {label} (n={total}): vanilla FPR={r['vanilla_fpr']:.3f}, "
              f"wobble FPR={r['wobble_fpr']:.3f}; "
              f"mean_z v={r['vanilla_mean_z']:.2f}/w={r['wobble_mean_z']:.2f}")
    return out


def run_tier3_ncbi() -> dict:
    print(f"\n=== Tier 3: NCBI natural fragments (n={NCBI_SAMPLE_SIZE}) ===")
    path = Path("ncbi_negative_sequences.json")
    if not path.exists():
        print("  [skip] not found")
        return {}
    with path.open() as f:
        data = json.load(f)
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(data), size=min(NCBI_SAMPLE_SIZE, len(data)), replace=False)

    vz, wz = [], []
    fp_v = fp_w = 0
    t0 = time.time()
    for j, idx in enumerate(sample_idx):
        seq = data[int(idx)]["sequence"]
        if len(seq) <= CFG.context_width + 1:
            continue
        vr = vanilla.detect(seq)
        wr = wobble.detect(seq)
        vz.append(vr.z_score)
        wz.append(wr.z_score)
        fp_v += vr.is_watermarked
        fp_w += wr.is_watermarked
        if (j + 1) % 1000 == 0:
            print(f"  {j+1}/{len(sample_idx)} ({time.time()-t0:.0f}s)")
    n = len(vz)
    out = {
        "n": n,
        "vanilla_fpr": fp_v / n,
        "wobble_fpr": fp_w / n,
        "vanilla_mean_z": float(np.mean(vz)),
        "wobble_mean_z": float(np.mean(wz)),
        "vanilla_max_z": float(np.max(vz)),
        "wobble_max_z": float(np.max(wz)),
        "elapsed_sec": time.time() - t0,
    }
    print(f"  Vanilla FPR: {out['vanilla_fpr']*100:.2f}% ({fp_v}/{n})")
    print(f"  Wobble  FPR: {out['wobble_fpr']*100:.2f}% ({fp_w}/{n})")
    print(f"  ({out['elapsed_sec']:.0f}s)")
    return out


def run_tier5_robustness() -> dict:
    print("\n=== Tier 5: Robustness (synonymous sub, mutation, truncation) ===")
    rng = np.random.default_rng(2026)
    # pre-generate the watermarked sense-only sequences
    N5 = 500
    print(f"  generating {N5} watermarked sequences...")
    seqs = []
    for _ in range(N5):
        s = generate_synthetic_sequence(
            CFG, SEQ_LEN, rng=np.random.default_rng(int(rng.integers(0, 2**32)))
        )
        seqs.append(sense_filter(s))

    out: dict = {"synonymous": {}, "mutation": {}, "truncation": {}}

    # synonymous (true) substitution sweep
    print("  synonymous substitution:")
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v_hits = w_hits = 0
        vz_list, wz_list = [], []
        for s in seqs:
            a = apply_true_synonymous_substitutions(s, rate, rng, frame=0) if rate > 0 else s
            vr = vanilla.detect(a)
            wr = wobble.detect(a)
            v_hits += vr.is_watermarked
            w_hits += wr.is_watermarked
            vz_list.append(vr.z_score)
            wz_list.append(wr.z_score)
        out["synonymous"][rate] = {
            "vanilla_tpr": v_hits / len(seqs),
            "wobble_tpr": w_hits / len(seqs),
            "vanilla_mean_z": float(np.mean(vz_list)),
            "wobble_mean_z": float(np.mean(wz_list)),
        }
        r = out["synonymous"][rate]
        print(f"    rate={rate}: vanilla TPR={r['vanilla_tpr']:.2f} (mean_z={r['vanilla_mean_z']:.2f}), "
              f"wobble TPR={r['wobble_tpr']:.2f} (mean_z={r['wobble_mean_z']:.2f})")

    # random point mutations
    print("  point mutation:")
    for rate in [0.0, 0.05, 0.10, 0.15, 0.20]:
        v_hits = w_hits = 0
        vz_list, wz_list = [], []
        for s in seqs:
            a = apply_random_mutations(s, rate, rng) if rate > 0 else s
            vz_list.append(vanilla.detect(a).z_score)
            wz_list.append(wobble.detect(a).z_score)
            v_hits += vanilla.detect(a).is_watermarked
            w_hits += wobble.detect(a).is_watermarked
        out["mutation"][rate] = {
            "vanilla_tpr": v_hits / len(seqs),
            "wobble_tpr": w_hits / len(seqs),
            "vanilla_mean_z": float(np.mean(vz_list)),
            "wobble_mean_z": float(np.mean(wz_list)),
        }
        r = out["mutation"][rate]
        print(f"    rate={rate}: vanilla TPR={r['vanilla_tpr']:.2f}, "
              f"wobble TPR={r['wobble_tpr']:.2f}")

    # truncation
    print("  truncation:")
    for keep in [1.0, 0.75, 0.5, 0.33, 0.25]:
        v_hits = w_hits = 0
        for s in seqs:
            a = apply_truncation(s, keep)
            if len(a) <= CFG.context_width + 1:
                continue
            v_hits += vanilla.detect(a).is_watermarked
            w_hits += wobble.detect(a).is_watermarked
        out["truncation"][keep] = {
            "vanilla_tpr": v_hits / len(seqs),
            "wobble_tpr": w_hits / len(seqs),
        }
        r = out["truncation"][keep]
        print(f"    keep={keep}: vanilla TPR={r['vanilla_tpr']:.2f}, "
              f"wobble TPR={r['wobble_tpr']:.2f}")

    return out


def main() -> None:
    print(f"Config: key={len(KEY)}B, k={CFG.context_width}, gamma={CFG.gamma}, "
          f"delta={CFG.delta}, alpha={ALPHA}, seq_len={SEQ_LEN}")
    results = {
        "tier1": run_tier1_monte_carlo(),
        "tier2": run_tier2_evo2_unwatermarked(),
        "tier3": run_tier3_ncbi(),
        "tier5": run_tier5_robustness(),
    }
    out_path = Path("wobble_gauntlet_results.json")
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    main()
