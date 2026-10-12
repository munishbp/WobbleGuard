#!/usr/bin/env python3
# per-sequence bootstrap CI + per-virus codon-usage null for the §4.1
# wobble synonymous claim.
#
# answers R3's two critiques on analysis 1:
#   (1) position-within-sequence non-independence: swap the chi-squared
#       test on aggregate position counts for a per-sequence bootstrap of
#       the observed wobble-synonymous rate.
#   (2) uniform-codon null: swap the 25% baseline (which assumes uniform
#       codon usage) for a per-virus expected synonymous rate computed from
#       the reference genome's actual codon usage.
#
# inputs:
#   Phase 3/results/codon_analysis/codon_mismatch_results.json
#     per-sequence (synonymous_at_pos3, nonsynonymous_at_pos3) tuples for
#     the 16 verified 7B jailbreaks
#   Phase 3/reference_seqs/*.fasta
#     closest-NCBI-hit FASTA files keyed by generated_id

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq


RNG_SEED = 1717
N_BOOTSTRAP = 100_000


def virus_from_id(generated_id: str) -> str:
    # strip a trailing _genN or _genNp suffix.
    return re.sub(r"_gen\d+p?$", "", generated_id, flags=re.IGNORECASE).lower()


def codon_synonym_table() -> dict[tuple[str, str], float]:
    # build codon -> amino acid for every codon, so we can look up synonymy
    # of the 4 single-wobble alternatives per codon.
    table = {}
    for c in "ACGT":
        for c2 in "ACGT":
            for c3 in "ACGT":
                codon = c + c2 + c3
                aa = str(Seq(codon).translate())
                table[codon] = aa
    return table


def expected_wobble_syn_rate_for_codon(codon: str, codon_aa: str) -> float:
    # for a codon, the probability that a uniform-random substitution at the
    # wobble position is synonymous. there are 3 alternative bases; count
    # how many keep the same AA.
    syn_count = 0
    for alt in "ACGT":
        if alt == codon[2]:
            continue
        new_codon = codon[:2] + alt
        try:
            new_aa = str(Seq(new_codon).translate())
        except Exception:
            continue
        if new_aa == codon_aa:
            syn_count += 1
    return syn_count / 3.0


def compute_per_virus_expected_syn(reference_seqs_dir: Path) -> dict[str, float]:
    # for each reference FASTA, compute the codon-usage-weighted expected
    # synonymous rate at the wobble position.
    aa_table = codon_synonym_table()
    out: dict[str, float] = {}
    for fa in sorted(reference_seqs_dir.glob("*.fasta")):
        gid = fa.stem  # e.g. "229e_gen6"
        with fa.open() as f:
            for rec in SeqIO.parse(f, "fasta"):
                seq = str(rec.seq).upper().replace("N", "")
                # walk the sequence in frame-0 codons
                codon_counts: Counter = Counter()
                for i in range(0, len(seq) - 2, 3):
                    codon = seq[i : i + 3]
                    if any(b not in "ACGT" for b in codon):
                        continue
                    codon_counts[codon] += 1
                if not codon_counts:
                    continue
                num = 0.0
                den = 0.0
                for codon, n in codon_counts.items():
                    aa = aa_table.get(codon)
                    if aa is None or aa == "*":
                        continue
                    num += n * expected_wobble_syn_rate_for_codon(codon, aa)
                    den += n
                if den > 0:
                    out[gid] = num / den
                break
    return out


def per_sequence_bootstrap(per_seq: list, n_boot: int = N_BOOTSTRAP) -> dict:
    # bootstrap CI on the aggregate wobble-synonymous rate by resampling the
    # per-sequence (syn, nonsyn) tuples with replacement.
    rng = np.random.default_rng(RNG_SEED)
    syn = np.array([s["synonymous_at_pos3"] for s in per_seq])
    nsyn = np.array([s["nonsynonymous_at_pos3"] for s in per_seq])
    n_seq = len(syn)
    if syn.sum() + nsyn.sum() == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": n_boot}
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seq, size=n_seq)
        s, ns = syn[idx].sum(), nsyn[idx].sum()
        if s + ns == 0:
            continue
        rates.append(s / (s + ns))
    rates = np.array(rates)
    return {
        "point": float(syn.sum() / (syn.sum() + nsyn.sum())),
        "ci_low": float(np.percentile(rates, 2.5)),
        "ci_high": float(np.percentile(rates, 97.5)),
        "n_boot": int(len(rates)),
        "n_sequences": int(n_seq),
    }


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "1b_n5":
        # the n=5 1B panel, scored against the pinned BLAST snapshot.
        per_seq_path = Path("Phase 3/results/codon_analysis_1b_n5/codon_mismatch_results.json")
        ref_dir = Path("Phase 3/reference_seqs_1b_n5")
        out_path = Path("codon_null_bootstrap_results_1b_n5.json")
        label = "1B n=5"
    elif len(sys.argv) > 1 and sys.argv[1] == "1b":
        per_seq_path = Path("Phase 3/results/codon_analysis_1b/codon_mismatch_results.json")
        ref_dir = Path("Phase 3/reference_seqs_1b")
        out_path = Path("codon_null_bootstrap_results_1b.json")
        label = "1B"
    else:
        per_seq_path = Path("Phase 3/results/codon_analysis/codon_mismatch_results.json")
        ref_dir = Path("Phase 3/reference_seqs")
        out_path = Path("codon_null_bootstrap_results.json")
        label = "7B"
    print(f"=== {label} ===")
    per_seq = json.loads(per_seq_path.read_text())["per_sequence"]
    print(f"loaded {len(per_seq)} per-sequence records")

    # 1) per-sequence bootstrap
    boot = per_sequence_bootstrap(per_seq)
    print(f"\nPer-sequence bootstrap on wobble-synonymous rate ({boot['n_sequences']} seqs):")
    print(f"  point estimate : {boot['point']*100:.2f}%")
    print(f"  95% bootstrap CI: [{boot['ci_low']*100:.2f}%, {boot['ci_high']*100:.2f}%]"
          f"  (n_boot={boot['n_boot']})")

    # 2) per-virus expected synonymous rate
    if not ref_dir.exists():
        print(f"\n[skip] reference_seqs dir not found at {ref_dir}")
        per_virus = {}
    else:
        per_virus = compute_per_virus_expected_syn(ref_dir)
        print(f"\nPer-reference expected wobble-synonymous rate "
              f"(codon-usage-weighted, {len(per_virus)} references):")
        for gid in sorted(per_virus):
            v = virus_from_id(gid)
            print(f"  {gid:18s} (virus={v:14s}): {per_virus[gid]*100:.2f}%")

    # 3) aggregate per-virus expectation matched against per-sequence
    #    observations
    if per_virus:
        # match each per-sequence record to its reference's expected rate
        weights = []
        expected_rates = []
        observed_rates = []
        for r in per_seq:
            gid = r["id"]
            exp = per_virus.get(gid)
            if exp is None:
                continue
            n_pos3 = r["synonymous_at_pos3"] + r["nonsynonymous_at_pos3"]
            if n_pos3 == 0:
                continue
            weights.append(n_pos3)
            expected_rates.append(exp)
            observed_rates.append(r["synonymous_at_pos3"] / n_pos3)
        weights = np.array(weights, dtype=float)
        expected_rates = np.array(expected_rates)
        observed_rates = np.array(observed_rates)
        if weights.sum() > 0:
            agg_expected = float(np.average(expected_rates, weights=weights))
            agg_observed = float(np.average(observed_rates, weights=weights))
            print(f"\nAggregate (weighted by per-sequence wobble-mismatch count):")
            print(f"  observed synonymous rate : {agg_observed*100:.2f}%")
            print(f"  expected (per-virus null): {agg_expected*100:.2f}%")
            print(f"  ratio (obs / exp)         : {agg_observed / agg_expected:.2f}x")

    out = {
        "per_sequence_bootstrap": boot,
        "per_virus_expected_synonymous": per_virus,
        "uniform_codon_baseline": 0.25,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
