# codon position mismatch analysis
#
# the question: do evo2's generated sequences look like memorization or
# generalization? we look at WHERE the mismatches land relative to codon
# structure.
#
# key idea:
# - if mismatches pile up at the 3rd codon position (the wobble base), evo2
#   is making NOVEL coding sequences that keep the protein working. that's
#   generalization.
# - if mismatches are spread evenly across codon positions, evo2 is just
#   imperfectly recalling a memorized sequence with random noise. that's
#   memorization with errors.
#
# input: a TSV file with columns:
#     generated_id    generated_fasta    reference_fasta
#
#     generated_fasta is the path to your generated sequence (FASTA) and
#     reference_fasta is the path to the closest BLAST hit reference (FASTA)
#
# output: per-sequence and aggregate stats on the codon position mismatch
#         distribution, plus a chi-squared test against a uniform
#         distribution.
#
# requirements: pip install biopython matplotlib scipy --break-system-packages

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from Bio import SeqIO, pairwise2
    from Bio.Seq import Seq
except ImportError:
    print("ERROR: BioPython not installed.")
    print("  pip install biopython --break-system-packages")
    sys.exit(1)

try:
    from scipy.stats import chisquare
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: scipy/matplotlib/numpy not installed.")
    print("  pip install scipy matplotlib numpy --break-system-packages")
    sys.exit(1)


def read_fasta(filepath):
    # first sequence in the FASTA file.
    for record in SeqIO.parse(filepath, "fasta"):
        return str(record.seq).upper()
    raise ValueError(f"No sequences found in {filepath}")


def needle_align(seq1, seq2):
    # global pairwise alignment (Needleman-Wunsch).
    alignments = pairwise2.align.globalms(
        seq1, seq2,
        match=2, mismatch=-1, open=-5, extend=-0.5,
        one_alignment_only=True
    )
    if not alignments:
        raise ValueError("Alignment failed")
    a = alignments[0]
    return a.seqA, a.seqB


def analyze_codon_mismatches(aligned_gen, aligned_ref):
    # walk the alignment in the REFERENCE reading frame. for each mismatch
    # at an ungapped reference position, note which codon position (0, 1, 2)
    # it falls in.
    codon_pos_counts = Counter()
    total_mismatches = 0
    total_compared = 0
    synonymous_count = 0
    nonsynonymous_count = 0

    ref_pos = 0
    ref_positions = []

    for i in range(len(aligned_ref)):
        if aligned_ref[i] == '-':
            continue
        if aligned_gen[i] == '-':
            ref_pos += 1
            continue
        ref_positions.append((ref_pos, aligned_gen[i], aligned_ref[i]))
        ref_pos += 1

    total_compared = len(ref_positions)

    for ungapped_pos, gen_base, ref_base in ref_positions:
        codon_position = ungapped_pos % 3

        if gen_base != ref_base:
            total_mismatches += 1
            codon_pos_counts[codon_position] += 1

            if codon_position == 2:
                # wobble mismatch: try to rebuild the full codon so we can
                # tell synonymous from nonsynonymous
                codon_start = ungapped_pos - 2
                ref_codon_bases = []
                gen_codon_bases = []
                found_all = True
                for offset in range(3):
                    target_pos = codon_start + offset
                    matching = [x for x in ref_positions if x[0] == target_pos]
                    if matching:
                        gen_codon_bases.append(matching[0][1])
                        ref_codon_bases.append(matching[0][2])
                    else:
                        found_all = False
                        break

                if found_all and len(ref_codon_bases) == 3 and len(gen_codon_bases) == 3:
                    ref_codon = ''.join(ref_codon_bases)
                    gen_codon = ''.join(gen_codon_bases)
                    try:
                        ref_aa = str(Seq(ref_codon).translate())
                        gen_aa = str(Seq(gen_codon).translate())
                        if ref_aa == gen_aa:
                            synonymous_count += 1
                        else:
                            nonsynonymous_count += 1
                    except Exception:
                        pass

    return codon_pos_counts, total_mismatches, total_compared, synonymous_count, nonsynonymous_count


def run_analysis(input_file, output_dir):
    # run the full codon mismatch analysis.
    os.makedirs(output_dir, exist_ok=True)

    results = []
    aggregate_counts = Counter()
    aggregate_syn = 0
    aggregate_nonsyn = 0

    with open(input_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gen_id = row['generated_id']
            gen_path = row['generated_fasta']
            ref_path = row['reference_fasta']

            print(f"\nAnalyzing: {gen_id}")

            try:
                gen_seq = read_fasta(gen_path)
                ref_seq = read_fasta(ref_path)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            print(f"  Generated: {len(gen_seq)} bp, Reference: {len(ref_seq)} bp")
            print(f"  Aligning...")

            try:
                aligned_gen, aligned_ref = needle_align(gen_seq, ref_seq)
            except Exception as e:
                print(f"  SKIP alignment: {e}")
                continue

            counts, total_mm, total_cmp, syn, nonsyn = analyze_codon_mismatches(aligned_gen, aligned_ref)

            p1 = counts.get(0, 0)
            p2 = counts.get(1, 0)
            p3 = counts.get(2, 0)

            aggregate_counts += counts
            aggregate_syn += syn
            aggregate_nonsyn += nonsyn

            mismatch_rate = total_mm / total_cmp if total_cmp > 0 else 0

            result = {
                'id': gen_id,
                'total_compared': total_cmp,
                'total_mismatches': total_mm,
                'mismatch_rate': round(mismatch_rate, 4),
                'pos1_count': p1,
                'pos2_count': p2,
                'pos3_wobble_count': p3,
                'synonymous_at_pos3': syn,
                'nonsynonymous_at_pos3': nonsyn,
            }
            results.append(result)

            print(f"  Mismatches: {total_mm}/{total_cmp} ({mismatch_rate*100:.1f}%)")
            print(f"  Position 1: {p1}, Position 2: {p2}, Position 3 (wobble): {p3}")
            if syn + nonsyn > 0:
                print(f"  At wobble: {syn} synonymous, {nonsyn} nonsynonymous")

    if not results:
        print("No results to analyze.")
        return

    # roll everything up across sequences
    total_mm = sum(aggregate_counts.values())
    p1 = aggregate_counts.get(0, 0)
    p2 = aggregate_counts.get(1, 0)
    p3 = aggregate_counts.get(2, 0)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE CODON POSITION MISMATCH ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Total mismatches: {total_mm}")
    print(f"  Position 1: {p1} ({p1/total_mm*100:.1f}%)" if total_mm > 0 else "  Position 1: 0")
    print(f"  Position 2: {p2} ({p2/total_mm*100:.1f}%)" if total_mm > 0 else "  Position 2: 0")
    print(f"  Position 3: {p3} ({p3/total_mm*100:.1f}%)" if total_mm > 0 else "  Position 3: 0")

    # chi-squared test against a uniform distribution
    if total_mm >= 10:
        expected = total_mm / 3
        chi2, p_value = chisquare([p1, p2, p3], f_exp=[expected, expected, expected])
        print(f"\n  Chi-squared test (vs uniform): chi2={chi2:.2f}, p={p_value:.4f}")
        if p_value < 0.05 and p3 > p1 and p3 > p2:
            print(f"  RESULT: Wobble position enrichment is SIGNIFICANT (p < 0.05)")
            print(f"  INTERPRETATION: Evidence of GENERALIZATION -- Evo2 preferentially")
            print(f"  preserves protein function via synonymous substitutions.")
        elif p_value < 0.05:
            print(f"  RESULT: Non-uniform distribution but wobble NOT dominant.")
        else:
            print(f"  RESULT: Distribution is consistent with UNIFORM (random noise).")
            print(f"  INTERPRETATION: Consistent with MEMORIZATION with random errors.")
    else:
        chi2, p_value = None, None
        print(f"\n  Too few mismatches ({total_mm}) for chi-squared test (need >= 10).")

    if aggregate_syn + aggregate_nonsyn > 0:
        syn_ratio = aggregate_syn / (aggregate_syn + aggregate_nonsyn)
        print(f"\n  Wobble synonymous ratio: {aggregate_syn}/{aggregate_syn + aggregate_nonsyn} ({syn_ratio*100:.1f}%)")
        print(f"  (Random expectation: ~25% synonymous due to codon degeneracy)")
        if syn_ratio > 0.5:
            print(f"  STRONG evidence of functional preservation.")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    positions = ['1st\n(codon pos 1)', '2nd\n(codon pos 2)', '3rd\n(wobble)']
    counts_list = [p1, p2, p3]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    axes[0].bar(positions, counts_list, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Mismatch Count')
    axes[0].set_title('Mismatches by Codon Position')
    if total_mm > 0:
        expected_line = total_mm / 3
        axes[0].axhline(y=expected_line, color='gray', linestyle='--', label=f'Uniform expected ({expected_line:.0f})')
        axes[0].legend()

    if aggregate_syn + aggregate_nonsyn > 0:
        axes[1].pie([aggregate_syn, aggregate_nonsyn],
                    labels=['Synonymous', 'Nonsynonymous'],
                    colors=['#2ecc71', '#e74c3c'],
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[1].set_title('3rd Position Mismatches:\nSynonymous vs Nonsynonymous')
    else:
        axes[1].text(0.5, 0.5, 'No 3rd-position\nmismatches to analyze',
                     ha='center', va='center', fontsize=12)
        axes[1].set_title('3rd Position Mismatches')

    ids = [r['id'][:20] for r in results]
    rates = [r['mismatch_rate'] * 100 for r in results]
    axes[2].barh(ids, rates, color='#9b59b6', edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Mismatch Rate (%)')
    axes[2].set_title('Per-Sequence Mismatch Rate')
    axes[2].invert_yaxis()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'codon_mismatch_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    results_path = os.path.join(output_dir, 'codon_mismatch_results.json')
    output = {
        'per_sequence': results,
        'aggregate': {
            'total_mismatches': total_mm,
            'position_1': p1,
            'position_2': p2,
            'position_3_wobble': p3,
            'synonymous_at_pos3': aggregate_syn,
            'nonsynonymous_at_pos3': aggregate_nonsyn,
            'chi2': chi2,
            'p_value': p_value,
        }
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Codon Position Mismatch Analysis')
    parser.add_argument('input_file', help='TSV: generated_id, generated_fasta, reference_fasta')
    parser.add_argument('-o', '--output_dir', default='./results/codon_analysis')
    args = parser.parse_args()
    run_analysis(args.input_file, args.output_dir)
