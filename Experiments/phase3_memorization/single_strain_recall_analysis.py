# phylogenetic placement analysis
#
# the question: are evo2's generated sequences reproductions of one known
# strain (memorization), or novel sequences that fall BETWEEN known strains
# in sequence space (generalization)?
#
# method:
# 1. take each generated sequence and its BLAST XML results.
# 2. compute pairwise identity to the top N hits.
# 3. classify:
#    - EXACT_RECALL: one hit >= 99% identity, next-best < 95%. the model
#      recalled one strain.
#    - NEAR_COPY: one hit >= 97%, gap to next > 3%. mostly one strain with
#      minor drift.
#    - INTERPOLATION: top 2+ hits within 2% of each other. the sequence
#      sits between strains.
#    - DIVERGENT: best hit < 93%. the sequence is substantially different
#      from all known strains.
#
# if INTERPOLATION dominates -> generalization. evo2 learned the viral family
# manifold. if EXACT_RECALL dominates -> memorization. evo2 is reproducing
# training data.
#
# input: a TSV file with columns:
#     generated_id    generated_fasta    blast_xml
#
#     blast_xml is the path to BLAST XML output (format 5) for that sequence.
#     run megablast against nt with output format XML.
#
# output: per-sequence classification and aggregate summary.
#
# requirements: pip install biopython matplotlib --break-system-packages

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from Bio import SeqIO
    from Bio.Blast import NCBIXML
except ImportError:
    print("ERROR: BioPython not installed.")
    print("  pip install biopython --break-system-packages")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib/numpy not installed.")
    print("  pip install matplotlib numpy --break-system-packages")
    sys.exit(1)


def parse_blast_xml(xml_path, max_hits=20):
    # parse BLAST XML, pull the top hits with identity and coverage.
    hits = []
    with open(xml_path) as f:
        try:
            records = NCBIXML.parse(f)
            record = next(records)
        except Exception as e:
            print(f"  BLAST XML parse error: {e}")
            return hits

    query_length = record.query_length

    for alignment in record.alignments[:max_hits]:
        best_hsp = max(alignment.hsps, key=lambda h: h.score)
        identity = best_hsp.identities / best_hsp.align_length * 100
        coverage = best_hsp.align_length / query_length * 100 if query_length > 0 else 0

        hits.append({
            'accession': alignment.accession,
            'title': alignment.hit_def[:120],
            'identity': round(identity, 2),
            'coverage': round(coverage, 2),
            'score': best_hsp.score,
            'evalue': best_hsp.expect,
            'align_length': best_hsp.align_length,
        })

    return hits


def classify_sequence(hits):
    # classify a generated sequence from its BLAST hit distribution.
    # returns (category, details_dict).
    if not hits:
        return 'NO_HITS', {'reason': 'No BLAST hits found'}

    if len(hits) == 1:
        h = hits[0]
        if h['identity'] >= 99:
            return 'EXACT_RECALL', {
                'best_identity': h['identity'],
                'best_accession': h['accession'],
                'best_title': h['title'],
                'reason': 'Single hit with >= 99% identity'
            }
        elif h['identity'] >= 93:
            return 'NEAR_COPY', {
                'best_identity': h['identity'],
                'best_accession': h['accession'],
                'best_title': h['title'],
                'reason': 'Single hit with >= 93% identity'
            }
        else:
            return 'DIVERGENT', {
                'best_identity': h['identity'],
                'best_accession': h['accession'],
                'best_title': h['title'],
                'reason': 'Single hit below 93% identity'
            }

    best = hits[0]
    second = hits[1]
    gap = best['identity'] - second['identity']

    if best['identity'] >= 99 and gap > 4:
        category = 'EXACT_RECALL'
        reason = f"Top hit {best['identity']:.1f}%, gap to second {gap:.1f}pp"
    elif best['identity'] >= 97 and gap > 3:
        category = 'NEAR_COPY'
        reason = f"Top hit {best['identity']:.1f}%, gap to second {gap:.1f}pp"
    elif best['identity'] >= 90 and gap <= 2:
        # top hits are clustered, so the sequence sits between strains.
        # count how many hits are within 2% of the best one.
        cluster = [h for h in hits if best['identity'] - h['identity'] <= 2]
        category = 'INTERPOLATION'
        reason = f"Top {len(cluster)} hits within 2% of each other ({best['identity']:.1f}% to {cluster[-1]['identity']:.1f}%)"
    elif best['identity'] >= 93:
        category = 'NEAR_COPY'
        reason = f"Top hit {best['identity']:.1f}%, moderate gap {gap:.1f}pp"
    elif best['identity'] >= 90:
        # hits are spread out
        category = 'DIVERGENT'
        reason = f"Top hit {best['identity']:.1f}% with spread hits"
    else:
        category = 'BELOW_THRESHOLD'
        reason = f"Best identity {best['identity']:.1f}% below 90%"

    details = {
        'best_identity': best['identity'],
        'second_identity': second['identity'],
        'identity_gap': round(gap, 2),
        'best_accession': best['accession'],
        'best_title': best['title'],
        'second_accession': second['accession'],
        'second_title': second['title'],
        'num_hits_within_2pct': len([h for h in hits if best['identity'] - h['identity'] <= 2]),
        'reason': reason,
    }
    return category, details


def run_analysis(input_file, output_dir):
    # run the full phylogenetic placement analysis.
    os.makedirs(output_dir, exist_ok=True)

    results = []
    categories = Counter()

    with open(input_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gen_id = row['generated_id']
            blast_xml = row['blast_xml']

            print(f"\nAnalyzing: {gen_id}")

            if not os.path.exists(blast_xml):
                print(f"  SKIP: BLAST XML not found at {blast_xml}")
                continue

            hits = parse_blast_xml(blast_xml)
            if not hits:
                print(f"  SKIP: No BLAST hits parsed")
                continue

            category, details = classify_sequence(hits)
            categories[category] += 1

            result = {
                'id': gen_id,
                'category': category,
                'top_hits': hits[:5],
                **details,
            }
            results.append(result)

            print(f"  Category: {category}")
            print(f"  {details['reason']}")
            print(f"  Best: {hits[0]['accession']} - {hits[0]['title'][:60]}")
            if len(hits) > 1:
                print(f"  2nd:  {hits[1]['accession']} - {hits[1]['title'][:60]}")

    if not results:
        print("No results to analyze.")
        return

    # summary
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"PHYLOGENETIC PLACEMENT SUMMARY")
    print(f"{'=' * 60}")

    for cat in ['EXACT_RECALL', 'NEAR_COPY', 'INTERPOLATION', 'DIVERGENT', 'BELOW_THRESHOLD', 'NO_HITS']:
        count = categories.get(cat, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {cat}: {count}/{total} ({pct:.1f}%)")

    recall_pct = categories.get('EXACT_RECALL', 0) / total * 100 if total > 0 else 0
    interp_pct = categories.get('INTERPOLATION', 0) / total * 100 if total > 0 else 0

    print(f"\nINTERPRETATION:")
    if recall_pct > 50:
        print(f"  MAJORITY EXACT RECALL: Evo2 is primarily reproducing memorized strains.")
    elif interp_pct > 30:
        print(f"  SIGNIFICANT INTERPOLATION: Evo2 is generating novel sequences between known strains.")
        print(f"  This is evidence of GENERALIZATION on the viral sequence manifold.")
    else:
        print(f"  MIXED: Both memorization and generalization contribute.")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cat_labels = ['EXACT_RECALL', 'NEAR_COPY', 'INTERPOLATION', 'DIVERGENT', 'BELOW_THRESHOLD']
    cat_counts = [categories.get(c, 0) for c in cat_labels]
    cat_colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#95a5a6']

    nonzero = [(l, c, col) for l, c, col in zip(cat_labels, cat_counts, cat_colors) if c > 0]
    if nonzero:
        labels, counts_nz, colors_nz = zip(*nonzero)
        axes[0].pie(counts_nz, labels=labels, colors=colors_nz, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 9})
    axes[0].set_title('Sequence Classification')

    # identity vs gap scatter
    identities = [r['best_identity'] for r in results]
    gaps = [r.get('identity_gap', 0) for r in results]
    colors_scatter = []
    for r in results:
        cat = r['category']
        idx = cat_labels.index(cat) if cat in cat_labels else 4
        colors_scatter.append(cat_colors[idx])

    axes[1].scatter(identities, gaps, c=colors_scatter, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
    axes[1].set_xlabel('Best Hit Identity (%)')
    axes[1].set_ylabel('Gap to Second Hit (pp)')
    axes[1].set_title('Identity vs Gap Distribution')
    axes[1].axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='3pp gap threshold')
    axes[1].axvline(x=99, color='gray', linestyle=':', alpha=0.5, label='99% identity')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'phylogenetic_placement.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

    results_path = os.path.join(output_dir, 'phylogenetic_placement_results.json')
    output = {
        'per_sequence': results,
        'aggregate': dict(categories),
        'total_sequences': total,
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phylogenetic Placement Analysis')
    parser.add_argument('input_file', help='TSV: generated_id, blast_xml')
    parser.add_argument('-o', '--output_dir', default='./results/phylo_analysis')
    args = parser.parse_args()
    run_analysis(args.input_file, args.output_dir)
