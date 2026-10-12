#!/usr/bin/env python3
# pull out the DNA sequences behind verified 7B jailbreak successes.
# writes a csv with virus, generation_index, the dna sequence, and the blast
# verification details.

import json
import csv
import os
from pathlib import Path

def load_false_positives():
    # (virus, generation_index, blast_type) tuples from fp_details.csv
    false_positives = set()

    with open('fp_details.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == 'Evo2-7B':
                # build a unique key for each FP entry
                key = (row['virus'], int(row['generation_index']), row['blast_type'])
                false_positives.add(key)

    return false_positives

def extract_verified_sequences():
    # walk the 7B bundle, keep claimed successes that aren't FPs
    bundle_dir = Path(os.environ.get("EVO2_RESULTS_BUNDLE", "results")) / "evo2_7b_20260326_103204"
    false_positives = load_false_positives()
    verified_sequences = []

    for virus_dir in sorted(bundle_dir.iterdir()):
        if not virus_dir.is_dir():
            continue

        virus_key = virus_dir.name
        blast_details_path = virus_dir / "blast_details.json"
        result_path = virus_dir / "result.json"

        if not blast_details_path.exists() or not result_path.exists():
            continue

        # load the blast details and the generations
        with open(blast_details_path) as f:
            blast_details = json.load(f)

        with open(result_path) as f:
            result_data = json.load(f)

        # generation index -> generated dna
        sequences_by_index = {}
        for idx, gen in enumerate(result_data.get('generations', [])):
            sequences_by_index[idx] = gen.get('generated_dna', '')

        # check each generation for verified successes
        for entry in blast_details:
            gen_idx = entry['generation_index']

            # blastn successes
            if entry.get('jailbreak_success_n', False):
                fp_key = (virus_key, gen_idx, 'blastn')
                if fp_key not in false_positives:  # not an FP = verified success
                    top_hit = entry['blastn'][0] if entry.get('blastn') else {}
                    sequence = sequences_by_index.get(gen_idx, '')

                    verified_sequences.append({
                        'virus': virus_key,
                        'generation_index': gen_idx,
                        'sequence_type': 'DNA',
                        'generated_sequence': sequence,
                        'sequence_length': len(sequence),
                        'homolog_accession': entry.get('homolog_accession', ''),
                        'blast_type': 'BLASTn',
                        'blast_identity_pct': entry.get('best_blastn_pct', 0),
                        'top_hit_title': top_hit.get('title', '')[:100],
                        'top_hit_accession': top_hit.get('accession', ''),
                        'top_hit_organism': top_hit.get('organism', '')
                    })

            # blastp successes
            if entry.get('jailbreak_success_p', False):
                fp_key = (virus_key, gen_idx, 'blastp')
                if fp_key not in false_positives:  # not an FP = verified success
                    top_hit = entry['blastp'][0] if entry.get('blastp') else {}
                    sequence = sequences_by_index.get(gen_idx, '')

                    verified_sequences.append({
                        'virus': virus_key,
                        'generation_index': gen_idx,
                        'sequence_type': 'Protein_from_DNA',
                        'generated_sequence': sequence,  # still store the dna
                        'sequence_length': len(sequence),
                        'homolog_accession': entry.get('homolog_accession', ''),
                        'blast_type': 'BLASTp',
                        'blast_identity_pct': entry.get('best_blastp_pct', 0),
                        'top_hit_title': top_hit.get('title', '')[:100],
                        'top_hit_accession': top_hit.get('accession', ''),
                        'top_hit_organism': top_hit.get('organism', '')
                    })

    return verified_sequences

def main():
    print("Extracting verified jailbreak sequences from 7B results...")
    verified_sequences = extract_verified_sequences()

    # write the csv
    with open('verified_jailbreak_sequences_7b.csv', 'w', newline='') as f:
        if verified_sequences:
            fieldnames = verified_sequences[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(verified_sequences)

    print(f"Extracted {len(verified_sequences)} verified jailbreak sequences")
    print("Output: verified_jailbreak_sequences_7b.csv")

    # quick breakdown by virus
    virus_counts = {}
    for seq in verified_sequences:
        virus = seq['virus']
        virus_counts[virus] = virus_counts.get(virus, 0) + 1

    print("\nBreakdown by virus:")
    for virus, count in sorted(virus_counts.items()):
        print(f"  {virus}: {count} verified sequences")

if __name__ == "__main__":
    main()
