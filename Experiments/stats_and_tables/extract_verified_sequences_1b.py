#!/usr/bin/env python3
# derive verified_jailbreak_sequences_1b.csv from 1B_all_results.json.
#
# basically a copy of extract_jailbreak_sequences.py, except it reads the
# consolidated all_results.json (one file, all viruses) instead of per-virus
# subdirs, and it drops false positives using the model == 'Evo2-1B' rows in
# fp_details.csv.

import csv
import json


def load_false_positives(path: str = "fp_details.csv") -> set:
    # (virus, generation_index, blast_type) tuples for every 1B false positive
    fps = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["model"] == "Evo2-1B":
                fps.add((row["virus"], int(row["generation_index"]), row["blast_type"]))
    return fps


def extract(all_results_path: str = "1B_all_results.json") -> list:
    with open(all_results_path) as f:
        data = json.load(f)
    fps = load_false_positives()
    rows = []

    for virus_entry in data:
        virus_key = virus_entry["virus"]
        generations = virus_entry.get("generations", [])
        blast_results = virus_entry.get("blast_results", [])

        # map generation index -> generated dna
        sequences_by_index = {
            idx: gen.get("generated_dna", "")
            for idx, gen in enumerate(generations)
        }

        for entry in blast_results:
            gen_idx = entry["generation_index"]

            # only keep claimed blastn successes that aren't in the FP list
            if entry.get("jailbreak_success_n", False):
                if (virus_key, gen_idx, "blastn") not in fps:
                    top_hit = entry["blastn"][0] if entry.get("blastn") else {}
                    rows.append({
                        "virus": virus_key,
                        "generation_index": gen_idx,
                        "sequence_type": "DNA",
                        "generated_sequence": sequences_by_index.get(gen_idx, ""),
                        "sequence_length": len(sequences_by_index.get(gen_idx, "")),
                        "homolog_accession": entry.get("homolog_accession", ""),
                        "blast_type": "BLASTn",
                        "blast_identity_pct": entry.get("best_blastn_pct", 0),
                        "top_hit_title": top_hit.get("title", "")[:100],
                        "top_hit_accession": top_hit.get("accession", ""),
                        "top_hit_organism": top_hit.get("organism", ""),
                    })

            # same for blastp
            if entry.get("jailbreak_success_p", False):
                if (virus_key, gen_idx, "blastp") not in fps:
                    top_hit = entry["blastp"][0] if entry.get("blastp") else {}
                    rows.append({
                        "virus": virus_key,
                        "generation_index": gen_idx,
                        "sequence_type": "Protein_from_DNA",
                        "generated_sequence": sequences_by_index.get(gen_idx, ""),
                        "sequence_length": len(sequences_by_index.get(gen_idx, "")),
                        "homolog_accession": entry.get("homolog_accession", ""),
                        "blast_type": "BLASTp",
                        "blast_identity_pct": entry.get("best_blastp_pct", 0),
                        "top_hit_title": top_hit.get("title", "")[:100],
                        "top_hit_accession": top_hit.get("accession", ""),
                        "top_hit_organism": top_hit.get("organism", ""),
                    })

    return rows


def main():
    rows = extract()
    out_path = "verified_jailbreak_sequences_1b.csv"
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Extracted {len(rows)} verified 1B jailbreak rows to {out_path}")

    by_virus = {}
    for r in rows:
        by_virus[r["virus"]] = by_virus.get(r["virus"], 0) + 1
    print("Breakdown by virus:")
    for v, n in sorted(by_virus.items()):
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
