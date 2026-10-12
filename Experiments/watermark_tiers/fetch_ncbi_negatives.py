#!/usr/bin/env python3
"""
Fetch natural pathogen sequences from NCBI for Tier 3 evaluation.

For each of the 23 viruses in JailbreakDNABench, downloads multiple
isolate genomes from NCBI Nucleotide and extracts non-overlapping
640bp windows as negative controls for watermark detection.

Requires: biopython (pip install biopython)

Usage: python fetch_ncbi_negatives.py [--max-isolates 20] [--email your@email.com]
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

# NCBI search queries per virus — species name + filters for complete genomes
VIRUS_QUERIES = {
    "hiv": '"Human immunodeficiency virus 1"[Organism] AND complete genome[Title]',
    "sars_cov2": '"Severe acute respiratory syndrome coronavirus 2"[Organism] AND complete genome[Title]',
    "hpv": '"Human papillomavirus"[Organism] AND complete genome[Title]',
    "hsv1": '"Human alphaherpesvirus 1"[Organism] AND complete genome[Title]',
    "vzv": '"Human alphaherpesvirus 3"[Organism] AND complete genome[Title]',
    "adenovirus": '"Human mastadenovirus"[Organism] AND complete genome[Title]',
    "varv": '"Variola virus"[Organism] AND complete genome[Title]',
    "parvovirus_b19": '"Primate erythroparvovirus 1"[Organism] AND complete genome[Title]',
    "mers": '"Middle East respiratory syndrome-related coronavirus"[Organism] AND complete genome[Title]',
    "oc43": '"Human coronavirus OC43"[Organism] AND complete genome[Title]',
    "hku1": '"Human coronavirus HKU1"[Organism] AND complete genome[Title]',
    "nl63": '"Human coronavirus NL63"[Organism] AND complete genome[Title]',
    "229e": '"Human coronavirus 229E"[Organism] AND complete genome[Title]',
    "jev": '"Japanese encephalitis virus"[Organism] AND complete genome[Title]',
    "dengue": '"Dengue virus"[Organism] AND complete genome[Title]',
    "hcv": '"Hepacivirus C"[Organism] AND complete genome[Title]',
    "rabies": '"Rabies lyssavirus"[Organism] AND complete genome[Title]',
    "measles": '"Measles morbillivirus"[Organism] AND complete genome[Title]',
    "mumps": '"Mumps orthorubulavirus"[Organism] AND complete genome[Title]',
    "reovirus": '"Mammalian orthoreovirus"[Organism]',
    "poliovirus": '"Enterovirus C"[Organism] AND poliovirus[Title] AND complete genome[Title]',
    "norovirus": '"Norovirus"[Organism] AND complete genome[Title]',
    "influenza": '"Influenza A virus"[Organism] AND complete genome[Title]',
}

# Reference accessions from our experiments (fallback if search yields few results)
REFERENCE_ACCESSIONS = {
    "hiv": "AF224507", "sars_cov2": "NC_045512", "hpv": "NC_001526",
    "hsv1": "AY446894", "vzv": "NC_001348", "adenovirus": "PP591774",
    "varv": "L22579", "parvovirus_b19": "AF043303", "mers": "JX869059",
    "oc43": "AY585228", "hku1": "AY597011", "nl63": "AY567487",
    "229e": "AF304460", "jev": "EF571853", "dengue": "NC_001477",
    "hcv": "JF735110", "rabies": "JX088694", "measles": "PP446317",
    "mumps": "EU370206", "reovirus": "NC_077840", "poliovirus": "AY184219",
    "norovirus": "OP649638", "influenza": "KU143292",
}

WINDOW_SIZE = 640
VALID_BASES = set("ACGT")

OUTPUT_PATH = Path("D:/Research Paper/ncbi_negative_sequences.json")


def is_clean_window(seq: str) -> bool:
    """Check that a 640bp window contains only A, C, G, T."""
    return all(c in VALID_BASES for c in seq.upper())


def extract_windows(sequence: str, window_size: int = WINDOW_SIZE) -> list[tuple[int, str]]:
    """Extract non-overlapping clean 640bp windows from a genome."""
    seq = sequence.upper()
    windows = []
    for start in range(0, len(seq) - window_size + 1, window_size):
        window = seq[start : start + window_size]
        if is_clean_window(window):
            windows.append((start, window))
    return windows


def fetch_sequences(email: str, max_isolates: int = 20) -> list[dict]:
    """
    Fetch natural pathogen sequences from NCBI.

    Returns list of dicts with: virus, accession, strain, genome_length,
    window_start, window_end, sequence.
    """
    from Bio import Entrez, SeqIO

    Entrez.email = email
    all_fragments = []

    for virus, query in VIRUS_QUERIES.items():
        print(f"\n  [{virus}] Searching NCBI: {query[:60]}...")

        try:
            # Search for accessions
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_isolates)
            result = Entrez.read(handle)
            handle.close()
            id_list = result["IdList"]

            if not id_list:
                # Fallback: fetch the reference accession directly
                ref = REFERENCE_ACCESSIONS.get(virus)
                if ref:
                    print(f"    No search results, falling back to reference {ref}")
                    handle = Entrez.esearch(db="nucleotide", term=ref, retmax=1)
                    result = Entrez.read(handle)
                    handle.close()
                    id_list = result["IdList"]

            if not id_list:
                print(f"    No sequences found for {virus}, skipping")
                continue

            print(f"    Found {len(id_list)} sequences, fetching...")

            # Fetch in batches to be polite to NCBI
            batch_size = 5
            virus_fragments = []

            for batch_start in range(0, len(id_list), batch_size):
                batch_ids = id_list[batch_start : batch_start + batch_size]
                handle = Entrez.efetch(
                    db="nucleotide", id=batch_ids, rettype="fasta", retmode="text",
                )
                records = list(SeqIO.parse(handle, "fasta"))
                handle.close()

                for record in records:
                    seq_str = str(record.seq).upper()
                    windows = extract_windows(seq_str)

                    for start, window_seq in windows:
                        virus_fragments.append({
                            "virus": virus,
                            "accession": record.id,
                            "description": record.description[:100],
                            "genome_length": len(seq_str),
                            "window_start": start,
                            "window_end": start + WINDOW_SIZE,
                            "sequence": window_seq,
                        })

                # NCBI rate limit: max 3 requests/second
                time.sleep(0.4)

            print(f"    Extracted {len(virus_fragments)} clean 640bp windows "
                  f"from {min(len(id_list), max_isolates)} isolates")
            all_fragments.extend(virus_fragments)

        except Exception as e:
            print(f"    ERROR for {virus}: {e}")
            continue

    return all_fragments


def main():
    parser = argparse.ArgumentParser(description="Fetch NCBI negative controls")
    parser.add_argument("--email", default="research@example.com",
                        help="Email for NCBI Entrez API (required by NCBI)")
    parser.add_argument("--max-isolates", type=int, default=20,
                        help="Max isolates to fetch per virus (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("Fetching natural pathogen sequences from NCBI")
    print(f"Viruses: {len(VIRUS_QUERIES)}, Max isolates/virus: {args.max_isolates}")
    print("=" * 60)

    fragments = fetch_sequences(args.email, args.max_isolates)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fragments, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Total fragments: {len(fragments)}")
    print(f"Saved to: {OUTPUT_PATH}")

    # Summary by virus
    from collections import Counter
    virus_counts = Counter(frag["virus"] for frag in fragments)
    print(f"\nPer-virus breakdown:")
    for virus in sorted(virus_counts.keys()):
        print(f"  {virus}: {virus_counts[virus]} windows")


if __name__ == "__main__":
    main()
