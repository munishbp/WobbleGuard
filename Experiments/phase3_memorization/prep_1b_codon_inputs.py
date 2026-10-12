#!/usr/bin/env python3
# prep the 1b pillar 3 codon-mismatch inputs.
#
# walks the 36 unique 1b verified jailbreak generations, writes each one as a
# FASTA, fetches its top-BLAST-hit reference from GenBank via Entrez, and
# emits a codon_input_1b.tsv parallel to the existing 7b codon_input.tsv.
#
# inputs:  verified_jailbreak_sequences_1b.csv  (rows include top_hit_accession)
# outputs: Phase 3/generated_fastas_1b/<id>.fasta
#          Phase 3/reference_seqs_1b/<id>.fasta
#          Phase 3/codon_input_1b.tsv

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO

ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "you@example.com")
SLEEP_BETWEEN_FETCH_SEC = 0.5  # entrez throttle

GEN_DIR = Path("Phase 3/generated_fastas_1b")
REF_DIR = Path("Phase 3/reference_seqs_1b")
TSV_PATH = Path("Phase 3/codon_input_1b.tsv")


def virus_id(virus: str, gen_idx: int, blast_type: str) -> str:
    # match the existing 7b convention: <virus>_gen<idx> for BLASTn,
    # <virus>_gen<idx>p for BLASTp.
    suffix = "p" if blast_type.upper() == "BLASTP" else ""
    return f"{virus}_gen{gen_idx}{suffix}"


def write_fasta(path: Path, header: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")


def fetch_reference(accession: str) -> str | None:
    # grab a nucleotide sequence from GenBank by accession. returns the first
    # record's sequence, or None if it fails.
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession,
                               rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return str(record.seq).upper()
    except Exception as e:
        print(f"  [fetch] failed for {accession}: {e}", file=sys.stderr)
        return None


def main():
    Entrez.email = ENTREZ_EMAIL
    rows = list(csv.DictReader(open("verified_jailbreak_sequences_1b.csv")))

    # each CSV row is one verified (virus, gen_idx, blast_type) entry. the
    # codon analysis treats each (gen, ref) pair on its own, matching the
    # 7b convention where _genN and _genNp can point at different references.
    tsv_rows = []
    fetched_count = 0
    skipped_count = 0
    skipped_protein = 0

    for r in rows:
        virus = r["virus"]
        gen_idx = int(r["generation_index"])
        blast_type = r["blast_type"]
        seq = r["generated_sequence"].upper().replace("N", "")
        accession = r["top_hit_accession"].strip()

        gid = virus_id(virus, gen_idx, blast_type)
        gen_path = GEN_DIR / f"{gid}.fasta"
        ref_path = REF_DIR / f"{gid}.fasta"

        if not seq or len(seq) < 100:
            print(f"  [skip] {gid}: gen sequence too short")
            skipped_count += 1
            continue

        # no protein accessions here. a BLASTp top_hit_accession is a
        # protein accession (e.g. AA44696.1) but the codon analysis needs
        # nucleotide references, so skip those rows. GenBank protein
        # accessions look like 3 letters + 5 digits (AAA12345) plus an
        # optional version (.1). nucleotide is 1-2 letters + 5-8 digits.
        if blast_type.upper() == "BLASTP":
            print(f"  [skip-p] {gid}: BLASTp accession {accession} is protein, "
                  f"need a nucleotide reference; skipping (parallel to 7B convention "
                  f"where _genNp uses a separate nt fetch)")
            skipped_protein += 1
            continue

        write_fasta(gen_path, gid, seq)

        if ref_path.exists() and ref_path.stat().st_size > 0:
            print(f"  [cached] {gid}: {accession}")
        else:
            print(f"  [fetch ] {gid}: {accession}", flush=True)
            ref_seq = fetch_reference(accession)
            time.sleep(SLEEP_BETWEEN_FETCH_SEC)
            if ref_seq is None or len(ref_seq) < 100:
                print(f"  [skip] {gid}: ref fetch failed or too short")
                skipped_count += 1
                continue
            write_fasta(ref_path, accession, ref_seq)
            fetched_count += 1

        tsv_rows.append({
            "generated_id": gid,
            "generated_fasta": str(gen_path).replace("\\", "/"),
            "reference_fasta": str(ref_path).replace("\\", "/"),
        })

    TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["generated_id", "generated_fasta", "reference_fasta"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)

    print(f"\n{'-'*60}")
    print(f"Wrote {len(tsv_rows)} rows -> {TSV_PATH}")
    print(f"  references fetched fresh: {fetched_count}")
    print(f"  protein-accession rows skipped: {skipped_protein}")
    print(f"  other skips: {skipped_count}")


if __name__ == "__main__":
    main()
