# pull out the trial 2 unique strict-verified generated sequences (hpv gen 5,
# adenovirus gen 2, hcv gen 2) from
# `Phase 1/evo2_7b_trial2_600776/all_results.json`, grab the matching
# reference NCBI sequences, and write FASTA + TSV rows that extend
# `Phase 3/codon_input.tsv` and `Phase 3/phylo_input.tsv`.
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    from Bio import Entrez, SeqIO
except ImportError:
    print("Need biopython: py -m pip install biopython")
    sys.exit(1)

Entrez.email = "anonymous@example.org"

ROOT = Path(os.environ.get("EVO2_ROOT", "."))
T2_RESULTS = ROOT / "Phase 1" / "evo2_7b_trial2_600776" / "all_results.json"
PHASE3 = ROOT / "Phase 3"
GEN_DIR = PHASE3 / "generated_fastas"
REF_DIR = PHASE3 / "reference_seqs"

# (virus_key in all_results, gen_idx, generated_id_for_tsv, ref_acc, ref_db)
TARGETS = [
    ("hpv", 5, "hpv_gen5_t2", "JQ791068.1", "nuccore"),
    ("adenovirus", 2, "adenovirus_gen2_t2", "PV550965.1", "nuccore"),
    ("hcv", 2, "hcv_gen2_t2", "EU370584.1", "nuccore"),
    # hcv gen 2 also has a BLASTp hit, but the codon test only uses DNA so a
    # nucleotide reference is fine. the phylo test runs the generated DNA
    # against NCBI nt, so it's DNA-only too.
]


def find_generation(data: dict, virus: str, gen_idx: int) -> dict | None:
    for vrec in data["viruses"] if "viruses" in data else [data]:
        # all_results.json is either a list of per-virus records or a dict
        # at the top
        if isinstance(vrec, dict) and vrec.get("virus") == virus:
            for gen in vrec.get("generations", []):
                if gen.get("generation_index") == gen_idx or gen.get("gen_idx") == gen_idx:
                    return gen
    return None


def write_fasta(path: Path, seq_id: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{seq_id}\n{seq}\n", encoding="utf-8")


def fetch_ncbi(acc: str, db: str = "nuccore") -> str | None:
    print(f"  fetching {acc} from NCBI {db}...")
    try:
        with Entrez.efetch(db=db, id=acc, rettype="fasta", retmode="text") as h:
            rec = SeqIO.read(h, "fasta")
        return str(rec.seq).upper()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main() -> None:
    print(f"Loading {T2_RESULTS}...")
    data = json.loads(T2_RESULTS.read_text(encoding="utf-8"))

    # first figure out what shape all_results.json is
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "viruses" in data:
        records = data["viruses"]
    elif isinstance(data, dict):
        records = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    else:
        print(f"Unknown schema for all_results.json: {type(data)}")
        sys.exit(1)

    # virus -> record lookup
    by_virus = {}
    for rec in records:
        if isinstance(rec, dict) and "virus" in rec:
            by_virus[rec["virus"]] = rec

    print(f"Found {len(by_virus)} virus records.")

    new_rows = []
    for virus, gen_idx, seq_id, ref_acc, ref_db in TARGETS:
        rec = by_virus.get(virus)
        if not rec:
            print(f"\n!! virus {virus} not found")
            continue

        # 'generations' is indexed positionally, so gen_idx == list index.
        gen_dna = None
        gens = rec.get("generations", [])
        if 0 <= gen_idx < len(gens):
            gen_dna = gens[gen_idx].get("generated_dna")

        if not gen_dna:
            print(f"\n!! {virus} gen {gen_idx} DNA not found in record")
            continue

        gen_path = GEN_DIR / f"{seq_id}.fasta"
        write_fasta(gen_path, seq_id, gen_dna)
        print(f"\n{seq_id}: {len(gen_dna)} bp -> {gen_path.name}")

        ref_seq = fetch_ncbi(ref_acc, ref_db)
        if ref_seq is None:
            print(f"  SKIP: could not fetch reference {ref_acc}")
            continue
        time.sleep(0.4)  # don't hammer NCBI

        ref_path = REF_DIR / f"{seq_id}.fasta"
        write_fasta(ref_path, ref_acc, ref_seq)
        print(f"  ref {ref_acc}: {len(ref_seq)} bp -> {ref_path.name}")

        new_rows.append({
            "generated_id": seq_id,
            "generated_fasta": str(gen_path),
            "reference_fasta": str(ref_path),
            "generated_dna": gen_dna,
        })

    out_tsv = PHASE3 / "codon_input_t2_additions.tsv"
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        f.write("generated_id\tgenerated_fasta\treference_fasta\n")
        for r in new_rows:
            f.write(f"{r['generated_id']}\t{r['generated_fasta']}\t{r['reference_fasta']}\n")
    print(f"\nWrote {out_tsv} ({len(new_rows)} rows)")


if __name__ == "__main__":
    main()
