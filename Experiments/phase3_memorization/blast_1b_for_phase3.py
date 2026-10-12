#!/usr/bin/env python3
# BLAST the verified 1b jailbreak FASTAs for phase 3 single-strain recall +
# megablast.
#
# for each FASTA in Phase 3/generated_fastas_1b/, run NCBI Web megablast
# (top 20 hits, XML output, e-value 1e-10) and save the results to
# Phase 3/reference_seqs_1b/{name}.xml. then build phylo_input_1b.tsv so
# 02_phylogenetic_placement.py and 03_exact_match_detection.py can consume
# the 1b results the same way they consumed 7b's.
#
# uses ThreadPoolExecutor (3 workers) to stay within NCBI's rate limit.
# each megablast query takes 30-90s server-side; ~34 queries finish in
# ~10-20 min wall.
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from Bio import SeqIO
from Bio.Blast import NCBIWWW

PHASE3 = Path("Phase 3")
FASTA_DIR = PHASE3 / "generated_fastas_1b"
XML_DIR = PHASE3 / "reference_seqs_1b"
PHYLO_TSV = PHASE3 / "phylo_input_1b.tsv"


def blast_one(fasta_path: Path) -> tuple[str, str, str]:
    # BLAST one FASTA, return (name, xml_path, status).
    name = fasta_path.stem
    out_xml = XML_DIR / f"{name}.xml"
    if out_xml.exists() and out_xml.stat().st_size > 1000:
        return (name, str(out_xml), "skipped-cached")
    seq_record = next(SeqIO.parse(str(fasta_path), "fasta"))
    seq_str = str(seq_record.seq)
    try:
        rh = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=seq_str,
            megablast=True,
            hitlist_size=20,
            expect=1e-10,
        )
        xml_text = rh.read()
        rh.close()
        with open(out_xml, "w") as f:
            f.write(xml_text)
        return (name, str(out_xml), "ok")
    except Exception as e:
        return (name, str(out_xml), f"error: {e}")


def main():
    XML_DIR.mkdir(parents=True, exist_ok=True)
    fastas = sorted(FASTA_DIR.glob("*.fasta"))
    print(f"Found {len(fastas)} FASTAs in {FASTA_DIR}")
    print(f"BLASTing against NCBI nt (megablast, top-20, e-value 1e-10)")

    rows: list[tuple[str, str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(blast_one, fp): fp for fp in fastas}
        for fut in as_completed(futures):
            name, xml_path, status = fut.result()
            rows.append((name, xml_path, status))
            elapsed = time.time() - t0
            done = len(rows)
            rate = done / elapsed
            eta = (len(fastas) - done) / rate if rate > 0 else 0
            print(f"  [{done:2d}/{len(fastas)}] {name:30s} {status} | rate={rate:.2f}/s | ETA {eta/60:.1f}m")

    rows.sort(key=lambda r: r[0])
    with open(PHYLO_TSV, "w") as f:
        f.write("generated_id\tblast_xml\n")
        for name, xml_path, status in rows:
            if status in ("ok", "skipped-cached"):
                f.write(f"{name}\t{xml_path}\n")

    n_ok = sum(1 for _, _, s in rows if s in ("ok", "skipped-cached"))
    print()
    print(f"=== Summary ===")
    print(f"  total FASTAs:    {len(fastas)}")
    print(f"  successful XMLs: {n_ok}")
    print(f"  saved -> {PHYLO_TSV}")
    print(f"  elapsed:         {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
