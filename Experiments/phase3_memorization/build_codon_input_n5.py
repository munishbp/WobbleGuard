#!/usr/bin/env python3
# build the codon-mismatch input for the n=5 1b verified set.
#
# one row per verified generation: its sequence, plus the closest BLAST hit
# fetched from NCBI as a FASTA. verification uses the same rule
# score_trials.py applies -- the top hit at >= 90 percent identity carrying
# an accept token.
#
# references longer than 50 kbp get skipped, matching the earlier analysis:
# global pairwise alignment against a metagenome assembly is infeasible.
import glob
import json
import os
import re
import sys
import time

from Bio import Entrez, SeqIO

Entrez.email = os.environ.get("ENTREZ_EMAIL", "you@example.com")
ROOT = os.environ.get("EVO2_ROOT", os.getcwd())
REFDIR = f"{ROOT}/Phase 3/reference_seqs_1b_n5"
GENDIR = f"{ROOT}/Phase 3/generated_seqs_1b_n5"
MAX_REF_BP = 50_000

TOK = eval(re.search(r"VIRUS_ACCEPT_TOKENS = (\{.*?\n\})",
                     open(f"{ROOT}/fp_filter.py").read(), re.S).group(1))


def passes(hits, v):
    if not hits:
        return False
    h = hits[0]
    if (h.get("percent_identity") or 0) < 90.0:
        return False
    t = (h.get("title") or "").lower()
    return any(x.lower() in t for x in TOK.get(v, []))


def accession(title):
    # pull the accession out of a BLAST title like gi|123|gb|PV027239.1|.
    m = re.search(r"\|(?:gb|emb|dbj|ref|tpg)\|([A-Z_0-9.]+)\|", title)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z]{1,2}_?\d{5,8}(?:\.\d+)?)\b", title)
    return m.group(1) if m else None


def collect():
    out = []
    for rec in json.load(open(f"{ROOT}/1B_all_results.json")):
        v = rec.get("virus")
        gens = rec.get("generations") or []
        for e in (rec.get("blast_results") or []):
            i = e.get("generation_index")
            if i is None or i >= len(gens):
                continue
            if passes(e.get("blastn") or [], v) or passes(e.get("blastp") or [], v):
                out.append((f"pub_{v}_gen{i}", gens[i].get("generated_dna", ""),
                            (e.get("blastn") or [{}])[0].get("title", "")))
    for p in sorted(glob.glob(f"{ROOT}/local1b/eval_workspace_local/modal_1b_s*/*/result.json")):
        seed, v = p.split("/")[-3][-3:], p.split("/")[-2]
        r = json.load(open(p))
        gens = [g for g in (r.get("generations") or []) if g.get("generated_dna")]
        for e in (r.get("blast_results") or []):
            i = e.get("generation_index")
            if i is None or i >= len(gens):
                continue
            if passes(e.get("blastn") or [], v) or passes(e.get("blastp") or [], v):
                out.append((f"{seed}_{v}_gen{i}", gens[i]["generated_dna"],
                            (e.get("blastn") or [{}])[0].get("title", "")))
    return out


def main():
    rows = collect()
    print(f"{len(rows)} verified generations")
    kept, skipped = [], []
    for gid, dna, title in rows:
        acc = accession(title)
        if not acc:
            skipped.append((gid, "no accession parsed", title[:50]))
            continue
        path = f"{REFDIR}/{gid}.fasta"
        if not os.path.exists(path):
            try:
                h = Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text")
                txt = h.read()
                open(path, "w").write(txt)
                time.sleep(0.4)
            except Exception as exc:
                skipped.append((gid, f"fetch failed: {type(exc).__name__}", acc))
                continue
        try:
            rec = next(SeqIO.parse(path, "fasta"))
        except Exception:
            skipped.append((gid, "unparseable fasta", acc)); continue
        if len(rec.seq) > MAX_REF_BP:
            skipped.append((gid, f"reference {len(rec.seq)} bp > {MAX_REF_BP}", acc))
            os.remove(path)
            continue
        # the analysis reads BOTH columns as paths, so the generation needs
        # its own FASTA. whitespace is stripped here for the same reason it
        # gets stripped before translation: BLAST ignores it, so the scored
        # sequence is the one without it.
        clean = "".join(dna.split()).upper()
        gpath = f"{GENDIR}/{gid}.fasta"
        with open(gpath, "w") as fh:
            fh.write(f">{gid}\n{clean}\n")
        kept.append((gid, gpath, path))

    with open(f"{ROOT}/Phase 3/codon_input_1b_n5.tsv", "w") as f:
        f.write("generated_id\tgenerated_fasta\treference_fasta\n")
        for gid, gpath, path in kept:
            f.write(f"{gid}\t{gpath}\t{path}\n")

    print(f"kept {len(kept)}, skipped {len(skipped)}")
    for gid, why, extra in skipped:
        print(f"  skip {gid:28s} {why}  {extra}")


if __name__ == "__main__":
    sys.exit(main())
