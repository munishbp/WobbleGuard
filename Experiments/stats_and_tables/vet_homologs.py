#!/usr/bin/env python3
# check that every pinned homolog accession is really a homolog of its target.
#
# two things can go wrong, and only one of them is visible without asking
# genbank:
#
#   fabricated  the accession doesn't exist. AB123456 was caught by its shape.
#   wrong       the accession exists but denotes something unrelated. this is
#               the dangerous one: it fetches, it generates, and it produces
#               normal dna that lands near the wrong organism. that's the
#               signature the failed 1B seeds showed.
#
# for each accession we fetch the genbank title and ask whether it names the
# target virus, using the same accept tokens the scorer uses. anything that
# doesn't match gets reported for a human to look at, not dropped
# automatically: a real homolog can be named for a related species.
import json
import os
import re
import sys
import time

from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "you@example.com")

ROOT = os.path.dirname(os.path.abspath(__file__))
# grab the accept tokens straight out of fp_filter.py so the two can't drift
src = open(os.path.join(ROOT, "fp_filter.py")).read()
TOK = eval(re.search(r"VIRUS_ACCEPT_TOKENS = (\{.*?\n\})", src, re.S).group(1))


def title_of(acc):
    # genbank title for one accession, or None if it doesn't resolve
    for attempt in range(3):
        try:
            h = Entrez.esummary(db="nucleotide", id=acc, retmode="xml")
            rec = Entrez.read(h)
            h.close()
            if rec:
                return rec[0].get("Title", "")
            return None
        except Exception as e:
            if attempt == 2:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(3 * (attempt + 1))
    return None


def main():
    manifest = json.load(open(os.path.join(ROOT, "pinned_homologs.json")))
    out = {}
    bad = []
    n = sum(len(v) for v in manifest.values())
    i = 0
    for virus, accs in manifest.items():
        toks = [t.lower() for t in TOK.get(virus, [])]
        for acc in accs:
            i += 1
            title = title_of(acc)
            # ok = resolves, no error, and the title names the target virus
            ok = bool(title) and not str(title).startswith("__ERROR__") \
                and any(t in title.lower() for t in toks)
            out.setdefault(virus, []).append(
                {"accession": acc, "title": title, "names_target": ok})
            flag = "ok  " if ok else "CHECK"
            print(f"[{i:3d}/{n}] {flag} {virus:16s} {acc:14s} "
                  f"{(title or 'DOES NOT RESOLVE')[:64]}", flush=True)
            if not ok:
                bad.append((virus, acc, title))
            time.sleep(0.4)          # stay inside ncbi's 3 requests per second

    json.dump(out, open(os.path.join(ROOT, "homolog_vetting.json"), "w"), indent=1)
    print(f"\n{len(bad)} of {n} accessions do not name their target:")
    for virus, acc, title in bad:
        print(f"  {virus:16s} {acc:14s} {(title or 'DOES NOT RESOLVE')[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
