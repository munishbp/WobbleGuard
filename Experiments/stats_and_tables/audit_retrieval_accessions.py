#!/usr/bin/env python3
# audit the homolog accessions the retrieval agent supplied, per trial.
#
# the live gpt-4o agent is part of the experimental design, so pinning the
# homologs would measure an easier attack. instead we make the agent's
# behaviour a measured quantity: per trial, how many retrieved accessions
# resolve in genbank, how many name a non-viral organism, and how many are
# fabricated.
#
# it matters because a fabricated or drifted accession hands the model a
# prompt unrelated to the target. the model then makes normal-looking dna
# that lands near the wrong organism: healthy sequence stats, zero viral top
# hits. that's the measured signature of the seeds 43-46 collapse.
#
# classification per accession:
#   fabricated  - ncbi returns no record for the identifier at all
#   non-viral   - resolves, but the organism isn't in the Viruses division
#   viral       - resolves and is viral
#
# usage:  python3 audit_accessions.py [workspace_dir]
import collections
import glob
import json
import os
import sys
import time

from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "you@example.com")
if os.environ.get("ENTREZ_API_KEY"):
    Entrez.api_key = os.environ["ENTREZ_API_KEY"]

WS = sys.argv[1] if len(sys.argv) > 1 else "eval_workspace_local"


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def resolve(accessions):
    # {accession: (taxid, title)} for the ones ncbi knows
    found = {}
    for batch in chunks(accessions, 50):
        for attempt in range(3):
            try:
                h = Entrez.esummary(db="nuccore", id=",".join(batch))
                for rec in Entrez.read(h):
                    acc = rec.get("AccessionVersion") or rec.get("Caption")
                    # biopython returns TaxId as an IntegerElement. str() on it
                    # gives "IntegerElement(10245, attributes={})", which is
                    # not a uid, so the taxonomy lookup silently fails and
                    # every accession falls through to "non-viral". int() first.
                    tid = rec.get("TaxId")
                    tid = str(int(tid)) if tid is not None else ""
                    found[acc] = (tid, rec.get("Title", ""))
                    # also key the bare accession, the agent may drop the version
                    if acc and "." in acc:
                        found.setdefault(acc.split(".")[0], found[acc])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] batch failed: {type(e).__name__}: {e}")
                time.sleep(2 * (attempt + 1))
        time.sleep(0.4)
    return found


def divisions(taxids):
    # {taxid: division}, e.g. 'Viruses', 'Primates', 'Bacteria'.
    out = {}
    ids = [t for t in taxids if t]
    for batch in chunks(ids, 50):
        for attempt in range(3):
            try:
                h = Entrez.esummary(db="taxonomy", id=",".join(batch))
                for rec in Entrez.read(h):
                    out[str(int(rec.get("Id")))] = rec.get("Division", "")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [warn] taxonomy batch failed: {e}")
                time.sleep(2 * (attempt + 1))
        time.sleep(0.4)
    return out


def main():
    # (seed, virus, accession) for every generation
    rows = []
    for p in sorted(glob.glob(f"{WS}/modal_1b_s*/*/result.json")):
        seed, virus = p.split("/")[-3], p.split("/")[-2]
        rec = json.load(open(p))
        for g in (rec.get("generations") or []):
            a = (g.get("homolog_accession") or "").strip()
            if g.get("generated_dna"):
                rows.append((seed, virus, a))

    distinct = sorted({a for _, _, a in rows if a})
    print(f"{len(rows)} generations, {len(distinct)} distinct accessions\n")

    found = resolve(distinct)
    div = divisions({t for t, _ in found.values()})

    def classify(acc):
        if not acc:
            return "absent"
        hit = found.get(acc) or found.get(acc.split(".")[0])
        if not hit:
            return "fabricated"
        # ncbi returns the division lowercased, "viruses", not "Viruses".
        # a case-sensitive compare marked every accession non-viral.
        return "viral" if div.get(hit[0], "").lower() == "viruses" else "non-viral"

    per_seed = collections.defaultdict(collections.Counter)
    bad = []
    for seed, virus, acc in rows:
        c = classify(acc)
        per_seed[seed][c] += 1
        if c in ("fabricated", "non-viral", "absent"):
            hit = found.get(acc) or found.get(acc.split(".")[0]) or ("", "")
            bad.append((seed, virus, acc, c, hit[1][:58]))

    print(f"{'trial':16s} {'gens':>5s} {'viral':>6s} {'non-viral':>10s} "
          f"{'fabricated':>11s}  {'bad %':>6s}")
    print("-" * 62)
    for seed in sorted(per_seed):
        c = per_seed[seed]
        tot = sum(c.values())
        badn = c["fabricated"] + c["non-viral"] + c["absent"]
        print(f"{seed:16s} {tot:5d} {c['viral']:6d} {c['non-viral']:10d} "
              f"{c['fabricated']:11d}  {100.0*badn/tot:5.1f}%")
    tot = sum(sum(c.values()) for c in per_seed.values())
    allbad = sum(c["fabricated"] + c["non-viral"] + c["absent"]
                 for c in per_seed.values())
    print("-" * 62)
    print(f"{'ALL':16s} {tot:5d} "
          f"{sum(c['viral'] for c in per_seed.values()):6d} "
          f"{sum(c['non-viral'] for c in per_seed.values()):10d} "
          f"{sum(c['fabricated'] for c in per_seed.values()):11d}  "
          f"{100.0*allbad/tot:5.1f}%")

    if bad:
        print(f"\ndistinct bad accessions ({len({b[2] for b in bad})}):")
        seen = set()
        for seed, virus, acc, c, title in bad:
            if acc in seen:
                continue
            seen.add(acc)
            print(f"  {c:11s} {acc:16s} {virus:14s} {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
