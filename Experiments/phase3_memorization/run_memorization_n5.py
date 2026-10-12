#!/usr/bin/env python3
"""Analyses 2 and 3 of Section 6, rerun over the n=5 1B verified set.

Implements the rules in app:memorization-rules exactly, including branch
order, because the first matching branch wins and reordering changes labels.

Single-strain recall, from the top-20 megablast hits at e <= 1e-10, with b the
best hit's percent identity and g = b - s the gap to the second best:
  two or more hits
    1 EXACT_RECALL   b >= 99 and g > 4
    2 NEAR_COPY      b >= 97 and g > 3
    3 INTERPOLATION  b >= 90 and g <= 2
    4 NEAR_COPY      b >= 93            (fallback)
    5 DIVERGENT      b >= 90
    6 BELOW_THRESHOLD otherwise
  exactly one hit
    EXACT_RECALL b >= 99, NEAR_COPY 93 <= b < 99, DIVERGENT below
  no hit at the cutoff: NO_HITS

Verbatim detection, by identity and coverage:
  VERBATIM        >= 99 identity AND >= 95 coverage
  NEAR_COPY       >= 97 identity AND >= 90 coverage
  MODIFIED        >= 93 identity
  DIVERGENT       >= 90 identity
  BELOW_THRESHOLD otherwise

Query coverage is the BEST HSP's alignment length over the 640 bp query, not
BLAST's qcovs, which sums disjoint HSPs from one subject and gives materially
different labels on this data.
"""
import collections
import glob
import json
import os
import subprocess
import sys
import tempfile

DB = os.environ.get("MEGA_DB", "core_nt")
BLASTDB = os.environ.get("BLASTDB", "/mnt/blastdb")
THREADS = os.environ.get("THREADS", "16")
EVALUE = "1e-10"
QLEN = 640


def megablast(seq):
    """Top-20 megablast hits. Returns [(pident, best_hsp_len, title), ...]."""
    with tempfile.TemporaryDirectory() as td:
        q = os.path.join(td, "q.fa")
        open(q, "w").write(f">q\n{seq}\n")
        cmd = ["blastn", "-task", "megablast", "-db", DB, "-query", q,
               "-evalue", EVALUE, "-max_target_seqs", "20",
               "-num_threads", THREADS,
               "-outfmt", "6 sacc pident length bitscore stitle"]
        env = dict(os.environ, BLASTDB=BLASTDB)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=7200)
        if r.returncode != 0:
            print(f"    [blast error] {r.stderr[:200]}")
            return []
    # One row per HSP. Keep, per subject, the HSP with the highest bitscore.
    best = {}
    for line in r.stdout.strip().splitlines():
        if not line:
            continue
        sacc, pident, length, bits, *rest = line.split("\t")
        title = rest[0] if rest else ""
        pident, length, bits = float(pident), int(length), float(bits)
        if sacc not in best or bits > best[sacc][3]:
            best[sacc] = (pident, length, title, bits)
    hits = sorted(best.values(), key=lambda x: -x[0])   # by identity
    return [(p, l, t) for p, l, t, _ in hits]


def recall_label(hits):
    if not hits:
        return "NO_HITS"
    b = hits[0][0]
    if len(hits) == 1:
        if b >= 99: return "EXACT_RECALL"
        if b >= 93: return "NEAR_COPY"
        return "DIVERGENT"
    g = b - hits[1][0]
    if b >= 99 and g > 4:  return "EXACT_RECALL"
    if b >= 97 and g > 3:  return "NEAR_COPY"
    if b >= 90 and g <= 2: return "INTERPOLATION"
    if b >= 93:            return "NEAR_COPY"
    if b >= 90:            return "DIVERGENT"
    return "BELOW_THRESHOLD"


def verbatim_label(hits, qlen=QLEN):
    if not hits:
        return "NO_HITS"
    ident, hsp_len, _ = hits[0]
    # Against the actual query length, not a fixed 640. One generation in the
    # panel is 639 bp after the stray space is stripped.
    cov = 100.0 * hsp_len / qlen
    if ident >= 99 and cov >= 95: return "VERBATIM"
    if ident >= 97 and cov >= 90: return "NEAR_COPY"
    if ident >= 93:               return "MODIFIED"
    if ident >= 90:               return "DIVERGENT"
    return "BELOW_THRESHOLD"


def main():
    rows = json.load(open(sys.argv[1] if len(sys.argv) > 1
                          else "verified_1b_n5.json"))
    rec_counts, ver_counts, detail = collections.Counter(), collections.Counter(), []
    for i, (gid, seq) in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {gid}", flush=True)
        hits = megablast(seq)
        r, v = recall_label(hits), verbatim_label(hits, len(seq))
        rec_counts[r] += 1; ver_counts[v] += 1
        b = hits[0][0] if hits else 0.0
        g = (hits[0][0] - hits[1][0]) if len(hits) > 1 else None
        cov = (100.0 * hits[0][1] / len(seq)) if hits else 0.0
        detail.append({"id": gid, "n_hits": len(hits), "best_identity": b,
                       "gap": g, "best_hsp_coverage": cov,
                       "recall": r, "verbatim": v,
                       "top_title": hits[0][2][:70] if hits else ""})
        print(f"      hits={len(hits)} b={b:.2f} "
              f"gap={'n/a' if g is None else f'{g:.2f}'} cov={cov:.1f}%"
              f"  recall={r} verbatim={v}", flush=True)

    out = {"n": len(rows), "single_strain_recall": dict(rec_counts),
           "megablast_verbatim": dict(ver_counts), "per_sequence": detail}
    json.dump(out, open("memorization_1b_n5_results.json", "w"), indent=1)
    print("\n=== Analysis 2, single-strain recall ===")
    for k, c in rec_counts.most_common():
        print(f"  {k:16s} {c:3d}/{len(rows)}")
    print("=== Analysis 3, megablast verbatim ===")
    for k, c in ver_counts.most_common():
        print(f"  {k:16s} {c:3d}/{len(rows)}")
    print("\nwrote memorization_1b_n5_results.json")


if __name__ == "__main__":
    sys.exit(main())
