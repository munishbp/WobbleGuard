# compare watermarked vs unwatermarked BLAST identity distributions for the
# tier 4 same-prompt 1B run (job 609244). produces the numbers needed to
# update the paper's BLAST-quality-cost paragraph (§4.4.4).
import json
from pathlib import Path

HERE = Path(__file__).parent
WM_GEN = json.load(open(HERE / "watermarked.json"))
UW_GEN = json.load(open(HERE / "unwatermarked.json"))
WM_BL = json.load(open(HERE / "watermarked_blast.json"))
UW_BL = json.load(open(HERE / "unwatermarked_blast.json"))


def index_by_sample(gen):
    # sample_index -> generation record
    return {r["sample_index"]: r for r in gen["results"]}


def index_blast(bl):
    # sample_index -> blast record
    return {r["sample_index"]: r for r in bl["results"]}


def summarize(label, gen, bl):
    # join generations with their blast hits, then print the per-arm summary
    g = index_by_sample(gen)
    b = index_blast(bl)
    rows = []
    for idx in sorted(g.keys()):
        gr = g[idx]
        br = b.get(idx, {})
        rows.append(
            {
                "sample_index": idx,
                "virus": gr["virus"],
                "homolog_accession": gr["homolog_accession"],
                "has_hit": br.get("has_hit", False),
                "percent_identity": br.get("percent_identity"),
                "query_coverage": br.get("query_coverage"),
                "align_length": br.get("align_length"),
                "top_hit": (br.get("top_hit_def") or "")[:80],
            }
        )
    n = len(rows)
    n_hit = sum(1 for r in rows if r["has_hit"])
    n_ge90 = sum(
        1
        for r in rows
        if r["has_hit"] and r["percent_identity"] is not None and r["percent_identity"] >= 90
    )
    pids = [r["percent_identity"] for r in rows if r["has_hit"] and r["percent_identity"] is not None]
    covs = [r["query_coverage"] for r in rows if r["has_hit"] and r["query_coverage"] is not None]
    print(f"=== {label} ===")
    print(f"  N: {n}")
    print(f"  with any BLAST hit: {n_hit}/{n} = {100*n_hit/n:.1f}%")
    print(f"  with >=90% identity: {n_ge90}/{n} = {100*n_ge90/n:.1f}%")
    if pids:
        print(f"  hit %identity: mean={sum(pids)/len(pids):.2f} median={sorted(pids)[len(pids)//2]:.2f} min={min(pids):.2f} max={max(pids):.2f}")
    if covs:
        print(f"  hit %coverage: mean={sum(covs)/len(covs):.2f} median={sorted(covs)[len(covs)//2]:.2f} min={min(covs):.2f} max={max(covs):.2f}")
    print()
    print("  per-row:")
    for r in rows:
        idstr = f"{r['percent_identity']:.1f}%" if r["percent_identity"] is not None else " no-hit"
        covstr = f"cov={r['query_coverage']:.1f}%" if r["query_coverage"] is not None else "        "
        alnstr = f"aln={r['align_length']}" if r["align_length"] is not None else ""
        print(f"    [{r['sample_index']:2d}] {r['virus']:14s} {r['homolog_accession']:14s} id={idstr:>6s} {covstr} {alnstr}")
    print()
    return rows


print("Tier 4 same-prompt control (Evo2-1B):")
print("13 prompts matched between watermarked and unwatermarked runs")
print("same homolog accessions, same input_seq_base\n")

wm_rows = summarize("WATERMARKED (delta=1.5, gamma=0.5, k=4)", WM_GEN, WM_BL)
uw_rows = summarize("UNWATERMARKED baseline", UW_GEN, UW_BL)

# pairwise watermarked-vs-unwatermarked deltas
wm_by_idx = {r["sample_index"]: r for r in wm_rows}
uw_by_idx = {r["sample_index"]: r for r in uw_rows}
print("=== PAIRWISE (per (virus, homolog), watermarked vs unwatermarked) ===")
print(f"  {'idx':>3s}  {'virus':<14s} {'homolog':<14s}  wm-id    uw-id   delta")
deltas = []
for idx in sorted(wm_by_idx.keys()):
    if idx not in uw_by_idx:
        continue
    w = wm_by_idx[idx]
    u = uw_by_idx[idx]
    wid = w["percent_identity"]
    uid = u["percent_identity"]
    if wid is None and uid is None:
        line = f"  {idx:>3d}  {w['virus']:<14s} {w['homolog_accession']:<14s}  no-hit   no-hit  ---"
    elif wid is None:
        line = f"  {idx:>3d}  {w['virus']:<14s} {w['homolog_accession']:<14s}  no-hit   {uid:5.1f}%  ---"
    elif uid is None:
        line = f"  {idx:>3d}  {w['virus']:<14s} {w['homolog_accession']:<14s}  {wid:5.1f}%   no-hit  ---"
    else:
        # both sides hit, so the identity delta is meaningful
        d = wid - uid
        deltas.append(d)
        line = f"  {idx:>3d}  {w['virus']:<14s} {w['homolog_accession']:<14s}  {wid:5.1f}%   {uid:5.1f}%  {d:+5.2f}"
    print(line)

if deltas:
    print()
    print(f"  mean delta (wm minus uw, on rows with hits in both): {sum(deltas)/len(deltas):+.2f} pp")
    print(f"  median delta: {sorted(deltas)[len(deltas)//2]:+.2f} pp")
