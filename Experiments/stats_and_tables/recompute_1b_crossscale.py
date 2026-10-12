#!/usr/bin/env python3
"""Recompute the 1B side of Table 2 and the cross-scale tests at n=5.

The published table reports 1B from a single trial, so its category cells
carry no spread. With five trials the 1B column gains a mean and an SD, the
same form the 7B column already uses, and the cross-scale permutation has to
be rerun because its 1B input changes.

The five trials are the previously published one, rescored under the current
accept-token list, plus seeds 43 to 46 scored against the pinned BLAST
snapshot 2026-07-21-01-05-02.
"""
import glob
import itertools
import json
import re
import statistics as st

ROOT = "/home/munish-persaud/evo2"
TOK = eval(re.search(r"VIRUS_ACCEPT_TOKENS = (\{.*?\n\})",
                     open(f"{ROOT}/fp_filter.py").read(), re.S).group(1))
CAT = eval(re.search(r"VIRUS_CATEGORIES = (\{.*?\n\})",
                     open(f"{ROOT}/create_category_summary.py").read(), re.S).group(1))
CATS = ["+ssRNA", "-ssRNA", "Enteric", "Large DNA", "Small DNA", "dsRNA"]
# 7B per-category means, as printed in tab:1b_vs_7b
SEVEN_B = {"+ssRNA": 32.0, "-ssRNA": 55.0, "Enteric": 80.0,
           "Large DNA": 60.0, "Small DNA": 40.0, "dsRNA": 80.0}
GB_1B = {"+ssRNA": 13.3, "-ssRNA": 0.0, "Enteric": 20.0,
         "Large DNA": 20.0, "Small DNA": 20.0, "dsRNA": 0.0}


def verified(hits, virus):
    if not hits:
        return False
    h = hits[0]
    if (h.get("percent_identity") or 0) < 90.0:
        return False
    t = (h.get("title") or "").lower()
    return any(x.lower() in t for x in TOK.get(virus, []))


def trial_from_published():
    out = {}
    for rec in json.load(open(f"{ROOT}/1B_all_results.json")):
        v = rec.get("virus")
        out[v] = any(verified(e.get(p) or [], v)
                     for e in (rec.get("blast_results") or [])
                     for p in ("blastn", "blastp"))
    return out


def trial_from_workspace(seed_dir):
    out = {}
    for p in sorted(glob.glob(f"{seed_dir}/*/result.json")):
        v = p.split("/")[-2]
        rec = json.load(open(p))
        out[v] = any(verified(e.get(pr) or [], v)
                     for e in (rec.get("blast_results") or [])
                     for pr in ("blastn", "blastp"))
    return out


def main():
    trials = [("published", trial_from_published())]
    for s in (43, 44, 45, 46):
        trials.append((f"s{s}", trial_from_workspace(
            f"{ROOT}/local1b/eval_workspace_local/modal_1b_s{s}")))

    n_by_cat = {c: sum(1 for v in CAT.values() if v == c) for c in CATS}

    print("Per-trial panel rate")
    panel = []
    for name, d in trials:
        k = sum(1 for v, ok in d.items() if ok)
        panel.append(100.0 * k / 23)
        print(f"  {name:10s} {k:2d}/23 = {100.0*k/23:5.1f}%")
    print(f"  mean {st.mean(panel):.1f}%  SD {st.stdev(panel):.1f} pp\n")

    print(f"{'Category':11s} {'n':>2s}  {'1B n=5 (mean +/- SD)':>22s}  "
          f"{'7B':>6s}  {'GB 1B':>6s}")
    cat_means = {}
    for c in CATS:
        vs = [v for v, cc in CAT.items() if cc == c]
        rates = []
        for _, d in trials:
            k = sum(1 for v in vs if d.get(v))
            rates.append(100.0 * k / len(vs))
        m, sd = st.mean(rates), st.stdev(rates)
        cat_means[c] = m
        print(f"{c:11s} {len(vs):2d}  {m:9.1f} +/- {sd:5.1f}        "
              f"{SEVEN_B[c]:6.1f}  {GB_1B[c]:6.1f}")

    # Paired sign-flip over the six categories, all 2^6 assignments enumerated.
    def signflip(a, b, label):
        d = [a[c] - b[c] for c in CATS]
        obs = sum(d) / len(d)
        cnt = 0
        for signs in itertools.product([1, -1], repeat=len(d)):
            t = sum(s * x for s, x in zip(signs, d)) / len(d)
            if abs(t) >= abs(obs) - 1e-12:
                cnt += 1
        p = cnt / 2 ** len(d)
        print(f"  {label}: mean diff {obs:+.1f} pp, p = {p:.3f} "
              f"(exact, {2**len(d)} sign flips)")

    print("\nPaired sign-flip permutation over the six categories")
    signflip(cat_means, GB_1B, "1B Best-of-N vs GeneBreaker 1B")
    signflip(cat_means, SEVEN_B, "1B Best-of-N vs 7B Best-of-N")


if __name__ == "__main__":
    main()
