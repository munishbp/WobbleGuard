"""Paired sign-flip permutation test: Best-of-N against GeneBreaker,
at Evo2-1B and Evo2-7B, across the 6 JailbreakDNABench Baltimore
categories.

Null: the per-category mean difference is zero, so the two methods are
equivalent on average across categories. Alternative: the methods
differ (two-sided).

The unit of analysis is the category, not the virus and not the trial.
There are 6 categories, so 2^6 = 64 sign assignments exist and the test
enumerates all of them. The smallest p-value the design can return is
therefore 1/64 = 0.016.

Numbers come from Table 1 of the main paper. Our columns are means over
5 trials. GeneBreaker's columns are their published per-category rates.
The script asserts the result against the p-values the paper prints, so
an edit to the table below fails here instead of contradicting the
paper in silence.
"""
from __future__ import annotations

import itertools

# (category, BoN 1B, BoN 7B, GeneBreaker 1B, GeneBreaker 7B)
CATEGORIES = [
    ("+ssRNA",    14.0, 32.0, 13.3, 28.8),
    ("-ssRNA",    10.0, 55.0,  0.0, 24.4),
    ("Enteric",   30.0, 80.0, 20.0, 50.0),
    ("Large DNA", 32.0, 60.0, 20.0, 48.0),
    ("Small DNA", 20.0, 40.0, 20.0, 46.7),
    ("dsRNA",     60.0, 80.0,  0.0, 20.0),
]

# Scale, column index of our rate, column index of theirs, and the count
# of extreme sign assignments the paper reports. 6/64 prints as 0.094 and
# 4/64 prints as 0.062.
SCALES = [
    ("Evo2-1B", 1, 3, 4),
    ("Evo2-7B", 2, 4, 6),
]


def signflip(diffs: list[float]) -> tuple[float, int, float]:
    """Return the observed mean difference, the count of sign
    assignments at least that extreme, and the two-sided p-value."""
    n = len(diffs)
    obs = sum(diffs) / n
    extreme = 0
    for signs in itertools.product([-1, 1], repeat=n):
        flipped = sum(s * d for s, d in zip(signs, diffs)) / n
        if abs(flipped) >= abs(obs) - 1e-12:
            extreme += 1
    return obs, extreme, extreme / 2 ** n


def main() -> None:
    for scale, ours, theirs, expected in SCALES:
        print(f"\n{scale}: Best-of-N minus GeneBreaker, by category")
        diffs = []
        for row in CATEGORIES:
            d = row[ours] - row[theirs]
            diffs.append(d)
            print(f"  {row[0]:<12}: {row[ours]:6.1f} - {row[theirs]:6.1f} "
                  f"= {d:+7.2f}")

        obs, extreme, p = signflip(diffs)
        print(f"  observed mean difference: {obs:+.3f} pp")
        print(f"  assignments with |mean| >= |observed|: {extreme}/64")
        print(f"  two-sided p-value: {p:.4f}")

        if extreme != expected:
            raise SystemExit(
                f"{scale}: the test returns {extreme}/64 (p = {p:.4f}) but "
                f"the paper prints {expected}/64. Reconcile CATEGORIES "
                f"against Table 1, then update the expected count here."
            )

        verdict = "reject" if p <= 0.05 else "cannot reject"
        print(f"  result: {verdict} the null at alpha = 0.05")

    print("\nBoth scales agree with the printed p-values.")
    print("Neither reaches alpha = 0.05, which supports the reading")
    print("'matches' rather than 'beats'.")


if __name__ == "__main__":
    main()
