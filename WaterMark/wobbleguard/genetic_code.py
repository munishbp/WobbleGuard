"""
Standard genetic code lookup tables for the wobble-aware detector.

The standard code has 64 codons over {A, C, G, T}, of which 61 encode amino
acids and 3 (TAA, TAG, TGA) are stop codons. See Crick, Barnett, Brenner, &
Watts-Tobin (1961) for the triplet-code framework and Crick (1966) for the
wobble hypothesis that motivates third-position synonymy.

Tables are built once at module import; no runtime cost per detection call.
"""

from __future__ import annotations

NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")


def _build_codon_table() -> dict[str, str]:
    """Standard code (NCBI translation table 1), DNA alphabet."""
    # Compact canonical encoding: amino acid string indexed by base1*16 + base2*4 + base3.
    # Order: T, C, A, G for each position (standard IUPAC table layout).
    order = ("T", "C", "A", "G")
    aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    table: dict[str, str] = {}
    idx = 0
    for b1 in order:
        for b2 in order:
            for b3 in order:
                table[b1 + b2 + b3] = aas[idx]
                idx += 1
    return table


CODON_TABLE: dict[str, str] = _build_codon_table()
"""64-entry codon → amino acid map. Stop codons map to '*'."""


def _build_synonym_classes() -> dict[str, frozenset[str]]:
    by_aa: dict[str, set[str]] = {}
    for codon, aa in CODON_TABLE.items():
        by_aa.setdefault(aa, set()).add(codon)
    return {codon: frozenset(by_aa[aa]) for codon, aa in CODON_TABLE.items()}


SYNONYM_CLASS: dict[str, frozenset[str]] = _build_synonym_classes()
"""Codon → set of codons encoding the same amino acid (including itself)."""


def _build_wobble_alternates() -> dict[str, frozenset[str]]:
    """
    For each codon, the subset of synonyms reachable by 3rd-position
    substitution only (the realistic adversarial edit).

    Stop codons are never considered alternates in a coding region — no
    adversary would introduce premature termination when trying to preserve
    the encoded protein.
    """
    alternates: dict[str, frozenset[str]] = {}
    for codon in CODON_TABLE:
        if CODON_TABLE[codon] == "*":
            alternates[codon] = frozenset({codon})
            continue
        prefix = codon[:2]
        candidates = {
            prefix + b
            for b in NUCLEOTIDES
            if CODON_TABLE[prefix + b] == CODON_TABLE[codon]
            and CODON_TABLE[prefix + b] != "*"
        }
        alternates[codon] = frozenset(candidates)
    return alternates


WOBBLE_ALTERNATES: dict[str, frozenset[str]] = _build_wobble_alternates()
"""Codon → set of synonymous codons reachable by 3rd-position substitution."""


def wobble_alternates_at_position(seq: str, pos: int, frame: int) -> frozenset[str]:
    """
    Feasible wobble alternates for the nucleotide at sequence position `pos`,
    given the reading frame (0, 1, or 2). Returns the set of nucleotides that
    could replace seq[pos] while preserving the encoded amino acid.

    If pos is not a wobble (3rd) position in the frame, returns {seq[pos]}
    (no alternative is available through a same-codon edit).
    """
    codon_offset = (pos - frame) % 3
    if codon_offset != 2:
        return frozenset({seq[pos]})
    codon_start = pos - 2
    if codon_start < frame or codon_start + 3 > len(seq):
        return frozenset({seq[pos]})
    codon = seq[codon_start : codon_start + 3]
    if codon not in WOBBLE_ALTERNATES:
        return frozenset({seq[pos]})
    return frozenset(alt[2] for alt in WOBBLE_ALTERNATES[codon])


def _longest_orf_length(seq: str, frame: int) -> int:
    """Number of in-frame codons before the first stop from `frame`."""
    count = 0
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if CODON_TABLE.get(codon, "*") == "*":
            break
        count += 1
    return count


def detect_reading_frame(seq: str, min_orf_codons: int = 50) -> int | None:
    """
    Return the frame (0/1/2) with the longest in-frame ORF, or None if no
    frame exceeds `min_orf_codons` codons before a stop. Conservative: a
    return of None means "treat as non-coding, fall back to vanilla detector."

    The default threshold of 50 codons (= 150 nt) is below the typical 640 nt
    sequence but comfortably above the ~5–10-codon stop-free runs expected
    from random DNA (geometric distribution with p_stop = 3/64).
    """
    seq = seq.upper()
    lengths = [_longest_orf_length(seq, f) for f in (0, 1, 2)]
    best_frame = max(range(3), key=lambda f: lengths[f])
    if lengths[best_frame] < min_orf_codons:
        return None
    return best_frame
