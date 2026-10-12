# standard genetic code lookup tables for the wobble-aware detector.
# the standard code has 64 codons over {A, C, G, T}, 61 encode amino acids
# and 3 (TAA, TAG, TGA) are stops. see crick, barnett, brenner & watts-tobin
# (1961) for the triplet-code idea and crick (1966) for the wobble hypothesis
# behind third-position synonymy.
# tables are built once at import, so detection calls pay nothing extra.

from __future__ import annotations

NUCLEOTIDES: tuple[str, ...] = ("A", "C", "G", "T")


def _build_codon_table() -> dict[str, str]:
    # standard code (ncbi translation table 1), dna alphabet.
    # compact canonical encoding: the amino acid string is indexed by
    # base1*16 + base2*4 + base3, with order T, C, A, G per position
    # (standard iupac table layout).
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
# 64-entry codon -> amino acid map. stops map to '*'.


def _build_synonym_classes() -> dict[str, frozenset[str]]:
    by_aa: dict[str, set[str]] = {}
    for codon, aa in CODON_TABLE.items():
        by_aa.setdefault(aa, set()).add(codon)
    return {codon: frozenset(by_aa[aa]) for codon, aa in CODON_TABLE.items()}


SYNONYM_CLASS: dict[str, frozenset[str]] = _build_synonym_classes()
# codon -> set of codons that code the same amino acid (itself included).


def _build_wobble_alternates() -> dict[str, frozenset[str]]:
    # for each codon, the synonyms you can reach by swapping only the 3rd
    # position — the realistic adversarial edit.
    # stops never count as alternates in a coding region: nobody trying to
    # keep the protein intact would introduce an early stop.
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
# codon -> set of synonymous codons reachable by a 3rd-position swap.


def wobble_alternates_at_position(seq: str, pos: int, frame: int) -> frozenset[str]:
    # feasible wobble alternates for the nucleotide at position `pos`, given
    # the reading frame (0, 1, or 2). returns the set of nucleotides that
    # could replace seq[pos] while keeping the amino acid the same.
    # if pos isn't a wobble (3rd) position in this frame, you get
    # {seq[pos]} — no same-codon edit is available there.
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
    # number of in-frame codons before the first stop, starting from `frame`
    count = 0
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if CODON_TABLE.get(codon, "*") == "*":
            break
        count += 1
    return count


def detect_reading_frame(seq: str, min_orf_codons: int = 50) -> int | None:
    # return the frame (0/1/2) with the longest in-frame orf, or None if no
    # frame clears `min_orf_codons` codons before a stop. conservative on
    # purpose: None means "treat as non-coding, fall back to vanilla detector".
    # the default threshold of 50 codons (= 150 nt) is below the typical
    # 640 nt sequence but comfortably above the ~5-10-codon stop-free runs
    # you'd expect from random dna (geometric with p_stop = 3/64).
    seq = seq.upper()
    lengths = [_longest_orf_length(seq, f) for f in (0, 1, 2)]
    best_frame = max(range(3), key=lambda f: lengths[f])
    if lengths[best_frame] < min_orf_codons:
        return None
    return best_frame
