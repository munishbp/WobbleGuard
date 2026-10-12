# does a codon-optimized gene break the green-list z-test independence
# assumption?
#
# structural test. uses SHA-256 as the random oracle exactly the way
# watermark/utils.py does, but rebuilds the permutation with random.Random
# instead of numpy PCG64 (not available here). the question is structural,
# so the PRNG choice doesn't change the conclusion.
import hashlib, struct, random, statistics as s
from collections import Counter

BASES = "ACGT"; IDX = {c: i for i, c in enumerate(BASES)}
GAMMA, K, NGREEN = 0.5, 4, 2

CODON_TABLE = {}
_aas = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
for i, b1 in enumerate("AGCT"):
    for j, b2 in enumerate("AGCT"):
        for k3, b3 in enumerate("AGCT"):
            CODON_TABLE[b1+b2+b3] = _aas[i*16 + j*4 + k3]
SYN = {}
for c, aa in CODON_TABLE.items(): SYN.setdefault(aa, []).append(c)

# e. coli k-12's most-frequent codon per amino acid -- this is what a codon
# optimizer picks.
ECOLI_PREF = {"A":"GCG","R":"CGC","N":"AAC","D":"GAT","C":"TGC","Q":"CAG","E":"GAA",
  "G":"GGC","H":"CAT","I":"ATT","L":"CTG","K":"AAA","M":"ATG","F":"TTT","P":"CCG",
  "S":"AGC","T":"ACC","W":"TGG","Y":"TAT","V":"GTG","*":"TAA"}

def green(ctx, key):
    seed = struct.unpack("<Q", hashlib.sha256(key + bytes(ctx)).digest()[:8])[0]
    perm = list(range(4)); random.Random(seed).shuffle(perm)
    return set(perm[:NGREEN])

def zscore(seq, key):
    ids = [IDX[c] for c in seq]
    T = len(ids) - K
    if T <= 0: return 0.0, 0
    G = sum(1 for t in range(K, len(ids)) if ids[t] in green(ids[t-K:t], key))
    return (G - T*GAMMA) / ((T*GAMMA*(1-GAMMA))**0.5), T

def build(protein, mode, rng):
    if mode == "opt":  return "".join(ECOLI_PREF[a] for a in protein)
    return "".join(rng.choice(SYN[a]) for a in protein)

KEY = bytes.fromhex("00"*32)
rng = random.Random(7)
aas = [a for a in ECOLI_PREF if a != "*"]
N, L = 1500, 300
res = {}
ctxdiv = {}
for mode in ("rand", "opt"):
    zs = []; divs = []
    r2 = random.Random(11)
    for _ in range(N):
        prot = [rng.choice(aas) for _ in range(L)]
        seq = build(prot, mode, r2)
        z, T = zscore(seq, KEY); zs.append(z)
        ids = [IDX[c] for c in seq]
        ctxs = [tuple(ids[t-K:t]) for t in range(K, len(ids))]
        divs.append(len(set(ctxs)))
    res[mode] = zs; ctxdiv[mode] = divs

print(f"n={N} proteins, L={L} residues ({L*3} bp), key fixed, threshold z*=3.065\n")
print(f"{'mode':6} {'mean z':>8} {'std z':>7} {'max z':>7} {'FPR@2.326':>10} {'FPR@3.065':>10} {'uniq 4-mer ctx':>15}")
for mode in ("rand", "opt"):
    zs = res[mode]
    f1 = sum(1 for z in zs if z > 2.326)/len(zs)
    f2 = sum(1 for z in zs if z > 3.065)/len(zs)
    print(f"{mode:6} {s.mean(zs):>8.3f} {s.stdev(zs):>7.3f} {max(zs):>7.2f} {f1:>10.4f} {f2:>10.4f} {s.mean(ctxdiv[mode]):>15.1f}")
print("\nExpected under a valid N(0,1) null: mean 0, std 1, FPR@2.326=0.0100, FPR@3.065=0.0011")
