#!/usr/bin/env python3
# tier 2 on unwatermarked evo2-1B output at BOTH thresholds, to match tab:tier2.
#
# the published 1B row only reports z_v at the analytical 2.326 on 102
# generations. the 7B rows report z_v and max(z_v, z_w) at both the analytical
# 2.326 and the deployed empirical 3.065. this computes the 1B arm the same
# way over the full 532-generation panel so the rows are actually comparable.
#
# the bootstrap snippet below was never live code (it just sat inside the old
# docstring), so it stays commented out:
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkDetector
from wobbleguard.wobble_detector import WobbleAwareDetector as WobbleDetector
from run_tier_evaluation import SECRET_KEY

CFG = WatermarkConfig(key=SECRET_KEY, context_width=4, gamma=0.5, delta=1.5)
ANALYTICAL, EMPIRICAL = 2.326, 3.065


def wilson(k, n, z=1.96):
    # wilson score interval, scaled to percent
    if not n:
        return 0.0, 0.0
    c = (k + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * (k * (n - k) / n + z * z / 4) ** 0.5
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def run(path, label):
    # score one results json at both thresholds, for z_v and max(z_v, z_w)
    data = json.load(open(path))
    dv = WatermarkDetector(CFG, alpha=0.01)
    try:
        dw = WobbleDetector(CFG)
    except Exception:
        # wobble detector didn't build, so just fall back to z_v alone
        dw = None
    zv, zmax = [], []
    for entry in data:
        for gen in entry.get("generations", []):
            s = gen.get("generated_dna", "")
            if len(s) <= CFG.context_width + 1:
                continue
            a = dv.detect(s).z_score
            zv.append(a)
            b = a
            if dw is not None:
                try:
                    b = max(a, dw.detect(s).z_score)
                except Exception:
                    pass
            zmax.append(b)
    n = len(zv)
    print(f"\n{label}  (N = {n})")
    for name, arr in (("z_v", zv), ("max(z_v,z_w)", zmax)):
        for tname, t in (("analytical 2.326", ANALYTICAL), ("empirical 3.065", EMPIRICAL)):
            k = sum(1 for x in arr if x >= t)
            lo, hi = wilson(k, n)
            print(f"  {name:13s} {tname:17s} {100*k/n:5.2f}% ({k}/{n})  "
                  f"Wilson95=[{lo:.2f}, {hi:.2f}]")


run("1B_all_results.json", "Evo2-1B published")
run("1B_all_results_n5.json", "Evo2-1B panel n=5")
