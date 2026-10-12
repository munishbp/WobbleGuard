#!/usr/bin/env python3
"""Tier 2 negative control on unwatermarked Evo2-1B output, at the full panel.

The published Tier 2 1B arm used the 102 generations of one trial. The panel
is now five trials and 532 generations, so the same false-positive test runs
over 5.2 times the sequence. Nothing about the test changes: these are
unwatermarked generations and the detector should fire on none of them.

Uses the same config as run_watermark_evaluation.py: context_width 4,
gamma 0.5, delta 1.5, alpha 0.01, and the same key.

# Make the wobbleguard package importable when run from this directory.
import sys as _sys
from pathlib import Path as _Path
_WG_ROOT = _Path(__file__).resolve().parent
for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
    if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wobbleguard.config import WatermarkConfig
from wobbleguard.evaluate import evaluate_evo2_unwatermarked
from run_tier_evaluation import SECRET_KEY

config = WatermarkConfig(key=SECRET_KEY, context_width=4, gamma=0.5, delta=1.5)

for label, path in (("1B published (n=1)", "1B_all_results.json"),
                    ("1B panel (n=5)",     "1B_all_results_n5.json"),
                    ("7B published",       "7B_all_results.json")):
    r = evaluate_evo2_unwatermarked(Path(path), config, alpha=0.01)
    # Wilson 95% interval on the false-positive rate.
    k, n, z = r["false_positives"], r["total_sequences"], 1.96
    if n:
        c = (k + z*z/2) / (n + z*z)
        h = z/(n + z*z) * (k*(n-k)/n + z*z/4) ** 0.5
        lo, hi = max(0.0, c-h), min(1.0, c+h)
    else:
        lo = hi = 0.0
    print(f"{label:22s} n={n:4d}  FP={k:3d}  FPR={r['fpr']*100:.2f}%  "
          f"Wilson95=[{lo*100:.2f}, {hi*100:.2f}]  "
          f"mean z={r['mean_z']:+.3f}  max z={r['max_z']:+.3f}")
