#!/usr/bin/env python3
# phase 4c (tier 4): watermarked evo2 generation on H100 (Lambda).
#
# generates DNA from evo2 with the WatermarkLogitsProcessor active during
# sampling, then runs the WatermarkDetector on the resulting strings.
# writes watermarked_{model}_results.json plus a per-sequence detection csv.
#
# the tricky part is the vocab mismatch. the watermark processor wants a
# 4-element logits vector indexed A=0, C=1, G=2, T=3. evo2 uses byte-level
# tokenization where DNA chars live at their ASCII codes (A=65, C=67, G=71,
# T=84). EvoWmAdapter bridges the two:
#
#   1. each step, grab evo2's full-vocab logits [vocab_size].
#   2. mask every non-DNA token to -inf so the model can only sample {A,C,G,T}.
#   3. pull out the 4 DNA logits in canonical (A,C,G,T) order.
#   4. run WatermarkLogitsProcessor on that 4-vector.
#   5. write the modified logits back into the full-vocab tensor.
#   6. sample from the full-vocab distribution.
#
# the processor's internal token_ids list stays in the 4-token indexing
# (A=0..T=3), separate from evo2's byte-indexed input_ids, so the green-list
# hash matches what the detector will compute on the final DNA string.
#
# usage (on Lambda H100):
#     python3 run_watermark_evo2.py --model 7B --num-sequences 100 \
#         --prompt ATG --seq-length 640 --output watermarked_7b_results.json
#
# for the k-context ablation (#12), run it 4 times with --k {1,2,4,8}.
#
# the bootstrap snippet below was never live code (it just sat inside the old
# docstring), so it stays commented out:
# import sys as _sys
# from pathlib import Path as _Path
# _WG_ROOT = _Path(__file__).resolve().parent
# for _cand in (_WG_ROOT, _WG_ROOT.parent / "WaterMark"):
#     if (_cand / "wobbleguard").is_dir() and str(_cand) not in _sys.path:
#         _sys.path.insert(0, str(_cand))

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wobbleguard.config import WatermarkConfig
from wobbleguard.core import WatermarkLogitsProcessor, WatermarkDetector

SECRET_KEY = b"neurips2026-anonymous-evo2-watermark"
DNA_ASCII = {"A": 65, "C": 67, "G": 71, "T": 84}
DNA_ASCII_TO_DNA_IDX = {65: 0, 67: 1, 71: 2, 84: 3}  # ASCII -> 4-token idx
DNA_IDX_TO_ASCII = {0: 65, 1: 67, 2: 71, 3: 84}      # 4-token idx -> ASCII


class EvoWmAdapter:
    # bridges evo2's byte-level vocab with the 4-token watermark processor

    def __init__(self, processor: WatermarkLogitsProcessor):
        self.processor = processor
        self.dna_ids: list[int] = []  # 4-token indexing
        # evo2 ASCII -> our 4-token idx, in canonical (A,C,G,T) order
        self.dna_byte_positions = np.array([65, 67, 71, 84])  # for masking

    def step(self, full_logits: np.ndarray) -> np.ndarray:
        # apply the watermark to the DNA-token slice of the full-vocab evo2
        # logits. takes the raw next-token logits [vocab_size] and returns
        # the modified version: non-DNA tokens at -inf, green-list DNA tokens
        # carrying the watermark bias.
        # pull out the DNA-only 4-vector in (A,C,G,T) order
        dna_4vec = np.array([
            full_logits[DNA_ASCII["A"]],
            full_logits[DNA_ASCII["C"]],
            full_logits[DNA_ASCII["G"]],
            full_logits[DNA_ASCII["T"]],
        ])

        # run the watermark on the 4-vec (processor modifies and returns it)
        dna_4vec_wm = self.processor(self.dna_ids, dna_4vec)

        # write back: mask everything to -inf, then restore the 4 DNA spots
        masked = np.full_like(full_logits, -np.inf)
        for ascii_code, idx in DNA_ASCII_TO_DNA_IDX.items():
            masked[ascii_code] = dna_4vec_wm[idx]

        return masked

    def record(self, sampled_ascii: int) -> None:
        # log the sampled token in 4-token indexing for the next-step hash
        self.dna_ids.append(DNA_ASCII_TO_DNA_IDX[sampled_ascii])


def softmax_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    # temperature-scaled softmax sample from full-vocab logits
    z = logits / temperature
    z = z - np.nanmax(z)  # numerical stability; keeps the -inf masks intact
    p = np.exp(z)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        # degenerate (all -inf or all NaN), so just pick uniformly over DNA
        return int(np.random.choice([65, 67, 71, 84]))
    p = p / s
    return int(np.random.choice(len(p), p=p))


def generate_one(
    model,
    prompt: str,
    adapter: EvoWmAdapter,
    seq_length: int,
    temperature: float,
) -> str:
    # generate one watermarked sequence of length `seq_length` after `prompt`.
    # returns the generated DNA string (prompt not included).
    import torch

    device = next(model.parameters()).device
    # byte-level tokenization: each DNA char becomes its ASCII code
    prompt_ids = [ord(c) for c in prompt.upper()]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ascii: list[int] = []

    with torch.inference_mode():
        for step in range(seq_length):
            outputs = model(input_ids)
            # standard convention: outputs.logits or outputs[0]
            if hasattr(outputs, "logits"):
                last_logits = outputs.logits[0, -1, :]
            elif isinstance(outputs, tuple):
                last_logits = outputs[0][0, -1, :]
            else:
                last_logits = outputs[0, -1, :]
            full_logits = last_logits.float().cpu().numpy()

            wm_logits = adapter.step(full_logits)
            sampled = softmax_sample(wm_logits, temperature=temperature)
            adapter.record(sampled)
            generated_ascii.append(sampled)

            input_ids = torch.cat(
                [input_ids, torch.tensor([[sampled]], dtype=torch.long, device=device)],
                dim=1,
            )

    return "".join(chr(c) for c in generated_ascii)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="7B", choices=["1B", "7B"])
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--prompt", default="ATGAAAGGCTGCAATAGCATCATGAAAGGCTGC",
                        help="DNA prompt seed for all generations")
    parser.add_argument("--seq-length", type=int, default=640,
                        help="Number of nucleotides to generate after prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4,
                        help="Watermark context width")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"watermarked_{args.model.lower()}_results.json"

    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    config = WatermarkConfig(
        key=SECRET_KEY,
        context_width=args.k,
        gamma=args.gamma,
        delta=args.delta,
    )

    print("=" * 70)
    print(f"Tier 4: Watermarked Evo2-{args.model} generation")
    print(f"  config: delta={args.delta}, gamma={args.gamma}, k={args.k}")
    print(f"  num_sequences={args.num_sequences}, seq_length={args.seq_length}")
    print(f"  prompt={args.prompt!r}, temperature={args.temperature}")
    print(f"  seed={args.seed}")
    print("=" * 70)

    print("Loading Evo2...")
    t0 = time.time()
    from evo2 import Evo2
    model_name = "evo2_7b" if args.model == "7B" else "evo2_1b_base"
    evo2 = Evo2(model_name)
    # the actual torch model hangs off evo2.model
    underlying = evo2.model
    underlying.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    detector = WatermarkDetector(config, alpha=0.01)
    results = []
    t_gen_start = time.time()

    for i in range(args.num_sequences):
        # fresh adapter per sequence so the token history starts clean
        adapter = EvoWmAdapter(WatermarkLogitsProcessor(config))
        seq = generate_one(
            underlying, args.prompt, adapter,
            seq_length=args.seq_length, temperature=args.temperature,
        )
        det_result = detector.detect(seq)
        results.append({
            "sample_index": i,
            "prompt": args.prompt,
            "generated_dna": seq,
            "generated_bp": len(seq),
            "watermark_z": float(det_result.z_score),
            "watermark_p": float(det_result.p_value),
            "is_watermarked_at_alpha_0p01": bool(det_result.is_watermarked),
            "green_fraction": float(det_result.green_fraction),
            "config": {"delta": args.delta, "gamma": args.gamma, "k": args.k},
        })
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_gen_start
            rate = (i + 1) / elapsed
            eta = (args.num_sequences - i - 1) / rate if rate > 0 else 0
            tpr_so_far = sum(r["is_watermarked_at_alpha_0p01"] for r in results) / len(results)
            print(
                f"  [{i+1:3d}/{args.num_sequences}] "
                f"z={det_result.z_score:6.2f} | "
                f"running TPR={tpr_so_far:.1%} | "
                f"rate={rate:.2f}/s | ETA {eta/60:.1f}m"
            )

    summary = {
        "model": args.model,
        "config": {
            "delta": args.delta, "gamma": args.gamma, "k": args.k,
            "temperature": args.temperature, "seed": args.seed,
        },
        "num_sequences": args.num_sequences,
        "seq_length": args.seq_length,
        "prompt": args.prompt,
        "elapsed_seconds": time.time() - t_gen_start,
        "tpr_at_alpha_0p01": sum(r["is_watermarked_at_alpha_0p01"] for r in results) / len(results),
        "mean_z": float(np.mean([r["watermark_z"] for r in results])),
        "median_z": float(np.median([r["watermark_z"] for r in results])),
        "min_z": float(np.min([r["watermark_z"] for r in results])),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== Tier 4 Summary (Evo2-{args.model}) ===")
    print(f"  TPR @ alpha=0.01:  {summary['tpr_at_alpha_0p01']:.1%}")
    print(f"  mean z:            {summary['mean_z']:.2f}")
    print(f"  median z:          {summary['median_z']:.2f}")
    print(f"  min z:             {summary['min_z']:.2f}")
    print(f"  elapsed:           {summary['elapsed_seconds']/60:.1f} min")
    print(f"  saved -> {args.output}")


if __name__ == "__main__":
    main()
