"""Run the Evo2-1B jailbreak panel on Modal H100s, on Newton's exact stack.

Why this exists. Every published 1B, 7B and 40B number came from Newton's
`evo2test` environment: evo2 0.1.0, vtx 1.0.6, torch 2.5.1+cu121 and
transformer_engine 1.13.0. That environment installs but cannot execute on the
local RTX 5090, whose compute capability is 12.0 while those wheels carry
kernels only up to sm_90 and ship no PTX to JIT forward. Measured: a plain
matmul fails with "no kernel image is available for execution on the device".
An H100 is sm_90, so the pinned stack runs there unmodified, and trials
generated on it are comparable to the published ones by version identity
rather than by argument.

Division of labour. Generation is GPU work and runs here. BLAST is CPU and
network work at roughly eight minutes per generation, so it runs afterwards on
the local machine against the already-tested eval pipeline. Paying H100 rates
to wait on NCBI would roughly double the bill for nothing. SKIP_BLAST=1 makes
the pipeline save generations and defer validation, which it already supports.

The GPT-4o retrieval agent stays live. It is part of the experimental design,
not a convenience, so its run-to-run variation belongs in the measured
variance. The key arrives as a Modal Secret rather than baked into the image.

Usage:
    modal run  modal_run/evo2_1b_panel.py::verify           # stack check, ~2 min
    modal run  modal_run/evo2_1b_panel.py::smoke            # HIV gate, one virus
    modal run  modal_run/evo2_1b_panel.py::panel --seeds 43,44,45,46
    modal volume get evo2-results / ./newton/results/       # pull generations
"""
import os

import modal

APP_NAME = "evo2-1b-panel"

# Newton's environment, pinned. Versions come from
# newton-environment/evo2export.yml, captured from the cluster's conda env.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "ninja-build")
    # ORDER MATTERS. transformer_engine and flash-attn compile native
    # extensions against whatever torch is installed at build time, and they
    # break with "undefined symbol" if anything moves torch afterwards. vtx
    # 1.0.6 pulls its own torch, which silently replaced 2.5.1 and left the
    # freshly compiled flash_attn .so referencing symbols the downgraded
    # libtorch does not export. So: install every pure-Python dependency
    # first, pin torch in the same command so pip cannot move it, then build
    # the native extensions last against the settled torch.
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # evo2 0.1.0 was never published to PyPI; its earliest release there is
    # 0.2.0. It is the version that produced every published number, so the
    # real 0.1.0 source is copied in below and put on PYTHONPATH. vtx 1.0.6
    # does exist on PyPI. torch is repeated here as a pin, not a reinstall.
    .pip_install(
        "torch==2.5.1",
        "transformers==4.53.2",
        "biopython==1.85",
        "einops==0.8.0",
        "openai",
        "huggingface_hub",
        "numpy<2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "4",
          "NVTE_FRAMEWORK": "pytorch",
          # The CUDA base image has no cuDNN headers, so the TE build dies on
          # "fatal error: cudnn.h: No such file or directory". torch's own
          # nvidia-cudnn-cu12 wheel carries them, at 9.1.0.70, the version
          # Newton pins, so use those rather than a different major version.
          "CPATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/include",
          "LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
          "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib"})
    .run_commands(
        # Fail loudly here rather than at import time in a paid container.
        'python -c "import torch; assert torch.__version__.startswith(\'2.5.1\'), torch.__version__; print(\'torch pinned at\', torch.__version__)"',
        # Modal's bundled Python was built with clang, so distutils links
        # native extensions with clang++ even though nvcc and g++ compile
        # them. Without clang the TE build finishes every compile step and
        # then dies at link. Installed here so the torch layer stays cached.
        "apt-get update && apt-get install -y clang && rm -rf /var/lib/apt/lists/*",
        # Without build isolation pip provisions no build environment, so the
        # builder itself needs wheel, setuptools and pybind11 present.
        "pip install --upgrade pip setuptools wheel packaging pybind11 ninja",
        # vtx 1.0.6 ships as an sdist whose setup.py imports torch, so it
        # needs no build isolation either. It also declares its own torch
        # requirement, which is what silently replaced the pinned 2.5.1 on the
        # previous attempt and left flash-attn linked against symbols the
        # downgraded libtorch no longer exported.
        "pip install --no-build-isolation vtx==1.0.6",
        'python -c "import torch; assert torch.__version__.startswith(\'2.5.1\'), '
        '\'vtx moved torch to \' + torch.__version__; print(\'torch still\', torch.__version__)"',
        "pip install --no-build-isolation transformer-engine-cu12==1.13.0 "
        "transformer-engine-torch==1.13.0 transformer-engine==1.13.0",
        "pip install --no-build-isolation flash-attn==2.7.0.post2",
        # Prove the built extensions load against the torch that shipped.
        'python -c "import torch, flash_attn_2_cuda, transformer_engine.pytorch; '
        'print(\'native extensions OK on torch\', torch.__version__)"',
    )
    .add_local_dir("/home/munish-persaud/evo2/oldstack", "/opt/evo2_010",
                   copy=True, ignore=["__pycache__"])
    .env({"PYTHONPATH": "/opt/evo2_010"})
    .add_local_dir("/home/munish-persaud/evo2/newton", "/work", copy=True,
                   ignore=["results", "__pycache__", ".entrez_cache", "*.log"])
)

app = modal.App(APP_NAME, image=IMAGE)

# The 1B checkpoint is 2.7 GB. Fetch it once, not per container.
hf_cache = modal.Volume.from_name("evo2-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("evo2-results", create_if_missing=True)

VOLUMES = {"/root/.cache/huggingface": hf_cache, "/results": results}
SECRETS = [modal.Secret.from_name("openai")]

# A trial is 23 viruses at roughly five generations each, five rounds of 640
# tokens per generation. Generation-only, so no BLAST wait is included here.
TRIAL_TIMEOUT = 20 * 60 * 60


@app.function(gpu="H100", volumes=VOLUMES, timeout=30 * 60)
def verify():
    """Prove the pinned stack executes here before any trial is paid for."""
    import hashlib
    import subprocess
    import torch

    print(f"torch            {torch.__version__}")
    print(f"built for CUDA   {torch.version.cuda}")
    print(f"arch list        {torch.cuda.get_arch_list()}")
    cap = torch.cuda.get_device_capability(0)
    print(f"device           {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    # The failure that forced this move: a kernel that will not launch.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"bf16 matmul      OK, checksum {(a @ a).float().sum().item():.3f}")

    import transformer_engine
    print(f"transformer_eng  {transformer_engine.__version__}")
    import evo2
    ver = getattr(evo2, "__version__", None)
    if ver is None:
        try:
            from evo2 import version as _v
            ver = getattr(_v, "version", None) or getattr(_v, "__version__", "unknown")
        except Exception:
            ver = "unknown"
    print(f"evo2             {ver}   (Newton: 0.1.0)")

    # The checkpoint must be the one the published numbers used.
    from huggingface_hub import snapshot_download
    path = snapshot_download("arcinstitute/evo2_1b_base")
    print(f"checkpoint path  {path}")
    for f in sorted(os.listdir(path)):
        full = os.path.join(path, f)
        if f.endswith(".pt"):
            size = os.path.getsize(full)
            print(f"  {f}  {size} bytes  (Newton: 2,698,327,083)")
    hf_cache.commit()

    from evo2 import Evo2
    model = Evo2("evo2_1b_base")
    print("model loaded OK")

    # The measurement that failed locally: real sequence against shuffled.
    import random
    import sys
    sys.path.insert(0, "/work")
    from virus_data import VIRUS_DATABASE
    # VIRUS_TUPLES[i][2] is the translated protein, not DNA. Take the same
    # source the local run measured so the two numbers compare directly.
    seq = ""
    for entry in VIRUS_DATABASE.values():
        for field in ("dna_sequence", "sequence", "target_dna"):
            v = entry.get(field) if isinstance(entry, dict) else None
            if isinstance(v, str):
                cand = "".join(c for c in v.upper() if c in "ACGT")
                if len(cand) > 2000:
                    seq = cand[:2000]
                    break
        if seq:
            break
    if len(seq) < 500:
        print("no usable genome found; skipping the separation test")
        return
    shuf = list(seq)
    random.Random(0).shuffle(shuf)
    for label, s in (("real", seq), ("shuffled", "".join(shuf))):
        ids = torch.tensor(model.tokenizer.tokenize(s),
                           dtype=torch.long).unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            out = model.model(ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred = logits[0, :-1].float().argmax(-1)
        acc = (pred == ids[0, 1:]).float().mean().item()
        conf = torch.softmax(logits[0, :-1].float(), -1).max(-1).values
        print(f"  {label:9s} next-token acc {100*acc:5.1f}%   "
              f"confident at {100*(conf > 0.5).float().mean().item():4.1f}% of sites")
    print("\nLocal RTX 5090 for comparison: real 34.6%, shuffled 31.3%, "
          "confident 0.0%. Evo2-7B locally: real 41.5%, shuffled 34.5%, "
          "confident 10.8%.")


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def smoke(seed: int = 900):
    """One virus, agent live. The gate before the full panel.

    BLAST is deferred like the panel's. Running it inline would hold an H100
    idle for half an hour waiting on NCBI, and the generations are five 640 bp
    strings that evaluate in minutes on the local machine against the eval
    pipeline that is already tested.
    """
    return _run(seed, out=f"modal_smoke_s{seed}", extra=["--virus_keys", "hiv"],
                skip_blast=True)


@app.function(gpu="H100", volumes=VOLUMES, secrets=SECRETS,
              timeout=TRIAL_TIMEOUT)
def trial(seed: int):
    """One full 23-virus trial, generation only. BLAST runs locally after."""
    return _run(seed, out=f"modal_1b_s{seed}", extra=[], skip_blast=True)


def _run(seed, out, extra, skip_blast):
    import subprocess
    import glob
    import json

    env = dict(os.environ)
    env["SKIP_BLAST"] = "1" if skip_blast else "0"
    # Resume by default. A virus commits its result.json to the volume when it
    # finishes, and jailbreak_pipeline.py skips any virus whose result.json
    # already carries generations, but only when RESUME=1; the pipeline
    # defaults it to 0. Without this a relaunch after the workspace usage cap
    # stops the run would silently regenerate all 23 viruses per seed and bill
    # the whole panel again. Set RESUME=0 explicitly to force a clean rerun.
    env["RESUME"] = os.environ.get("RESUME", "1")
    env["ENTREZ_CACHE_DIR"] = "/results/.entrez_cache"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(env["ENTREZ_CACHE_DIR"], exist_ok=True)

    outdir = f"/results/{out}"
    cmd = [
        "python3", "jailbreak_pipeline.py",
        "--model_name", "evo2_1b_base",
        "--seed", str(seed),
        "--n_samples", "1",
        "--output_dir", outdir,
    ] + extra

    print(f"seed {seed}: RESUME={env['RESUME']} {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd="/work", env=env).returncode

    # Slurm's COMPLETED lied about this on Newton and result.json presence lies
    # too, so count generations that actually carry sequence.
    n = 0
    for f in glob.glob(os.path.join(outdir, "*", "result.json")):
        try:
            r = json.load(open(f))
            n += len([g for g in (r.get("generations") or [])
                      if g.get("generated_dna")])
        except Exception:
            pass
    viruses = len(glob.glob(os.path.join(outdir, "*", "result.json")))
    results.commit()
    print(f"seed {seed}: rc={rc} viruses={viruses}/23 non_empty_generations={n}",
          flush=True)
    return {"seed": seed, "rc": rc, "viruses": viruses, "generations": n,
            "out": outdir}


@app.local_entrypoint()
def panel(seeds: str = "43,44,45,46"):
    """Fan the seeds out as parallel containers, one H100 each."""
    wanted = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"launching {len(wanted)} trials in parallel: {wanted}")
    for r in trial.map(wanted):
        print(r)
