"""
run_experiment.py
-----------------
Generates forward and Gillespie reverse samples for the problem instance
defined below, then saves the result (full particles + metadata) to disk.

Usage
-----
    python run_experiment.py              # uses all available cores
    python run_experiment.py --n_jobs 16  # use 16 cores (e.g. SLURM allocation)

The output file will be saved to:
    NFE_measure_N_60/NFE_measure_N_60.pkl
"""

import argparse
import numpy as np

from main_code_parallel import (
    DiffusionSamples,
    generate_forward_samples_at_checkpoints,
    add_gillespie_reverse,
)
from saving_experiments import save_samples

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=-1,
                    help='Number of parallel workers. -1 = all cores.')
args = parser.parse_args()

# ── Problem definition ────────────────────────────────────────────────────────
N, L, r = 60, 3, 5
n_mc    = 100_000
beta, T = 2, 4.5

w   = np.ones(r) / r
u   = np.linspace(0, 1, 31)
checkpoint_times = T * (u ** 12)

# Fixed problem instance — same mu every run
rng   = np.random.default_rng(42)
alpha = np.array([5.0, 1.0, 0.01])   # length L
mu    = rng.dirichlet(alpha, size=(r, N))   # shape (r, N, L)

print(f"Problem: N={N}, L={L}, r={r}, n_mc={n_mc}, beta={beta}, T={T}")
print(f"Checkpoint times: {len(checkpoint_times)} points in "
      f"[{checkpoint_times.min():.4f}, {checkpoint_times.max():.4f}]")
print(f"Parallel workers: {args.n_jobs}")

# ── Build container ───────────────────────────────────────────────────────────
samples = DiffusionSamples(checkpoint_times, n_mc, N, L=L)

# ── Forward process ───────────────────────────────────────────────────────────
print("Running forward process...")
samples.forward = generate_forward_samples_at_checkpoints(
    w, mu, beta, checkpoint_times, n_mc,
    n_jobs=args.n_jobs
)

# ── Gillespie reverse process ─────────────────────────────────────────────────
add_gillespie_reverse(
    samples, w, mu, beta, T,
    n_jobs=args.n_jobs
)

# ── Store all experiment parameters in metadata ───────────────────────────────
samples.metadata["n_mc"]  = n_mc
samples.metadata["N"]     = N
samples.metadata["L"]     = L
samples.metadata["r"]     = r
samples.metadata["w"]     = w
samples.metadata["mu"]    = mu
samples.metadata["beta"]  = beta
samples.metadata["T"]     = T

# ── Save ──────────────────────────────────────────────────────────────────────
save_samples(
    samples,
    filename='NFE_measure_N_60',
    lightweight=False,
    L=L,
)

print("Done.")
