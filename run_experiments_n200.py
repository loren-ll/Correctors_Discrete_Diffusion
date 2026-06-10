"""
run_experiment_v2.py
--------------------
Creates the problem, runs forward + Gillespie, saves meta.pkl and
gillespie.pkl, then runs each corrector config and saves its .pkl
immediately after completion.

No large combined file is ever created — everything goes directly
into the methods/ folder.

Usage
-----
    python run_experiment_v2.py              # uses all available cores
    python run_experiment_v2.py --n_jobs 8
"""

import argparse
import os
import gc
import pickle
import numpy as np

from main_code_parallel import (
    DiffusionSamples,
    generate_forward_samples_at_checkpoints,
    add_gillespie_reverse,
    add_tau_leap_reverse,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

# ── Problem definition ────────────────────────────────────────────────────────
N, L, r = 200, 3, 5
n_mc    = 30_000
beta, T = 2, 4.5

w   = np.ones(r) / r
u   = np.linspace(0, 1, 31)
checkpoint_times = T * (u ** 12)

rng   = np.random.default_rng(42)
alpha = np.array([3.0, 1.5, 0.5])
mu    = rng.dirichlet(alpha, size=(r, N))

print(f"Problem: N={N}, L={L}, r={r}, n_mc={n_mc}, beta={beta}, T={T}")
print(f"Dirichlet alpha: {alpha}")
print(f"Parallel workers: {args.n_jobs}")

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = 'methods_n200'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_pkl(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {os.path.basename(path)} ({size_mb:.1f} MB)")


def check_disk():
    stat = os.statvfs('.')
    free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
    print(f"  Free disk: {free_gb:.1f} GB")


# ── Forward process ───────────────────────────────────────────────────────────
print("\nRunning forward process...")
forward = generate_forward_samples_at_checkpoints(
    w, mu, beta, checkpoint_times, n_mc,
    n_jobs=args.n_jobs
)

# ── Save meta.pkl ─────────────────────────────────────────────────────────────
meta_path = os.path.join(OUTPUT_DIR, 'meta.pkl')
if not os.path.exists(meta_path):
    print("\nSaving meta.pkl ...")
    metadata = {
        'N': N, 'L': L, 'r': r,
        'w': w, 'mu': mu,
        'beta': beta, 'T': T,
        'n_mc': n_mc,
    }
    meta = {
        'times':    checkpoint_times,
        'forward':  forward,
        'metadata': metadata,
        'nfe':      {},
        'L':        L,
    }
    save_pkl(meta, meta_path)
    del meta
    gc.collect()
else:
    print("meta.pkl already exists, skipping.")

# ── Gillespie reverse ─────────────────────────────────────────────────────────
gillespie_path = os.path.join(OUTPUT_DIR, 'gillespie.pkl')
if not os.path.exists(gillespie_path):
    print("\nRunning Gillespie reverse...")
    samples = DiffusionSamples(checkpoint_times, n_mc, N, L=L)
    samples.forward = forward
    samples.reverse_methods = {}
    samples.nfe = {}
    samples.metadata = {
        'N': N, 'L': L, 'r': r,
        'w': w, 'mu': mu,
        'beta': beta, 'T': T, 'n_mc': n_mc,
    }

    add_gillespie_reverse(samples, w, mu, beta, T, n_jobs=args.n_jobs)

    save_pkl(samples.reverse_methods['gillespie'], gillespie_path)

    # Update nfe in meta.pkl
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta['nfe']['gillespie'] = samples.nfe.get('gillespie', N)
    save_pkl(meta, meta_path)
    del meta, samples
    gc.collect()
    check_disk()
else:
    print("gillespie.pkl already exists, skipping.")

# ═════════════════════════════════════════════════════════════════════════════
# METHODS TO RUN — edit this section
# ═════════════════════════════════════════════════════════════════════════════

tau_values = [4, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.15, 0.1, 0.08]

corrector_configs = [
    # ------------------------------------------------RM-------------------------------------------------

    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.2, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.05, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.2, 'apply_reverse': True}},


    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 0.6, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 4.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 4, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 8, 'tau_c': 0.1, 'apply_reverse': True}},
    {'tau': 1.0, 'corrector_method': 'random_masking', 'corrector_start': 5.0, 'corrector_hyperparameters': {'n_corr': 12, 'tau_c': 0.1, 'apply_reverse': True}},

 # ------------------------------------------------DPC-------------------------------------------------


    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 0.6, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 1.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 2.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 1, 'gamma': 1.0}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.2}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.5}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 0.8}},
    {'tau': 1.0, 'corrector_method': 'DPC', 'corrector_start': 3.0, 'corrector_hyperparameters': {'n_corr': 4, 'gamma': 1.0}},
   # ---------------------------------Informed Corrector---------------------------------------------------------
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},

    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},



    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},


    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},




    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 0.6, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 8, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.0, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': True}},
    {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 12, 'gamma': 1.5, 'use_margin': False}},







    # ----------------------------------PRISM--------------------------------------------------------------
    # ── tau = 1 ───────────────────────────────────────────────────────────────
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 1.0,'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 1.0, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},

    # # ── tau = 0.8 ─────────────────────────────────────────────────────────────
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},

    # # ── tau = 0.6 ─────────────────────────────────────────────────────────────
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},





]


# ═════════════════════════════════════════════════════════════════════════════
# RUN — no need to edit below this line
# ═════════════════════════════════════════════════════════════════════════════

def get_method_name(tau, corrector, corrector_method, corrector_start, hyperparams):
    name = f"tau_leap_{tau}"
    if not corrector:
        return name
    name += f"_corrector_{corrector_method}_start_{corrector_start}"
    if corrector_method == 'random_masking':
        n_corr = hyperparams.get('n_corr', 1)
        tau_c  = hyperparams.get('tau_c', 0.01)
        apply_reverse = hyperparams.get('apply_reverse', True)
        name += f"_ncorr_{n_corr}_tauc_{tau_c}"
        if not apply_reverse:
            name += "_fwd_only"
    elif corrector_method == 'PRISM':
        name += f"_eta_{hyperparams.get('eta', 0.2)}"
    elif corrector_method == 'informed_corrector':
        K = hyperparams.get('K', 1)
        n_corr = hyperparams.get('n_corr', 8)
        gamma = hyperparams.get('gamma', 1.0)
        use_margin = hyperparams.get('use_margin', True)
        name += f"_K_{K}_ncorr_{n_corr}_gamma_{gamma}"
        if use_margin:
            name += "_margin"
    elif corrector_method == 'DPC':
        name += f"_ncorr_{hyperparams.get('n_corr', 8)}_gamma_{hyperparams.get('gamma', 1.0)}"
    return name


# ── Plain tau-leaping ─────────────────────────────────────────────────────────
for tau in tau_values:
    method_name = f"tau_leap_{tau}"
    out_path = os.path.join(OUTPUT_DIR, f"{method_name}.pkl")
    if os.path.exists(out_path):
        print(f"Skipping {method_name} (already exists)")
        continue

    print(f"\n--- {method_name} ---")
    samples = DiffusionSamples(checkpoint_times, n_mc, N, L=L)
    samples.forward = forward
    samples.reverse_methods = {}
    samples.nfe = {}
    samples.metadata = {'N': N, 'L': L, 'r': r, 'w': w, 'mu': mu,
                        'beta': beta, 'T': T, 'n_mc': n_mc}

    add_tau_leap_reverse(samples, w, mu, beta, T, tau=tau,
                         corrector=False, n_jobs=args.n_jobs)

    save_pkl(samples.reverse_methods[method_name], out_path)

    # Update nfe in meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta['nfe'][method_name] = samples.nfe.get(method_name, None)
    save_pkl(meta, meta_path)

    del samples, meta
    gc.collect()
    check_disk()

# ── Corrector methods ─────────────────────────────────────────────────────────
for cfg in corrector_configs:
    tau              = cfg['tau']
    corrector_method = cfg['corrector_method']
    corrector_start  = cfg['corrector_start']
    hyperparams      = cfg['corrector_hyperparameters']

    method_name = get_method_name(tau, True, corrector_method, corrector_start, hyperparams)
    out_path    = os.path.join(OUTPUT_DIR, f"{method_name}.pkl")

    if os.path.exists(out_path):
        print(f"Skipping {method_name} (already exists)")
        continue

    print(f"\n--- {method_name} ---")
    samples = DiffusionSamples(checkpoint_times, n_mc, N, L=L)
    samples.forward = forward
    samples.reverse_methods = {}
    samples.nfe = {}
    samples.metadata = {'N': N, 'L': L, 'r': r, 'w': w, 'mu': mu,
                        'beta': beta, 'T': T, 'n_mc': n_mc}

    add_tau_leap_reverse(
        samples, w, mu, beta, T,
        tau=tau, corrector=True,
        corrector_method=corrector_method,
        corrector_start=corrector_start,
        corrector_hyperparameters=hyperparams,
        n_jobs=args.n_jobs
    )

    save_pkl(samples.reverse_methods[method_name], out_path)

    # Update nfe in meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta['nfe'][method_name] = samples.nfe.get(method_name, None)
    save_pkl(meta, meta_path)

    del samples, meta
    gc.collect()
    check_disk()

print("\nAll done.")
print(f"Methods folder: {OUTPUT_DIR}/")
check_disk()
