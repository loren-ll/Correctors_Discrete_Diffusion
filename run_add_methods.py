"""
run_add_methods.py
------------------
Loads existing samples and adds any combination of reverse methods:
- Plain tau-leaping (multiple tau values)
- Tau-leaping + corrector (grid of hyperparameters)

Edit the METHODS TO RUN section below to configure what to add.

Usage
-----
    python run_add_methods.py              # uses all available cores
    python run_add_methods.py --n_jobs 8
"""

import argparse
import numpy as np

from main_code_parallel import add_tau_leap_reverse, add_gillespie_reverse
from saving_experiments import load_samples, save_samples

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

# ── Load existing samples ─────────────────────────────────────────────────────
FILENAME = 'NFE_measure_N_60'

print(f"Loading samples from {FILENAME}...")
samples, _ = load_samples(FILENAME)

# Recover parameters from metadata
w    = np.array(samples.metadata['w'])
mu   = np.array(samples.metadata['mu'])
beta = samples.metadata['beta']
T    = samples.metadata['T']
L    = samples.metadata['L']

already_done = samples.list_methods()
print(f"Methods already in samples: {already_done}")

# ═════════════════════════════════════════════════════════════════════════════
# METHODS TO RUN — edit this section to add/remove methods
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Plain tau-leaping (no corrector) ──────────────────────────────────────
# tau_values = [4, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.08]

# ── 2. Tau-leaping + corrector grid ──────────────────────────────────────────
# Each entry is a dict describing one (tau, corrector) combination.
# Set to [] to skip correctors entirely for now.
corrector_configs = [

    # ── random_masking examples ───────────────────────────────────────────────
    # {
    #     'tau': 0.1,
    #     'corrector_method': 'random_masking',
    #     'corrector_start': T,
    #     'corrector_hyperparameters': {'n_corr': 1, 'tau_c': 0.01, 'apply_reverse': True},
    # },

    # ── PRISM examples ────────────────────────────────────────────────────────
    # {
    #     'tau': 0.1,
    #     'corrector_method': 'PRISM',
    #     'corrector_start': T,
    #     'corrector_hyperparameters': {'eta': 0.2},
    # },
    # {
    #     'tau': 0.1,
    #     'corrector_method': 'PRISM',
    #     'corrector_start': T,
    #     'corrector_hyperparameters': {'eta': 0.5},
    # },

    # ── informed_corrector examples ───────────────────────────────────────────
    # {
    #     'tau': 0.1,
    #     'corrector_method': 'informed_corrector',
    #     'corrector_start': T,
    #     'corrector_hyperparameters': {'K': 1, 'n_corr': 8, 'gamma': 1.0, 'use_margin': True},
    # },

    # ── DPC examples ──────────────────────────────────────────────────────────
    # {
    #     'tau': 0.1,
    #     'corrector_method': 'DPC',
    #     'corrector_start': T,
    #     'corrector_hyperparameters': {'n_corr': 8, 'gamma': 1.0},
    # },
]


corrector_configs = [
    # ----------------------------------PRISM--------------------------------------------------------------
    # ── tau = 1 ───────────────────────────────────────────────────────────────
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 1, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},

    # # ── tau = 0.8 ─────────────────────────────────────────────────────────────
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.8, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},
    #
    # # ── tau = 0.6 ─────────────────────────────────────────────────────────────
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 0.8,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 1.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 2.5,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.0,  'corrector_hyperparameters': {'eta': 1.0}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.2}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.5}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 0.8}},
    # {'tau': 0.6, 'corrector_method': 'PRISM', 'corrector_start': 3.5,  'corrector_hyperparameters': {'eta': 1.0}},

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



    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 1.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 2.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 1, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.2, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.5, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 1, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': True}},
    # {'tau': 1.0, 'corrector_method': 'informed_corrector', 'corrector_start': 3.0, 'corrector_hyperparameters': {'K': 2, 'n_corr': 4, 'gamma': 0.8, 'use_margin': False}},
]


# ═════════════════════════════════════════════════════════════════════════════
# RUN — no need to edit below this line
# ═════════════════════════════════════════════════════════════════════════════

# ── Plain tau-leaping ─────────────────────────────────────────────────────────
for tau in tau_values:
    method_name = f"tau_leap_{tau}"
    if method_name in already_done:
        print(f"Skipping {method_name} (already in samples)")
        continue
    print(f"\n--- Plain tau-leaping: tau={tau} ---")
    add_tau_leap_reverse(
        samples, w, mu, beta, T,
        tau=tau,
        corrector=False,
        n_jobs=args.n_jobs
    )

# ── Tau-leaping + correctors ──────────────────────────────────────────────────
for cfg in corrector_configs:
    tau               = cfg['tau']
    corrector_method  = cfg['corrector_method']
    corrector_start   = cfg['corrector_start']
    hyperparams       = cfg['corrector_hyperparameters']

    # Build the method name the same way add_tau_leap_reverse does,
    # so we can check if it's already been computed
    method_name = f"tau_leap_{tau}_corrector_{corrector_method}_start_{corrector_start}"
    if corrector_method == 'random_masking':
        n_corr = hyperparams.get('n_corr', 1)
        tau_c  = hyperparams.get('tau_c', 0.01)
        apply_reverse = hyperparams.get('apply_reverse', True)
        method_name += f"_ncorr_{n_corr}_tauc_{tau_c}"
        if not apply_reverse:
            method_name += "_fwd_only"
    elif corrector_method == 'PRISM':
        eta = hyperparams.get('eta', 0.2)
        method_name += f"_eta_{eta}"
    elif corrector_method == 'informed_corrector':
        K          = hyperparams.get('K', 1)
        n_corr     = hyperparams.get('n_corr', 8)
        gamma      = hyperparams.get('gamma', 1.0)
        use_margin = hyperparams.get('use_margin', True)
        method_name += f"_K_{K}_ncorr_{n_corr}_gamma_{gamma}"
        if use_margin:
            method_name += "_margin"
    elif corrector_method == 'DPC':
        n_corr = hyperparams.get('n_corr', 8)
        gamma  = hyperparams.get('gamma', 1.0)
        method_name += f"_ncorr_{n_corr}_gamma_{gamma}"

    if method_name in already_done:
        print(f"Skipping {method_name} (already in samples)")
        continue

    print(f"\n--- {method_name} ---")
    add_tau_leap_reverse(
        samples, w, mu, beta, T,
        tau=tau,
        corrector=True,
        corrector_method=corrector_method,
        corrector_start=corrector_start,
        corrector_hyperparameters=hyperparams,
        n_jobs=args.n_jobs
    )

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving updated samples...")
save_samples(
    samples,
    filename=FILENAME,
    lightweight=False,
    L=L,
)

print("Done.")
print(f"Methods now in samples: {samples.list_methods()}")
