"""
Script to add new methods to lightweight (PMF) saves.

Since lightweight saves don't have particles, this script:
1. Loads lightweight save (PMFs only)
2. Extracts parameters (w, mu, beta, T, etc.)
3. Runs NEW tau-leaping reverse process from scratch
4. Computes Hellinger distances against the saved forward PMFs
5. Adds the new method's PMFs and distances to the lightweight save

This allows you to add methods to lightweight saves without re-running forward process!
"""

import numpy as np
import os
import time
from saving_experiments import load_samples, save_samples
from main_code import (
    run_diffusion_experiment, 
    add_tau_leap_reverse,
    compute_empirical_joint_pmf,
    compute_hellinger_from_joint_pmfs
)

# ===== Configuration =====
# Which lightweight experiment to load
EXPERIMENT_FILENAME = 'gillespie_nmc_500.pkl'

# Tau-leaping parameters to try
TAU_VALUES = [0.01, 0.005]

# Corrector configurations
CORRECTOR_CONFIGS = [
    # No corrector
    {'corrector': False},
    
    # PRISM corrector
    {
        'corrector': True,
        'corrector_method': 'PRISM',
        'corrector_start': 2.0,
        'corrector_hyperparameters': {'eta': 0.5}
    },
]

print("=" * 70)
print("ADDING METHODS TO LIGHTWEIGHT SAVE")
print("=" * 70)

# ===== Load Lightweight Save =====
print(f"\nLoading: {EXPERIMENT_FILENAME}")
loaded_data, user_metadata = load_samples(EXPERIMENT_FILENAME)

# Verify it's lightweight
is_lightweight = isinstance(loaded_data, dict) and loaded_data.get('lightweight', False)

if not is_lightweight:
    print("\n⚠️  This is a FULL save, not lightweight!")
    print("Use the other script: add_methods_to_existing_samples.py")
    exit(1)

print("✓ Lightweight format confirmed")

# ===== Extract Parameters =====
print("\n" + "=" * 70)
print("EXTRACTING PARAMETERS")
print("=" * 70)

# Get from metadata
N = user_metadata['N']
L = user_metadata['L']
r = user_metadata['r']
beta = user_metadata['beta']
T = user_metadata['T']
n_mc = user_metadata['n_mc']
w = np.array(user_metadata['w'])
mu = np.array(user_metadata['mu'])

# Get times and existing methods
times = loaded_data['times']
checkpoint_times = times  # These are the forward checkpoint times
existing_methods = list(loaded_data['reverse_pmfs'].keys())

print(f"\nParameters:")
print(f"  N={N}, L={L}, r={r}")
print(f"  beta={beta}, T={T}, n_mc={n_mc}")
print(f"  Checkpoints: {len(checkpoint_times)}")
print(f"\nExisting methods: {existing_methods}")

# ===== Run New Tau-Leaping Methods =====
print("\n" + "=" * 70)
print("RUNNING NEW TAU-LEAPING METHODS")
print("=" * 70)

# We need to create a temporary DiffusionSamples object to run tau-leaping
# But we DON'T need to re-run forward - we'll just use it to run reverse
from main_code import DiffusionSamples

# Create a minimal DiffusionSamples with dummy forward samples
# (We only need the structure, not the actual forward samples)
temp_samples = DiffusionSamples(checkpoint_times, n_mc, N)

# Generate dummy forward samples (won't be used, just need structure)
print("\nGenerating temporary structure for reverse sampling...")
for t in checkpoint_times:
    temp_samples.forward[float(t)] = np.zeros((n_mc, N), dtype=np.int16)

methods_added = 0

for tau in TAU_VALUES:
    for config in CORRECTOR_CONFIGS:
        
        # Build method description
        if config['corrector']:
            method_desc = f"tau={tau}, {config['corrector_method']}"
        else:
            method_desc = f"tau={tau}, no corrector"
        
        print(f"\n[{methods_added + 1}] Running: {method_desc}")
        
        # Time the execution
        start_time = time.time()
        
        # Run tau-leaping reverse
        add_tau_leap_reverse(
            samples=temp_samples,
            w=w,
            mu=mu,
            beta=beta,
            T=T,
            tau=tau,
            **config
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.2f} seconds")
        
        # Get the method name that was created
        new_methods = temp_samples.list_methods()
        new_method_name = [m for m in new_methods if m not in existing_methods][-1]
        
        print(f"  Method name: {new_method_name}")
        
        # ===== Compute PMFs for this method =====
        print(f"  Computing joint PMFs...")
        method_pmfs = {}
        
        for t in checkpoint_times:
            particles = temp_samples.reverse_methods[new_method_name][float(t)]
            states, probs = compute_empirical_joint_pmf(particles, N, L, MASK=-1)
            method_pmfs[float(t)] = {'states': states, 'probs': probs}
        
        # ===== Compute Hellinger Distances =====
        print(f"  Computing Hellinger distances...")
        hellinger_distances = {}
        
        for t in checkpoint_times:
            forward_pmf = loaded_data['forward_pmfs'][float(t)]
            reverse_pmf = method_pmfs[float(t)]
            
            h_dist = compute_hellinger_from_joint_pmfs(
                forward_pmf['states'], forward_pmf['probs'],
                reverse_pmf['states'], reverse_pmf['probs']
            )
            hellinger_distances[float(t)] = h_dist
        
        # Show some distances
        t_sample = [checkpoint_times[0], checkpoint_times[len(checkpoint_times)//2], checkpoint_times[-1]]
        print(f"  Sample Hellinger distances:")
        for t in t_sample:
            print(f"    t={t:.4f}: H={hellinger_distances[float(t)]:.6f}")
        
        # ===== Add to Loaded Data =====
        loaded_data['reverse_pmfs'][new_method_name] = method_pmfs
        existing_methods.append(new_method_name)
        
        methods_added += 1

# ===== Save Updated Lightweight File =====
print("\n" + "=" * 70)
print("SAVING UPDATED LIGHTWEIGHT FILE")
print("=" * 70)

# Update metadata
user_metadata['methods'] = existing_methods
if 'notes' in user_metadata:
    user_metadata['notes'] += f" | Added {methods_added} tau-leap methods"
else:
    user_metadata['notes'] = f"Added {methods_added} tau-leap methods"

# Reconstruct the save data structure
output_data = {
    'times': loaded_data['times'],
    'metadata': loaded_data['metadata'],
    'forward_pmfs': loaded_data['forward_pmfs'],
    'reverse_pmfs': loaded_data['reverse_pmfs'],
    'L': loaded_data['L'],
    'lightweight': True,
    'user_metadata': user_metadata
}

# Save using pickle directly (since we already have the right format)
import pickle

base_name = EXPERIMENT_FILENAME.replace('.pkl', '')
folder = base_name
os.makedirs(folder, exist_ok=True)
filepath = os.path.join(folder, base_name + '.pkl')

with open(filepath, 'wb') as f:
    pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Get file size
file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

print(f"\n✓ Saved to: {EXPERIMENT_FILENAME}")
print(f"  File size: {file_size_mb:.2f} MB")
print(f"  Total methods: {len(existing_methods)}")

# ===== Summary =====
print("\n" + "=" * 70)
print("✓ SUCCESS!")
print("=" * 70)

print(f"\nAdded {methods_added} new methods:")
for method in existing_methods:
    print(f"  - {method}")

print(f"\nYou can now:")
print(f"  1. Load: data, _ = load_samples('{EXPERIMENT_FILENAME}')")
print(f"  2. Plot: plot_method_comparison(data, methods=[...])")
print(f"  3. The file now contains {len(existing_methods)} methods with PMFs")

print("\n" + "=" * 70)