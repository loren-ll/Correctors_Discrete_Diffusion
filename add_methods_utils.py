"""
Utility functions for adding methods to lightweight saves.
"""

import numpy as np
import time
import pickle
import os
from saving_experiments import load_samples
from main_code import (
    add_tau_leap_reverse,
    add_gillespie_reverse,
    compute_empirical_joint_pmf,
    compute_hellinger_from_joint_pmfs,
    DiffusionSamples
)


def add_method_to_lightweight(
    loaded_data, 
    user_metadata, 
    method_config, 
    checkpoint_times,
    existing_methods
):
    """
    Add a single method to the lightweight save.
    
    Parameters
    ----------
    loaded_data : dict
        The loaded lightweight data
    user_metadata : dict
        User metadata with parameters
    method_config : dict
        Configuration for the method to add
        Examples:
        - {'type': 'tau_leap', 'tau': 0.01, 'corrector': False}
        - {'type': 'tau_leap', 'tau': 0.01, 'corrector': True, 
           'corrector_method': 'PRISM', 'corrector_start': 2.0,
           'corrector_hyperparameters': {'eta': 0.5}}
        - {'type': 'gillespie'}
    checkpoint_times : array
        Checkpoint times
    existing_methods : list
        List of already existing method names
    
    Returns
    -------
    method_name : str
        The name of the added method
    method_pmfs : dict
        PMFs for all checkpoints
    """
    # Extract parameters
    N = user_metadata['N']
    L = user_metadata['L']
    r = user_metadata['r']
    beta = user_metadata['beta']
    T = user_metadata['T']
    n_mc = user_metadata['n_mc']
    w = np.array(user_metadata['w'])
    mu = np.array(user_metadata['mu'])
    
    # Create temporary DiffusionSamples
    temp_samples = DiffusionSamples(checkpoint_times, n_mc, N)
    
    # Add dummy forward samples (structure only)
    for t in checkpoint_times:
        temp_samples.forward[float(t)] = np.zeros((n_mc, N), dtype=np.int16)
    
    # Run the method
    method_type = method_config['type']
    
    if method_type == 'tau_leap':
        # Extract tau-leap parameters
        tau = method_config['tau']
        corrector = method_config.get('corrector', False)
        corrector_method = method_config.get('corrector_method', None)
        corrector_start = method_config.get('corrector_start', None)
        corrector_hyperparameters = method_config.get('corrector_hyperparameters', None)
        
        # Run tau-leaping
        add_tau_leap_reverse(
            samples=temp_samples,
            w=w,
            mu=mu,
            beta=beta,
            T=T,
            tau=tau,
            corrector=corrector,
            corrector_method=corrector_method,
            corrector_start=corrector_start,
            corrector_hyperparameters=corrector_hyperparameters
        )
        
    elif method_type == 'gillespie':
        # Run Gillespie
        add_gillespie_reverse(
            samples=temp_samples,
            w=w,
            mu=mu,
            beta=beta,
            T=T
        )
    
    else:
        raise ValueError(f"Unknown method type: {method_type}")
    
    # Get the method name that was created
    new_methods = temp_samples.list_methods()
    new_method_name = [m for m in new_methods if m not in existing_methods][-1]
    
    # Compute PMFs
    method_pmfs = {}
    for t in checkpoint_times:
        particles = temp_samples.reverse_methods[new_method_name][float(t)]
        states, probs = compute_empirical_joint_pmf(particles, N, L, MASK=-1)
        method_pmfs[float(t)] = {'states': states, 'probs': probs}
    
    return new_method_name, method_pmfs


def add_methods_to_lightweight_file(experiment_filename, methods_to_add, verbose=True):
    """
    Add multiple methods to a lightweight save file.
    
    Parameters
    ----------
    experiment_filename : str
        Path to the lightweight .pkl file
    methods_to_add : list of dict
        List of method configurations
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'methods_added': list of method names added
        - 'total_methods': total number of methods now
        - 'file_size_mb': final file size
        - 'existing_methods': all method names
    """
    if verbose:
        print("=" * 70)
        print("ADDING METHODS TO LIGHTWEIGHT SAVE")
        print("=" * 70)
    
    # Load
    if verbose:
        print(f"\nLoading: {experiment_filename}")
    
    loaded_data, user_metadata = load_samples(experiment_filename)
    
    # Verify lightweight
    is_lightweight = isinstance(loaded_data, dict) and loaded_data.get('lightweight', False)
    if not is_lightweight:
        raise ValueError("File is not in lightweight format!")
    
    if verbose:
        print("✓ Lightweight format confirmed")
    
    # Extract info
    N = user_metadata['N']
    L = user_metadata['L']
    beta = user_metadata['beta']
    T = user_metadata['T']
    n_mc = user_metadata['n_mc']
    
    times = loaded_data['times']
    checkpoint_times = times
    existing_methods = list(loaded_data['reverse_pmfs'].keys())
    
    if verbose:
        print(f"\nParameters: N={N}, L={L}, beta={beta}, T={T}, n_mc={n_mc}")
        print(f"Existing methods ({len(existing_methods)}): {existing_methods}")
        print(f"\nAdding {len(methods_to_add)} new methods...")
    
    # Add methods
    methods_added_names = []
    
    for i, method_config in enumerate(methods_to_add, 1):
        # Build description
        if method_config['type'] == 'tau_leap':
            tau = method_config['tau']
            if method_config.get('corrector', False):
                desc = f"tau={tau}, {method_config['corrector_method']}"
            else:
                desc = f"tau={tau}, no corrector"
        elif method_config['type'] == 'gillespie':
            desc = "Gillespie"
        else:
            desc = method_config['type']
        
        if verbose:
            print(f"\n[{i}/{len(methods_to_add)}] Adding: {desc}")
        
        start_time = time.time()
        
        try:
            method_name, method_pmfs = add_method_to_lightweight(
                loaded_data,
                user_metadata,
                method_config,
                checkpoint_times,
                existing_methods
            )
            
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"  ✓ Completed in {elapsed:.2f} seconds")
                print(f"  Method: {method_name}")
            
            # Add to data
            loaded_data['reverse_pmfs'][method_name] = method_pmfs
            existing_methods.append(method_name)
            methods_added_names.append(method_name)
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue
    
    # Save
    if verbose:
        print(f"\nSaving updated file...")
    
    user_metadata['methods'] = existing_methods
    if 'notes' in user_metadata:
        user_metadata['notes'] += f" | Added {len(methods_added_names)} methods"
    else:
        user_metadata['notes'] = f"Added {len(methods_added_names)} methods"
    
    output_data = {
        'times': loaded_data['times'],
        'metadata': loaded_data['metadata'],
        'forward_pmfs': loaded_data['forward_pmfs'],
        'reverse_pmfs': loaded_data['reverse_pmfs'],
        'L': loaded_data['L'],
        'lightweight': True,
        'user_metadata': user_metadata
    }
    
    base_name = experiment_filename.replace('.pkl', '')
    folder = base_name
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, base_name + '.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if verbose:
        print(f"✓ Saved to: {experiment_filename}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total methods: {len(existing_methods)}")
    
    return {
        'methods_added': methods_added_names,
        'total_methods': len(existing_methods),
        'file_size_mb': file_size_mb,
        'existing_methods': existing_methods
    }

def merge_lightweight_files(source_file, target_file, method_names_to_copy=None):
    """
    Copy methods from source file to target file.
    
    Parameters
    ----------
    source_file : str
        File to copy methods FROM
    target_file : str
        File to copy methods TO
    method_names_to_copy : list of str, optional
        Specific methods to copy. If None, copies all methods not already in target.
    
    Returns
    -------
    dict
        Results with 'methods_copied', 'methods_skipped', etc.
    """
    from saving_experiments import load_samples
    import pickle
    import os
    
    # Load both files
    print(f"Loading source: {source_file}")
    source_data, source_metadata = load_samples(source_file)
    
    print(f"Loading target: {target_file}")
    target_data, target_metadata = load_samples(target_file)
    
    # Verify same parameters
    if source_data['times'].shape != target_data['times'].shape:
        raise ValueError("Files have different checkpoint times!")
    
    # Determine which methods to copy
    if method_names_to_copy is None:
        # Copy all methods from source that aren't in target
        method_names_to_copy = list(source_data['reverse_pmfs'].keys())
    
    # Copy methods
    methods_copied = []
    methods_skipped = []
    
    for method_name in method_names_to_copy:
        if method_name not in source_data['reverse_pmfs']:
            print(f"  Warning: {method_name} not in source file, skipping")
            continue
            
        if method_name in target_data['reverse_pmfs']:
            print(f"  Skipped (already in target): {method_name}")
            methods_skipped.append(method_name)
        else:
            target_data['reverse_pmfs'][method_name] = source_data['reverse_pmfs'][method_name]
            methods_copied.append(method_name)
            print(f"  Copied: {method_name}")
    
    # Save updated target
    print(f"\nSaving updated target file...")
    
    target_metadata['methods'] = list(target_data['reverse_pmfs'].keys())
    if 'notes' in target_metadata:
        target_metadata['notes'] += f" | Merged {len(methods_copied)} methods from {source_file}"
    
    output_data = {
        'times': target_data['times'],
        'metadata': target_data['metadata'],
        'forward_pmfs': target_data['forward_pmfs'],
        'reverse_pmfs': target_data['reverse_pmfs'],
        'L': target_data['L'],
        'lightweight': True,
        'user_metadata': target_metadata
    }
    
    base_name = target_file.replace('.pkl', '')
    folder = base_name
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, base_name + '.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"\n✓ Merge complete!")
    print(f"  Methods copied: {len(methods_copied)}")
    print(f"  Methods skipped: {len(methods_skipped)}")
    print(f"  Total methods in target: {len(target_data['reverse_pmfs'])}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return {
        'methods_copied': methods_copied,
        'methods_skipped': methods_skipped,
        'total_methods': len(target_data['reverse_pmfs']),
        'file_size_mb': file_size_mb
    }