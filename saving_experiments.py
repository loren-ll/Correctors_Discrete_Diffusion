import os
import pickle
import json



def save_samples(samples, filename, metadata=None, create_folder = True):
    """
    Save DiffusionSamples object to disk with all data and metadata.
    
    Parameters
    ----------
    samples : DiffusionSamples
        The samples object to save
    filename : str
        Output filename (will add .pkl extension if not present)
    metadata : dict, optional
        Additional metadata to save (e.g., w, mu, beta, T, tau values)
        Example: {'beta': 5.0, 'T': 3.5, 'tau_values': [0.01, 0.05]}
    """
    base_name = filename.replace('.pkl', '')
    
    if create_folder:
        # Create folder with the base name
        folder = base_name
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, base_name + '.pkl')
    else:
        filepath = base_name + '.pkl'
    
    # Prepare data to save
    data = {
        'times': samples.times,
        'forward': samples.forward,
        'reverse_methods': samples.reverse_methods,
        'metadata': samples.metadata.copy(),  # n_mc, N, etc.
    }
    
    # Add user-provided metadata
    if metadata is not None:
        data['user_metadata'] = metadata
    
    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Samples saved to: {filename}")
    print(f"  - Forward samples: {len(data['forward'])} checkpoints")
    print(f"  - Reverse methods: {list(data['reverse_methods'].keys())}")
    print(f"  - n_mc: {data['metadata']['n_mc']}, N: {data['metadata']['N']}")
    if metadata:
        print(f"  - User metadata: {list(metadata.keys())}")


def load_samples(filename, from_folder=True):
    """
    Load DiffusionSamples object from disk.
    
    Parameters
    ----------
    filename : str
        Input filename
    
    Returns
    -------
    samples : DiffusionSamples
        Reconstructed samples object
    user_metadata : dict or None
        User-provided metadata if it was saved
    """
    from main_code import DiffusionSamples 
    base_name = filename.replace('.pkl', '')
    if from_folder:
        filepath = os.path.join(base_name, base_name + '.pkl')
    else:
        filepath = base_name + '.pkl'
    
    
    # Load from pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct DiffusionSamples object
    times = data['times']
    n_mc = data['metadata']['n_mc']
    N = data['metadata']['N']
    
    samples = DiffusionSamples(times, n_mc, N)
    samples.forward = data['forward']
    samples.reverse_methods = data['reverse_methods']
    samples.metadata = data['metadata']
    
    # Extract user metadata if present
    user_metadata = data.get('user_metadata', None)
    
    print(f"Samples loaded from: {filename}")
    print(f"  - Forward samples: {len(samples.forward)} checkpoints")
    print(f"  - Reverse methods: {samples.list_methods()}")
    print(f"  - n_mc: {n_mc}, N: {N}")
    if user_metadata:
        print(f"  - User metadata: {list(user_metadata.keys())}")
    
    return samples, user_metadata


def save_samples_summary(samples, filename, metadata=None, create_folder=True):
    """
    Save a human-readable JSON summary (without the actual particle data).
    
    Parameters
    ----------
    samples : DiffusionSamples
        The samples object
    filename : str
        Output filename (will add .json extension if not present)
    metadata : dict, optional
        Additional metadata to include
    create_folder : bool, optional
        If True, saves in folder named after filename
    """
    base_name = filename.replace('.pkl', '').replace('.json', '')
    
    if create_folder:
        folder = base_name
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, base_name + '_summary.json')
    else:
        filepath = base_name + '_summary.json'
    
    summary = {
        'n_mc': int(samples.metadata['n_mc']),
        'N': int(samples.metadata['N']),
        'n_checkpoints': len(samples.times),
        'time_range': [float(samples.times[0]), float(samples.times[-1])],
        'reverse_methods': samples.list_methods(),
        'forward_checkpoints': sorted([float(t) for t in samples.forward.keys()]),
    }
    
    if metadata is not None:
        summary['user_metadata'] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {filepath}")


# ==========================================================================================================
# Code to save
# ==========================================================================================================

    # # Prepare metadata
#     experiment_metadata = {
#         'N': N,
#         'L': L,
#         'r': r,
#         'beta': beta,
#         'T': T,
#         'n_mc': n_mc,
#         'tau_values': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
#         'w': w.tolist(),
#         'mu_shape': mu.shape,
#         'checkpoint_formula': 'T * (u ** 12) where u = linspace(0, 1, 31)',
#         'n_checkpoints': len(checkpoint_times),
#         'methods': samples_2.list_methods(),
#         'date': '2026-05-20',
#         'notes': 'Gillespie vs tau-leaping comparison with 6 tau values'
#     }
    
#     # Save full data
#     save_samples(samples_2, 'experiment_N8_L3_beta5_T3.5.pkl', metadata=experiment_metadata)
    
#     # Save summary
#     save_samples_summary(samples_2, 'experiment_N8_L3_beta5_T3.5_summary.json', 
#                         metadata=experiment_metadata)
    


# ==========================================================================================================
# Code to reload
# ==========================================================================================================
# # Load the saved experiment
# loaded_samples, metadata = load_samples('experiment_N8_L3_beta5_T3.5.pkl')

# print(f"Loaded: N={metadata['N']}, beta={metadata['beta']}, methods={metadata['methods']}")

# # Continue analysis
# plot_method_comparison(loaded_samples, methods=['gillespie', 'tau_leap_0.01', 'tau_leap_0.1'])



