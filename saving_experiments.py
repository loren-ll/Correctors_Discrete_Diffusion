import os
import pickle
import json
import numpy as np


def compute_empirical_pmf(particles, N, L, MASK=-1):
    """
    Compute empirical PMF from particle samples.
    
    Parameters
    ----------
    particles : np.ndarray, shape (n_mc, N)
        Particle samples
    N : int
        Number of dimensions
    L : int
        Vocabulary size
    MASK : int
        Mask token value
    
    Returns
    -------
    pmf : np.ndarray, shape (N, L+1)
        Empirical probability mass function
        Last column (index L) is for MASK token
    """
    n_mc = particles.shape[0]
    pmf = np.zeros((N, L + 1))  # +1 for MASK
    
    for d in range(N):
        for val in range(L):
            pmf[d, val] = np.sum(particles[:, d] == val) / n_mc
        # MASK probability
        pmf[d, L] = np.sum(particles[:, d] == MASK) / n_mc
    
    return pmf


def save_samples(samples, filename, metadata=None, create_folder=True, 
                 lightweight=False, L=None):
    """
    Save DiffusionSamples object to disk.
    
    Parameters
    ----------
    samples : DiffusionSamples
        The samples object to save
    filename : str
        Output filename (will add .pkl extension if not present)
    metadata : dict, optional
        Additional metadata to save
    create_folder : bool
        If True, creates a folder with base name
    lightweight : bool
        If True, only save empirical PMFs (not full particle data)
        Requires L parameter
    L : int, optional
        Vocabulary size (required if lightweight=True)
    """
    base_name = filename.replace('.pkl', '')
    
    if create_folder:
        folder = base_name
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, base_name + '.pkl')
    else:
        filepath = base_name + '.pkl'
    
    if lightweight:
        if L is None:
            raise ValueError("L (vocabulary size) is required for lightweight save")
        
        print("Computing empirical PMFs for lightweight save...")
        N = samples.metadata['N']
        MASK = -1  # Assuming MASK=-1
        
        # Compute PMFs for forward samples
        forward_pmfs = {}
        for t in samples.times:
            particles = samples.forward[float(t)]
            forward_pmfs[float(t)] = compute_empirical_pmf(particles, N, L, MASK)
        
        # Compute PMFs for reverse methods
        reverse_pmfs = {}
        for method in samples.list_methods():
            reverse_pmfs[method] = {}
            for t in samples.times:
                particles = samples.reverse_methods[method][float(t)]
                reverse_pmfs[method][float(t)] = compute_empirical_pmf(particles, N, L, MASK)
        
        data = {
            'times': samples.times,
            'metadata': samples.metadata.copy(),
            'forward_pmfs': forward_pmfs,
            'reverse_pmfs': reverse_pmfs,
            'L': L,  # Store vocab size for reconstruction
            'lightweight': True
        }
    else:
        # Full save (default)
        data = {
            'times': samples.times,
            'forward': samples.forward,
            'reverse_methods': samples.reverse_methods,
            'metadata': samples.metadata.copy(),
            'lightweight': False
        }
    
    # Add user-provided metadata
    if metadata is not None:
        data['user_metadata'] = metadata
    
    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if lightweight:
        print(f"Samples saved (LIGHTWEIGHT - PMFs) to: {filename}")
        print(f"  - Forward PMFs: {len(forward_pmfs)} checkpoints")
        print(f"  - Reverse PMFs: {len(reverse_pmfs)} methods")
        print(f"  - File size: {file_size_mb:.2f} MB")
    else:
        print(f"Samples saved (FULL) to: {filename}")
        print(f"  - Forward samples: {len(data['forward'])} checkpoints")
        print(f"  - Reverse methods: {list(data['reverse_methods'].keys())}")
        print(f"  - File size: {file_size_mb:.2f} MB")
    
    print(f"  - n_mc: {data['metadata']['n_mc']}, N: {data['metadata']['N']}")
    if metadata:
        print(f"  - User metadata: {list(metadata.keys())}")


def load_samples(filename, from_folder=True):
    """
    Load DiffusionSamples object from disk.
    
    Automatically detects if the file is lightweight (PMFs) or full (particles).
    
    Parameters
    ----------
    filename : str
        Input filename
    from_folder : bool
        If True, looks for file inside a folder with the same base name
    
    Returns
    -------
    samples : DiffusionSamples or dict
        If full: reconstructed DiffusionSamples object
        If lightweight: dict with PMFs and metadata
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
    
    # Check if lightweight
    is_lightweight = data.get('lightweight', False)
    
    if is_lightweight:
        # Lightweight format - return as dict with PMFs
        print(f"Samples loaded (LIGHTWEIGHT - PMFs) from: {filename}")
        print(f"  - Forward PMFs: {len(data['forward_pmfs'])} checkpoints")
        print(f"  - Reverse PMFs: {len(data['reverse_pmfs'])} methods")
        print(f"  - n_mc: {data['metadata']['n_mc']}, N: {data['metadata']['N']}, L: {data['L']}")
        
        user_metadata = data.get('user_metadata', None)
        if user_metadata:
            print(f"  - User metadata: {list(user_metadata.keys())}")
        
        return data, user_metadata
    
    else:
        # Full format - reconstruct DiffusionSamples object
        times = data['times']
        n_mc = data['metadata']['n_mc']
        N = data['metadata']['N']
        
        samples = DiffusionSamples(times, n_mc, N)
        samples.forward = data['forward']
        samples.reverse_methods = data['reverse_methods']
        samples.metadata = data['metadata']
        
        user_metadata = data.get('user_metadata', None)
        
        print(f"Samples loaded (FULL) from: {filename}")
        print(f"  - Forward samples: {len(samples.forward)} checkpoints")
        print(f"  - Reverse methods: {samples.list_methods()}")
        print(f"  - n_mc: {n_mc}, N: {N}")
        if user_metadata:
            print(f"  - User metadata: {list(user_metadata.keys())}")
        
        return samples, user_metadata