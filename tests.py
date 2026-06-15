import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import gc
from main_code import (
    run_diffusion_experiment,
    add_gillespie_reverse,
    DiffusionSamples,
    compute_empirical_joint_pmf,
    compute_hellinger_from_joint_pmfs,
    add_tau_leap_reverse
)

from itertools import combinations
from collections import defaultdict
 

# ==============================================================================
# PROBLEM SETUP
# ==============================================================================

def make_problem(N, L, r, w, seed=None):
    """
    Generate a fixed problem instance (mu) from the given parameters.

    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights (will be normalised internally).
    seed : int or None
        Random seed for reproducibility of mu.

    Returns
    -------
    w : np.ndarray, shape (r,)
        Normalised mixture weights.
    mu : np.ndarray, shape (r, N, L)
        Component-wise categorical probabilities.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mu = rng.dirichlet(np.ones(L), size=(r, N))   # shape (r, N, L)
    return w, mu


# ==============================================================================
# HELPER: single-shuffle Hellinger at every checkpoint
# ==============================================================================

def _shuffle_hellinger_fast(forward, reverse, times, n_mc, rng, MASK=-1):
    """
    Exact shuffle-null Hellinger using counts, not particle shuffling.

    Equivalent to:
        combined = concat(forward, reverse)
        shuffled = permutation(combined)
        A = shuffled[:n_mc]
        B = shuffled[n_mc:]
    but much faster.
    """
    out = {}

    for t in times:
        t = float(t)

        Xf = forward[t]
        Xr = reverse[t]

        combined = np.vstack([Xf, Xr])

        # Count unique joint states in the combined sample
        states, counts = np.unique(combined, axis=0, return_counts=True)

        # Randomly allocate n_mc particles to group A without replacement
        # from the combined empirical distribution.
        counts_A = rng.multivariate_hypergeometric(counts, n_mc)
        counts_B = counts - counts_A

        p = counts_A / n_mc
        q = counts_B / n_mc

        # Hellinger distance
        out[t] = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    return out


# ==============================================================================
# TEST 1: test_shuffle_gillespie
# ==============================================================================

def test_shuffle_gillespie(
    N, L, r, w, beta, T, n_mc, checkpoint_times,
    mu_seed=0,
    confidence_intervals=False,
    n_bootstrap=100,
    ci_level=0.95,
    plot_filename='test_shuffle_gillespie.png',
):
    """
    Run the Gillespie forward/reverse experiment, then validate by comparing
    the true Hellinger distance against a shuffle-based null distribution.

    Steps
    -----
    1. Generate a fixed problem instance (mu) from the given parameters.
    2. Run the forward process and Gillespie reverse process.
    3. Compute the true Hellinger distance at each checkpoint.
    4. Combine forward + reverse particles at each checkpoint, randomly split
       into two equal groups, compute Hellinger distance of those groups
       (this is the "null" — what we expect if both processes match).
    5. If confidence_intervals=True, repeat step 4 n_bootstrap times and
       plot a (ci_level*100)% confidence band around the null.

    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    n_mc : int
        Number of Monte Carlo samples.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    confidence_intervals : bool
        If True, perform n_bootstrap shuffles and show a CI band.
    n_bootstrap : int
        Number of bootstrap shuffles (used when confidence_intervals=True).
    ci_level : float
        Confidence level for the interval, e.g. 0.95.
    plot_filename : str
        Path to save the output plot.

    Returns
    -------
    samples : DiffusionSamples
        The samples object with forward and Gillespie reverse particles.
    original_distances : dict[float, float]
        Hellinger distances at each checkpoint.
    shuffle_distances : dict[float, float]
        Hellinger distances after one random shuffle (null).
    """
    print("=" * 70)
    print("TEST: Shuffle Gillespie")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Problem setup
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))

    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, n_mc={n_mc}")
    print(f"  checkpoints: {len(checkpoint_times)}  (mu_seed={mu_seed})")

    # ------------------------------------------------------------------
    # 2. Forward + Gillespie reverse
    # ------------------------------------------------------------------
    samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
    add_gillespie_reverse(samples, w, mu, beta, T)

    # ------------------------------------------------------------------
    # 3. True Hellinger distances
    # ------------------------------------------------------------------
    print("\nComputing true Hellinger distances...")
    original_distances = samples.compute_hellinger_distances('gillespie')

    # ------------------------------------------------------------------
    # 4. Single-shuffle null
    # ------------------------------------------------------------------
    print("Computing single-shuffle null distances...")
    rng = np.random.default_rng(42)
    shuffle_distances = _shuffle_hellinger_fast(
        samples.forward,
        samples.reverse_methods['gillespie'],
        samples.times,
        n_mc,
        rng,
    )

    # ------------------------------------------------------------------
    # 5. Bootstrap CI (optional)
    # ------------------------------------------------------------------
    bootstrap_matrix = None
    if confidence_intervals:
        print(f"Running {n_bootstrap} bootstrap shuffles for CI (level={ci_level})...")
        bootstrap_matrix = np.zeros((n_bootstrap, len(samples.times)))
        for b in range(n_bootstrap):
            if b % 100 == 0:
                print(f"  Bootstrap {b}/{n_bootstrap}")
            b_rng = np.random.default_rng(b)
            b_dist = _shuffle_hellinger_fast(
                samples.forward,
                samples.reverse_methods['gillespie'],
                samples.times,
                n_mc,
                b_rng,
            )
            for k, t in enumerate(samples.times):
                bootstrap_matrix[b, k] = b_dist[float(t)]

        alpha = 1.0 - ci_level
        ci_lo = np.quantile(bootstrap_matrix, alpha / 2,     axis=0)
        ci_hi = np.quantile(bootstrap_matrix, 1.0 - alpha / 2, axis=0)

    # # ------------------------------------------------------------------
    # # 6. Plot
    # # ------------------------------------------------------------------
    # print("\nGenerating plot...")
    # plt.figure(figsize=(10, 6))

    # times_arr = samples.times
    # orig_arr    = np.array([original_distances[float(t)] for t in times_arr])
    # shuffle_arr = np.array([shuffle_distances[float(t)]  for t in times_arr])

    # plt.plot(times_arr, orig_arr,    'o-', color='royalblue',  linewidth=2,
    #          markersize=4, label='True Hellinger (Gillespie vs Forward)')
    # plt.plot(times_arr, shuffle_arr, 's--', color='forestgreen', linewidth=2,
    #          markersize=4, label='Shuffle null (single)')

    # if confidence_intervals and bootstrap_matrix is not None:
    #     plt.fill_between(
    #         times_arr, ci_lo, ci_hi,
    #         color='forestgreen', alpha=0.2,
    #         label=f'Shuffle {int(ci_level*100)}% CI ({n_bootstrap} bootstraps)',
    #     )

    # plt.xlabel('Physical time t', fontsize=12)
    # plt.ylabel('Hellinger distance', fontsize=12)
    # plt.title('Gillespie: True Hellinger vs Shuffle Null', fontsize=14)
    # plt.legend(fontsize=10)
    # plt.grid(True, alpha=0.3)
    # plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    # print(f"Plot saved to: {plot_filename}")
    # plt.show()

    # # ------------------------------------------------------------------
    # # 7. Summary
    # # ------------------------------------------------------------------
    # print("\n" + "=" * 70)
    # print("SUMMARY")
    # print("=" * 70)
    # print(f"  True Hellinger   — mean: {orig_arr.mean():.4f}  "
    #       f"max: {orig_arr.max():.4f}  at t=0: {original_distances[float(times_arr[0])]:.4f}")
    # print(f"  Shuffle null     — mean: {shuffle_arr.mean():.4f}  "
    #       f"max: {shuffle_arr.max():.4f}  at t=0: {shuffle_distances[float(times_arr[0])]:.4f}")
    # if confidence_intervals and bootstrap_matrix is not None:
    #     print(f"  Shuffle CI [{int(ci_level*100)}%] at t=0: "
    #           f"[{ci_lo[0]:.4f}, {ci_hi[0]:.4f}]")
    # print("=" * 70)

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    times_arr = samples.times
    orig_arr = np.array([original_distances[float(t)] for t in times_arr])
    shuffle_arr = np.array([shuffle_distances[float(t)] for t in times_arr])

    cmap = plt.cm.plasma
    gillespie_color = cmap(0.15)   # dark purple
    shuffle_color = cmap(0.72)     # orange

    ax.plot(
        times_arr,
        orig_arr,
        marker="o",
        color=gillespie_color,
        linewidth=2.6,
        markersize=4.8,
        label="Gillespie vs forward",
    )

    if confidence_intervals and bootstrap_matrix is not None:
        ax.fill_between(
            times_arr,
            ci_lo,
            ci_hi,
            color=shuffle_color,
            alpha=0.28,
            label=f"Shuffle null {int(ci_level * 100)}% band",
        )
    else:
        ax.plot(
            times_arr,
            shuffle_arr,
            marker="s",
            linestyle="--",
            color=shuffle_color,
            linewidth=2.4,
            markersize=4.5,
            label="Shuffle null",
        )

    ax.set_xlabel(r"Physical time $t$", fontsize=13)
    ax.set_ylabel("Hellinger distance", fontsize=13)
    # ax.set_title(
    #     "Gillespie Reverse Sampler: Shuffle Null Validation",
    #     fontsize=15,
    #     fontweight="bold",
    #     pad=12,
    # )

    ax.set_xlim(times_arr[0], times_arr[-1])
    ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.35)

    ax.legend(
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_filename}")
    plt.show()
    return samples, original_distances, shuffle_distances

# ==============================================================================
# HELPER: find closest checkpoint time to a target
# ==============================================================================
 
def _closest_time(times, target):
    """Return the value in `times` closest to `target`."""
    times = np.asarray(times)
    return float(times[np.argmin(np.abs(times - target))])
 
 
# ==============================================================================
# TEST 2: test_nmc_convergence_gillespie
# ==============================================================================
 
def test_nmc_convergence_gillespie(
    N, L, r, w, beta, T,
    nmc_list,
    checkpoint_times,
    mu_seed=0,
    plot_filename='test_nmc_convergence_gillespie.png',
):
    """
    Run Gillespie forward/reverse for each n_mc in nmc_list on the same fixed
    problem instance (same mu), compute Hellinger distance at every checkpoint,
    plot all curves together, annotate t=0, and print a table at
    t = 0, T/4, T/2, 3T/4, T.
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    nmc_list : list of int
        List of Monte Carlo sample sizes to compare, e.g. [1000, 10000, 100000].
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    results : dict[int, dict[float, float]]
        results[n_mc][t] = Hellinger distance at checkpoint t.
    """
    print("=" * 70)
    print("TEST: n_mc Convergence (Gillespie)")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Fixed problem instance
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, mu_seed={mu_seed}")
    print(f"  checkpoints: {len(checkpoint_times)}")
    print(f"  nmc_list: {nmc_list}")
 
    # ------------------------------------------------------------------
    # 2. Table checkpoints: t=0, T/4, T/2, 3T/4, T
    # ------------------------------------------------------------------
    table_targets = [0.0, T / 4, T / 2, 3 * T / 4, T]
    table_times   = [_closest_time(checkpoint_times, tgt) for tgt in table_targets]
    table_labels  = ['t=0', 't=T/4', 't=T/2', 't=3T/4', 't=T']
 
    # ------------------------------------------------------------------
    # 3. Run for each n_mc
    # ------------------------------------------------------------------
    results = {}
 
    for n_mc in nmc_list:
        print(f"\n--- n_mc = {n_mc} ---")
        samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
        add_gillespie_reverse(samples, w, mu, beta, T)
        distances = samples.compute_hellinger_distances('gillespie')
        results[n_mc] = distances
 
    # # ------------------------------------------------------------------
    # # 4. Plot
    # # ------------------------------------------------------------------
    # print("\nGenerating plot...")
    # fig, ax = plt.subplots(figsize=(10, 6))
 
    # colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nmc_list)))
 
    # for color, n_mc in zip(colors, nmc_list):
    #     distances = results[n_mc]
    #     times_arr = checkpoint_times
    #     dist_arr  = np.array([distances[float(t)] for t in times_arr])
 
    #     ax.plot(times_arr, dist_arr, 'o-', color=color, linewidth=2,
    #             markersize=4, label=f'n_mc={n_mc:,}')
 
    #     # Annotate t=0 value
    #     t0 = _closest_time(times_arr, 0.0)
    #     h0 = distances[t0]
    #     ax.annotate(
    #         f'{h0:.4f}',
    #         xy=(t0, h0),
    #         xytext=(6, 0),
    #         textcoords='offset points',
    #         fontsize=8,
    #         color=color,
    #     )
 
    # ax.set_xlabel('Physical time t', fontsize=12)
    # ax.set_ylabel('Hellinger distance', fontsize=12)
    # ax.set_title('Gillespie: Hellinger Distance vs n_mc', fontsize=14)
    # ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)
    # ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    # print(f"Plot saved to: {plot_filename}")
    # plt.show()
    # ------------------------------------------------------------------
    # 4. Plot: polished version
    # ------------------------------------------------------------------
    print("\nGenerating plot...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(nmc_list)))

    for color, n_mc in zip(colors, nmc_list):
        distances = results[n_mc]
        times_arr = checkpoint_times
        dist_arr = np.array([distances[float(t)] for t in times_arr])

        t0 = _closest_time(times_arr, 0.0)
        h0 = distances[t0]

        ax.plot(
            times_arr,
            dist_arr,
            marker="o",
            linewidth=2.4,
            markersize=4.5,
            color=color,
            label=fr"$n_{{mc}}={n_mc:,}$   ($H(0)={h0:.4f}$)",
            alpha=0.95,
        )


    ax.set_xlabel(r"Physical time $t$", fontsize=13)
    ax.set_ylabel("Hellinger distance", fontsize=13)
    # ax.set_title(
    #     "Gillespie Reverse Sampler: Monte Carlo Convergence",
    #     fontsize=15,
    #     fontweight="bold",
    #     pad=12,
    # )

    ax.set_xlim(checkpoint_times[0], checkpoint_times[-1])
    ax.set_ylim(bottom=0)

    ax.grid(True, which="major", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Sample size",
        fontsize=10,
        title_fontsize=10,
        frameon=True,
        framealpha=0.9,
        loc="best",
    )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.35)

    # Optional: mark table checkpoints
    for t in table_times:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.25)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_filename}")
    plt.show()
    
    # ------------------------------------------------------------------
    # 5. Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HELLINGER DISTANCES AT KEY CHECKPOINTS")
    print("=" * 70)
 
    # Header
    col_w = 12
    header = f"{'n_mc':<12}" + "".join(f"{lbl:>{col_w}}" for lbl in table_labels)
    print(header)
    print("-" * len(header))
 
    for n_mc in nmc_list:
        row = f"{n_mc:<12}"
        for t in table_times:
            h = results[n_mc].get(float(t), float('nan'))
            row += f"{h:>{col_w}.4f}"
        print(row)
 
    print("=" * 70)
 
    return results


# ==============================================================================
# TEST 3: test_joint_vs_full_marginal
# ==============================================================================
 
def test_joint_vs_full_marginal(
    N, L, r, w, beta, T, n_mc, checkpoint_times,
    method='gillespie',
    mu_seed=0,
    plot_filename='test_joint_vs_full_marginal.png',
):
    """
    Sanity check: joint Hellinger distance should equal marginal Hellinger
    distance with k=N (the only N-dimensional subset is the full space).
 
    Steps
    -----
    1. Generate a fixed problem instance and run forward + reverse.
    2. Compute joint Hellinger at every checkpoint via compute_hellinger_distances.
    3. Compute marginal Hellinger with k=N, M=1 via compute_marginal_hellinger.
    4. Compare the two at every checkpoint, print a table, and plot both curves.
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    n_mc : int
        Number of Monte Carlo samples.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    method : str
        Reverse method to use, e.g. 'gillespie'.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    joint_distances : dict[float, float]
        Joint Hellinger distances at each checkpoint.
    marginal_distances : dict[float, float]
        Marginal Hellinger distances (k=N) at each checkpoint.
    """
    print("=" * 70)
    print("TEST: Joint Hellinger == Marginal Hellinger (k=N)")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Problem setup and simulation
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, n_mc={n_mc}, mu_seed={mu_seed}")
    print(f"  method={method}, k=N={N}")
 
    samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
 
    if method == 'gillespie':
        add_gillespie_reverse(samples, w, mu, beta, T)
    else:
        raise ValueError(f"Method '{method}' not supported in this test yet.")
 
    # ------------------------------------------------------------------
    # 2. Joint Hellinger
    # ------------------------------------------------------------------
    print("\nComputing joint Hellinger distances...")
    joint_distances = samples.compute_hellinger_distances(method)
 
    # ------------------------------------------------------------------
    # 3. Marginal Hellinger with k=N, M=1 (only one subset exists)
    # ------------------------------------------------------------------
    print("Computing marginal Hellinger distances (k=N, M=1)...")
    marginal_results = samples.compute_marginal_hellinger(method, k=N, M=1)
    # marginal_results[t]['h'] is the average over subsets (only one here)
    marginal_distances = {t: marginal_results[t]['h'] for t in marginal_results}
 
    # ------------------------------------------------------------------
    # 4. Print table
    # ------------------------------------------------------------------
    times_arr = checkpoint_times
    table_targets = [0.0, T / 4, T / 2, 3 * T / 4, T]
    table_times   = [_closest_time(times_arr, tgt) for tgt in table_targets]
    table_labels  = ['t=0', 't=T/4', 't=T/2', 't=3T/4', 't=T']
 
    print("\n" + "=" * 70)
    print("JOINT vs MARGINAL (k=N) HELLINGER AT KEY CHECKPOINTS")
    print("=" * 70)
 
    col_w = 12
    header = f"{'checkpoint':<12}" + "".join(f"{lbl:>{col_w}}" for lbl in table_labels)
    print(header)
    print("-" * len(header))
 
    for row_label, row_dict in [("Joint", joint_distances), ("Marginal s=D", marginal_distances)]:
        row = f"{row_label:<12}"
        for t in table_times:
            h = row_dict.get(float(t), float('nan'))
            row += f"{h:>{col_w}.4f}"
        print(row)
 
    # Differences
    row = f"{'Difference':<12}"
    for t in table_times:
        diff = abs(joint_distances.get(float(t), float('nan')) -
                   marginal_distances.get(float(t), float('nan')))
        row += f"{diff:>{col_w}.4f}"
    print(row)
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
 
    joint_arr    = np.array([joint_distances[float(t)]    for t in times_arr])
    marginal_arr = np.array([marginal_distances[float(t)] for t in times_arr])
    diff_arr     = np.abs(joint_arr - marginal_arr)
 
    plt.plot(times_arr, joint_arr,    'o-',  color='#0d0887',   linewidth=2,
             markersize=4, label='Joint Hellinger')
    plt.plot(times_arr, marginal_arr, 's--', color='#cc4778',   linewidth=2,
             markersize=4, label='Marginal Hellinger (s=D)')
    plt.plot(times_arr, diff_arr,     '^:',  color='#f89540',   linewidth=1.5,
             markersize=3, label='|Difference|')
 
    plt.xlabel('Physical time t', fontsize=12)
    plt.ylabel('Hellinger distance', fontsize=12)
    # plt.title(f'Joint vs Marginal (s=D) Hellinger — {method}', fontsize=14)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True,  alpha=0.25, linestyle=':')
    # plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    # ------------------------------------------------------------------
    # 6. Pass / Fail
    # ------------------------------------------------------------------
    max_diff = diff_arr.max()
    tol = 1e-10
    if max_diff < tol:
        print(f"\n✓ PASSED — max |difference| = {max_diff:.2e} < {tol:.2e}")
    else:
        print(f"\n✗ NOTE   — max |difference| = {max_diff:.6f}  "
              f"(non-zero due to sampling noise in empirical PMFs, expected to be small)")
 
    return joint_distances, marginal_distances


# ==============================================================================
# TEST 4: test_hellinger_vs_k
# ==============================================================================
 
def test_hellinger_vs_k(
    N, L, r, w, beta, T, n_mc, checkpoint_times,
    k_list,
    method='gillespie',
    mu_seed=0,
    plot_filename='test_hellinger_vs_k.png',
):
    """
    For a fixed problem and fixed particles, compute the average marginal
    Hellinger distance H(k) for each k in k_list (using M=1 random subset
    per k), and plot H(k) vs k.
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    n_mc : int
        Number of Monte Carlo samples.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    k_list : list of int
        Orders of marginal to compute, e.g. [1, 2, 3, 4].
        Each must satisfy 1 <= k <= N.
    method : str
        Reverse method to use, e.g. 'gillespie'.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    results : dict[int, dict[float, dict]]
        results[k][t] = {'h': float, 'per_subset': array, 'subsets': list}
        as returned by compute_marginal_hellinger.
    """
    print("=" * 70)
    print("TEST: H(k) vs k")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Problem setup and simulation
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, n_mc={n_mc}, mu_seed={mu_seed}")
    print(f"  method={method}, k_list={k_list}")
 
    samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
 
    if method == 'gillespie':
        add_gillespie_reverse(samples, w, mu, beta, T)
    else:
        raise ValueError(f"Method '{method}' not supported in this test yet.")
 
    # ------------------------------------------------------------------
    # 2. Compute H(k) for each k, using M=1 random subset
    # ------------------------------------------------------------------
    print("\nComputing marginal Hellinger for each k...")
    results = {}
    for k in k_list:
        # print(f"  k={k} ...")
        results[k] = samples.compute_marginal_hellinger(method, k=k, M=1)
 
    # ------------------------------------------------------------------
    # 3. Table: H(k) at key checkpoints
    # ------------------------------------------------------------------
    table_targets = [0.0, T / 4, T / 2, 3 * T / 4, T]
    table_times   = [_closest_time(checkpoint_times, tgt) for tgt in table_targets]
    table_labels  = ['t=0', 't=T/4', 't=T/2', 't=3T/4', 't=T']
 
    print("\n" + "=" * 70)
    print("H(k) AT KEY CHECKPOINTS")
    print("=" * 70)
 
    col_w = 12
    header = f"{'k':<8}" + "".join(f"{lbl:>{col_w}}" for lbl in table_labels)
    print(header)
    print("-" * len(header))
 
    for k in k_list:
        row = f"{k:<8}"
        for t in table_times:
            h = results[k][float(t)]['h']
            row += f"{h:>{col_w}.4f}"
        print(row)
 
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 4. Plot: H(k) vs k, one curve per checkpoint
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
 
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(table_times)))
 
    for color, t, lbl in zip(colors, table_times, table_labels):
        h_vals = np.array([results[k][float(t)]['h'] for k in k_list])
        ax.plot(k_list, h_vals, 'o-', color=color, linewidth=2,
                markersize=6, label=lbl)
 
    ax.set_xlabel('Marginal order k', fontsize=12)
    ax.set_ylabel('H(k)  (marginal Hellinger distance)', fontsize=12)
    # ax.set_title(f'Marginal Hellinger H(k) vs k — {method}  (M=1)', fontsize=14)
    ax.set_xticks(k_list)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    return results


# ==============================================================================
# TEST 5: test_approximation_error_vs_nmc
# ==============================================================================
 
def test_approximation_error_vs_nmc(
    N, L, r, w, beta, T,
    nmc_list,
    checkpoint_times,
    k,
    method='gillespie',
    mu_seed=0,
    plot_filename='test_approximation_error_vs_nmc.png',
):
    """
    Efficiency argument: show that H_marginal(k) reaches a stable estimate
    with far fewer samples than H_joint.
 
    For each n_mc in nmc_list (on the same fixed problem/mu):
      - Compute H_joint(n_mc)        at t=0
      - Compute H_marginal(k, n_mc)  at t=0  (M=1 random subset)
 
    The reference "truth" is H_joint computed at max(nmc_list).
 
    Then plot three curves vs n_mc:
      - H_joint(n_mc)
      - H_marginal(k, n_mc)
      - |H_marginal(k, n_mc) - H_joint_ref|   (gap to reference)
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    nmc_list : list of int
        List of Monte Carlo sample sizes, e.g. [500, 1000, 5000, 10000, 50000].
        The largest value is used as the reference for H_joint.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    k : int
        Marginal order for H_marginal, e.g. k=2.
    method : str
        Reverse method to use, e.g. 'gillespie'.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    h_joint : dict[int, float]
        H_joint at t=0 for each n_mc.
    h_marginal : dict[int, float]
        H_marginal(k) at t=0 for each n_mc.
    h_joint_ref : float
        Reference H_joint at max(nmc_list).
    """
    print("=" * 70)
    print("TEST: Approximation Error vs n_mc")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Problem setup
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
    t0 = _closest_time(checkpoint_times, 0.0)
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, mu_seed={mu_seed}")
    print(f"  method={method}, k={k}, evaluating at t={t0}")
    print(f"  nmc_list={nmc_list}")
    print(f"  Reference: H_joint at n_mc={max(nmc_list)}")
 
    # ------------------------------------------------------------------
    # 2. Run for each n_mc
    # ------------------------------------------------------------------
    h_joint    = {}
    h_marginal = {}
 
    for n_mc in nmc_list:
        print(f"\n--- n_mc = {n_mc} ---")
        samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
 
        if method == 'gillespie':
            add_gillespie_reverse(samples, w, mu, beta, T)
        else:
            raise ValueError(f"Method '{method}' not supported in this test yet.")
 
        # Joint Hellinger at t=0
        joint_dists = samples.compute_hellinger_distances(method)
        h_joint[n_mc] = joint_dists[t0]
 
        # Marginal Hellinger(k) at t=0, M=1
        marginal_results = samples.compute_marginal_hellinger(method, k=k, M=1)
        h_marginal[n_mc] = marginal_results[t0]['h']
 
    # ------------------------------------------------------------------
    # 3. Reference value: H_joint at max n_mc
    # ------------------------------------------------------------------
    h_joint_ref = h_joint[max(nmc_list)]
 
    # ------------------------------------------------------------------
    # 4. Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS AT t=0  (reference H_joint = {h_joint_ref:.4f})")
    print("=" * 70)
 
    col_w = 16
    header = (f"{'n_mc':<12}{('H_joint'):>{col_w}}"
              f"{(f'H_marginal(k={k})'):>{col_w}}{'|gap|':>{col_w}}")
    print(header)
    print("-" * len(header))
 
    for n_mc in nmc_list:
        gap = abs(h_marginal[n_mc] - h_joint_ref)
        print(f"{n_mc:<12}{h_joint[n_mc]:>{col_w}.4f}"
              f"{h_marginal[n_mc]:>{col_w}.4f}"
              f"{gap:>{col_w}.4f}")
 
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    nmc_arr      = np.array(nmc_list)
    joint_arr    = np.array([h_joint[n]    for n in nmc_list])
    marginal_arr = np.array([h_marginal[n] for n in nmc_list])
    gap_arr      = np.abs(marginal_arr - h_joint_ref)
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    ax.plot(nmc_arr, joint_arr,    'o-',  color='#0d0887', linewidth=2,
            markersize=6, label='H_joint')
    ax.plot(nmc_arr, marginal_arr, 's-',  color='#cc4778', linewidth=2,
            markersize=6, label=f'H_marginal (s={k})')
    ax.plot(nmc_arr, gap_arr,      '^--', color='#f89540', linewidth=1.5,
            markersize=5, label=f'|H_marginal(s={k}) - H_joint_ref|')

    ax.axhline(y=h_joint_ref, color='#0d0887', linestyle=':', linewidth=1.5,
               alpha=0.6, label=f'H_joint reference (n_mc={max(nmc_list):,})')
 
    ax.set_xlabel('$n_{mc}$', fontsize=12)
    ax.set_ylabel('Hellinger distance', fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    # ax.set_title(f'Approximation Error vs n_mc  —  {method},  t=0,  k={k}', fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    return h_joint, h_marginal, h_joint_ref

# ==============================================================================
# TEST 6: test_marginal_hellinger_vs_M
# ==============================================================================
 
def test_marginal_hellinger_vs_M(
    N, L, r, w, beta, T, n_mc, checkpoint_times,
    k_list,
    M_list,
    method='gillespie',
    mu_seed=0,
    mu=None,
    n_bootstrap=50,
    ci_level=0.95,
    plot_filename='test_marginal_hellinger_vs_M.png',
):
    """
    Show that the average marginal Hellinger H(k, M) does not depend on M.
 
    For each k in k_list and each M in M_list, run n_bootstrap replicates
    where each replicate draws M fresh random subsets of size k and computes
    the average Hellinger over those subsets. The simulation (forward + reverse)
    runs only once; bootstrapping only re-draws subsets from the existing particles.
 
    One subplot per k. Each subplot shows one shaded CI band per M value
    (mean ± CI over bootstrap replicates) vs time. If M is irrelevant, all
    bands overlap.
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    n_mc : int
        Number of Monte Carlo samples.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    k_list : list of int
        Marginal orders to test, e.g. [1, 2, 3].
    M_list : list of int
        Numbers of subsets to average over, e.g. [1, 5, 10, 20].
    method : str
        Reverse method to use, e.g. 'gillespie'.
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    mu : np.ndarray or None
        If provided, use this mu directly (and the given w) instead of
        generating a new problem. Shape (r, N, L).
    n_bootstrap : int
        Number of bootstrap replicates per (k, M) pair.
    ci_level : float
        Confidence level for the CI band, e.g. 0.95.
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    results : dict[(k, M), np.ndarray]  shape (n_bootstrap, n_checkpoints)
        Bootstrap matrix of H(k, M) values for each (k, M) pair.
    """
    print("=" * 70)
    print("TEST: Marginal Hellinger vs M (bootstrapped)")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Problem setup and single simulation
    # ------------------------------------------------------------------
    if mu is not None:
        mu = np.asarray(mu)
        print("Using provided mu and w (no new problem generated).")
    else:
        w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
    n_times = len(checkpoint_times)
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, n_mc={n_mc}, mu_seed={mu_seed}")
    print(f"  method={method}")
    print(f"  k_list={k_list}, M_list={M_list}")
    print(f"  n_bootstrap={n_bootstrap}, ci_level={ci_level}")
 
    samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
 
    if method == 'gillespie':
        add_gillespie_reverse(samples, w, mu, beta, T)
    else:
        raise ValueError(f"Method '{method}' not supported in this test yet.")
 
    # ------------------------------------------------------------------
    # 2. Bootstrap: for each (k, M), draw M subsets n_bootstrap times
    # ------------------------------------------------------------------
    from itertools import combinations
 
    # Pre-build all possible subsets for each k
    all_subsets = {k: list(combinations(range(N), k)) for k in k_list}
 
    # results[(k, M)] = np.ndarray shape (n_bootstrap, n_times)
    results = {}
 
    for k in k_list:
        subsets_k = all_subsets[k]
        n_possible = len(subsets_k)
 
        for M in M_list:
            print(f"\n  Bootstrapping k={k}, M={M}  ({n_bootstrap} replicates)...")
            boot_matrix = np.zeros((n_bootstrap, n_times))
 
            for b in range(n_bootstrap):
                rng_b = np.random.default_rng(b)
 
                # Draw M subsets (with replacement if M > n_possible)
                replace = M > n_possible
                chosen_indices = rng_b.choice(n_possible, size=M, replace=replace)
                chosen_subsets = [subsets_k[i] for i in chosen_indices]
 
                for ti, t in enumerate(checkpoint_times):
                    fwd = samples.forward[float(t)]          # (n_mc, N)
                    rev = samples.reverse_methods[method][float(t)]  # (n_mc, N)
 
                    h_vals = np.zeros(M)
                    for m, subset in enumerate(chosen_subsets):
                        proj_fwd = fwd[:, subset]
                        proj_rev = rev[:, subset]
                        sa, pa = compute_empirical_joint_pmf(proj_fwd, k, None, MASK=-1)
                        sb, pb = compute_empirical_joint_pmf(proj_rev, k, None, MASK=-1)
                        h_vals[m] = compute_hellinger_from_joint_pmfs(sa, pa, sb, pb)
 
                    boot_matrix[b, ti] = h_vals.mean()
 
            results[(k, M)] = boot_matrix
 
    # ------------------------------------------------------------------
    # 3. Plot: one subplot per k, one band per M
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    alpha = 1.0 - ci_level
    n_k = len(k_list)
    fig, axes = plt.subplots(1, n_k, figsize=(6 * n_k, 6), squeeze=False)
 
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(M_list)))
 
    for col, k in enumerate(k_list):
        ax = axes[0, col]
 
        for color, M in zip(colors, M_list):
            boot_matrix = results[(k, M)]           # (n_bootstrap, n_times)
            mean_h = boot_matrix.mean(axis=0)
            ci_lo  = np.quantile(boot_matrix, alpha / 2,       axis=0)
            ci_hi  = np.quantile(boot_matrix, 1.0 - alpha / 2, axis=0)
 
            ax.plot(checkpoint_times, mean_h, '-', color=color, alpha = 0.18,
                    linewidth=2, label=f'M={M}')
            ax.fill_between(checkpoint_times, ci_lo, ci_hi,
                            color=color, alpha=0.15)
 
        ax.set_xlabel('Physical time t', fontsize=11)
        ax.set_ylabel('H(k, M)', fontsize=11)
        ax.set_title(f'k={k}', fontsize=13)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
 
    # fig.suptitle(
        # f'Marginal Hellinger H(k,M) vs time — M independence check\n'
        # f'{method},  n_mc={n_mc},  {int(ci_level*100)}% CI over {n_bootstrap} bootstraps',
        # fontsize=13, y=1.02
    # )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    # ------------------------------------------------------------------
    # 4. Summary table: mean H at t=0 for each (k, M)
    # ------------------------------------------------------------------
    t0 = _closest_time(checkpoint_times, 0.0)
    t0_idx = np.argmin(np.abs(checkpoint_times - t0))
 
    print("\n" + "=" * 70)
    print(f"MEAN H(k, M) AT t=0  ({int(ci_level*100)}% CI)")
    print("=" * 70)
 
    col_w = 20
    header = f"{'(k, M)':<12}" + f"{'mean':>{col_w}}" + f"{'CI_lo':>{col_w}}" + f"{'CI_hi':>{col_w}}"
    print(header)
    print("-" * len(header))
 
    for k in k_list:
        for M in M_list:
            boot_col = results[(k, M)][:, t0_idx]
            mean_h = boot_col.mean()
            lo     = np.quantile(boot_col, alpha / 2)
            hi     = np.quantile(boot_col, 1.0 - alpha / 2)
            print(f"{str((k,M)):<12}{mean_h:>{col_w}.4f}{lo:>{col_w}.4f}{hi:>{col_w}.4f}")
 
    print("=" * 70)
 
    return results

# ==============================================================================
# TEST 7: test_hellinger_vs_tau
# ==============================================================================
 
 
def test_hellinger_vs_tau(
    N, L, r, w, beta, T, n_mc, checkpoint_times,
    tau_list,
    mu_seed=0,
    marginal_hellinger=False,
    k=None,
    plot_filename='test_hellinger_vs_tau.png',
):
    """
    Run tau-leaping reverse process for each tau in tau_list on the same fixed
    problem instance, compute Hellinger distance at every checkpoint, and plot
    all curves together with Gillespie as the exact reference.
 
    As tau -> 0, tau-leaping should converge to the Gillespie result.
 
    Parameters
    ----------
    N : int
        Number of dimensions.
    L : int
        Vocabulary size per dimension.
    r : int
        Number of mixture components.
    w : np.ndarray, shape (r,)
        Mixture weights.
    beta : float
        Forward masking rate.
    T : float
        Terminal time.
    n_mc : int
        Number of Monte Carlo samples.
    checkpoint_times : array-like
        Physical times in [0, T] at which to record samples.
    tau_list : list of float
        Tau-leaping step sizes to compare, e.g. [0.5, 0.2, 0.1, 0.05, 0.01].
    mu_seed : int
        Random seed for generating mu (fixes the problem instance).
    marginal_hellinger : bool
        If False (default), compute joint Hellinger distance.
        If True, compute marginal Hellinger of order k (requires k to be set).
    k : int or None
        Marginal order to use when marginal_hellinger=True.
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    results : dict[float, dict[float, float]]
        results[tau][t] = Hellinger distance at checkpoint t.
    gillespie_distances : dict[float, float]
        Hellinger distances for the Gillespie reference.
    """
    if marginal_hellinger and k is None:
        raise ValueError("k must be specified when marginal_hellinger=True.")
 
    def _get_distances(samp, mname):
        if marginal_hellinger:
            raw = samp.compute_marginal_hellinger(mname, k=k, M=1)
            return {t: raw[t]["h"] for t in raw}
        else:
            return samp.compute_hellinger_distances(mname)
 
    hellinger_label = f"marginal H ($s={k}$)" if marginal_hellinger else "joint H"
 
    print("=" * 70)
    print("TEST: Hellinger Distance vs Tau (tau-leaping, no corrector)")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 1. Problem setup — single forward simulation shared by all tau
    # ------------------------------------------------------------------
    w, mu = make_problem(N, L, r, w, seed=mu_seed)
    checkpoint_times = np.sort(np.unique(np.asarray(checkpoint_times, dtype=float)))
 
    print(f"\nProblem parameters:")
    print(f"  N={N}, L={L}, r={r}, beta={beta}, T={T}, n_mc={n_mc}, mu_seed={mu_seed}")
    print(f"  tau_list={tau_list}")
    print(f"  hellinger mode: {hellinger_label}")
 
    samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
 
    # ------------------------------------------------------------------
    # 2. Gillespie reference
    # ------------------------------------------------------------------
    print("\n--- Gillespie reference ---")
    add_gillespie_reverse(samples, w, mu, beta, T)
    gillespie_distances = _get_distances(samples, "gillespie")
 
    # ------------------------------------------------------------------
    # 3. Tau-leaping for each tau (no corrector)
    # ------------------------------------------------------------------
    results = {}
    for tau in tau_list:
        print(f"\n--- tau = {tau} ---")
        add_tau_leap_reverse(samples, w, mu, beta, T, tau=tau, corrector=False)
        method_name = f"tau_leap_{tau}"
        results[tau] = _get_distances(samples, method_name)
 
    # ------------------------------------------------------------------
    # 4. Table at key checkpoints
    # ------------------------------------------------------------------
    table_targets = [0.0, T / 4, T / 2, 3 * T / 4, T]
    table_times   = [_closest_time(checkpoint_times, tgt) for tgt in table_targets]
    table_labels  = ["t=0", "t=T/4", "t=T/2", "t=3T/4", "t=T"]
 
    print("\n" + "=" * 70)
    print(f"HELLINGER DISTANCES AT KEY CHECKPOINTS  ({hellinger_label})")
    print("=" * 70)
 
    col_w = 12
    header = f"{'method':<20}" + "".join(f"{lbl:>{col_w}}" for lbl in table_labels)
    print(header)
    print("-" * len(header))
 
    row = f"{'Gillespie':<20}"
    for t in table_times:
        row += f"{gillespie_distances[float(t)]:>{col_w}.4f}"
    print(row)
 
    for tau in sorted(tau_list):
        row = f"{f'tau={tau}':<20}"
        for t in table_times:
            row += f"{results[tau][float(t)]:>{col_w}.4f}"
        print(row)
 
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
 
    gill_arr = np.array([gillespie_distances[float(t)] for t in checkpoint_times])
    ax.plot(checkpoint_times, gill_arr, "k--",color="#333333", linewidth=2,
            label="Gillespie", zorder=5)
 
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(tau_list)))
    for color, tau in zip(colors, sorted(tau_list, reverse=True)):
        tau_arr = np.array([results[tau][float(t)] for t in checkpoint_times])
        ax.plot(checkpoint_times, tau_arr, "o-", color=color, linewidth=2,
                markersize=4, label=f"$\\tau={tau}$")
 
    ax.set_xlabel("Physical time t", fontsize=12)
    ax.set_ylabel(f"Hellinger distance  ({hellinger_label})", fontsize=12)
    # ax.set_title(
    #     f"Hellinger Distance vs tau  —  tau-leaping (no corrector)\n"
    #     f"N={N}, L={L}, r={r}, n_mc={n_mc},  {hellinger_label}",
    #     fontsize=13
    # )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    return results, gillespie_distances

# ==============================================================================
# PLOT: plot_vocabulary_distribution
# ==============================================================================
 
def plot_vocabulary_distribution(
    samples,
    indices,
    methods,
    t,
    L,
    plot_filename='vocabulary_distribution.png',
):
    """
    For each index in `indices`, plot the empirical marginal distribution over
    vocabulary values {0, ..., L-1} as side-by-side bars for forward and each
    reverse method.
 
    Parameters
    ----------
    samples : DiffusionSamples
        Samples object containing forward and reverse particles.
    indices : list of int
        Dimension indices to plot, e.g. [0, 3, 7]. One subplot per index.
    methods : list of str
        Reverse method names to include, e.g. ['gillespie', 'tau_leap_0.1'].
        Maximum 3 methods recommended for readability.
    t : float
        Checkpoint time to plot at. Will use the closest available checkpoint.
    L : int
        Vocabulary size (number of possible values per dimension).
    plot_filename : str
        Path to save the output plot.
    """
    # Find closest checkpoint
    times_arr = np.asarray(samples.times)
    t_actual = float(times_arr[np.argmin(np.abs(times_arr - t))])
    print(f"Plotting at t={t_actual:.4f}  (requested t={t})")
 
    # All bars:  methods
    all_labels =  methods
    vocab_values = np.arange(L)
 
    # Colour per label
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(all_labels))]
 
    n_indices = len(indices)
    fig, axes = plt.subplots(1, n_indices, figsize=(5 * n_indices, 5), squeeze=False)
 
    bar_width = 0.8 / len(all_labels)
 
    for col, idx in enumerate(indices):
        ax = axes[0, col]
 
        for b, (label, color) in enumerate(zip(all_labels, colors)):
            # Get particles for this dimension
            if label == 'forward':
                particles = samples.forward[t_actual][:, idx]  # (n_mc,)
            else:
                particles = samples.reverse_methods[label][t_actual][:, idx]
 
            # Empirical distribution over {0, ..., L-1}
            counts = np.array([(particles == v).sum() for v in vocab_values], dtype=float)
            probs  = counts / counts.sum()
 
            offset = (b - len(all_labels) / 2 + 0.5) * bar_width
            ax.bar(vocab_values + offset, probs, width=bar_width,
                   color=color, label=label, alpha=0.85)
 
        ax.set_xlabel('Vocabulary value', fontsize=11)
        ax.set_ylabel('Empirical probability', fontsize=11)
        ax.set_title(f'Dimension {idx}', fontsize=12)
        ax.set_xticks(vocab_values)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
 
    fig.suptitle(f'Vocabulary distribution at t={t_actual:.4f}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()


# ==============================================================================
# PLOT: plot_hellinger_vs_k_three_panels
# ==============================================================================
 
def plot_hellinger_vs_k_three_panels(
    samples,
    k_list,
    tau,
    r,
    t,
    gillespie_method='gillespie',
    plot_filename='hellinger_vs_k_three_panels.png',
):
    """
    Three-panel figure showing the bias-corrected marginal Hellinger vs k.
 
    Panel 1 — Raw curves:
        H_gillespie(k) and H_tau(k) vs k on the same axes.
        The Gillespie curve growing with k reveals the estimation bias.
 
    Panel 2 — Corrected curve:
        H_corrected(k) = H_tau(k) - H_gillespie(k) vs k.
        Vertical dashed line at k=r. Horizontal dashed line at 0.
        Should flatten after k=r.
 
    Panel 3 — Incremental gain:
        Delta H_corrected(k) = H_corrected(k) - H_corrected(k-1) as a bar chart.
        Bars should drop to near zero after k=r.
 
    Parameters
    ----------
    samples : DiffusionSamples
        Samples object with forward and reverse particles.
    k_list : list of int
        Marginal orders to evaluate, e.g. [1, 2, 3, 4, 5].
    tau : float
        Tau value used for the tau-leaping method, e.g. 0.5.
        Used to look up the method name 'tau_leap_{tau}'.
    r : int
        Number of mixture components. Vertical line drawn at k=r.
    t : float
        Checkpoint time to evaluate at. Closest available checkpoint is used.
    gillespie_method : str
        Name of the Gillespie method in samples, default 'gillespie'.
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    h_gill : dict[int, float]
        H_gillespie(k) at the given t for each k.
    h_tau : dict[int, float]
        H_tau(k) at the given t for each k.
    h_corr : dict[int, float]
        H_corrected(k) = H_tau(k) - H_gillespie(k) for each k.
    """
    tau_method = f"tau_leap_{tau}"
 
    # Find closest checkpoint
    times_arr = np.asarray(samples.times)
    t_actual = float(times_arr[np.argmin(np.abs(times_arr - t))])
    print(f"Evaluating at t={t_actual:.4f}  (requested t={t})")
 
    # ------------------------------------------------------------------
    # Compute H_gillespie(k) and H_tau(k) for each k
    # ------------------------------------------------------------------
    h_gill = {}
    h_tau  = {}
 
    for k in k_list:
        gill_raw = samples.compute_marginal_hellinger(gillespie_method, k=k, M=1)
        h_gill[k] = gill_raw[t_actual]['h']
 
        tau_raw = samples.compute_marginal_hellinger(tau_method, k=k, M=1)
        h_tau[k] = tau_raw[t_actual]['h']
 
    # Corrected curve
    h_corr = {k: h_tau[k] - h_gill[k] for k in k_list}
 
    # Incremental gain (needs k_list sorted)
    k_sorted = sorted(k_list)
    delta_h_corr = {}
    for i, k in enumerate(k_sorted):
        if i == 0:
            delta_h_corr[k] = h_corr[k]
        else:
            delta_h_corr[k] = h_corr[k] - h_corr[k_sorted[i - 1]]
 
    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"H(k) AT t={t_actual:.4f}")
    print("=" * 70)
    col_w = 14
    header = f"{'k':<6}{'H_gill(k)':>{col_w}}{'H_tau(k)':>{col_w}}{'H_corr(k)':>{col_w}}{'DeltaH_corr':>{col_w}}"
    print(header)
    print("-" * len(header))
    for k in k_sorted:
        print(f"{k:<6}{h_gill[k]:>{col_w}.4f}{h_tau[k]:>{col_w}.4f}"
              f"{h_corr[k]:>{col_w}.4f}{delta_h_corr[k]:>{col_w}.4f}")
    print("=" * 70)
 
    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
 
    k_arr      = np.array(k_sorted)
    gill_arr   = np.array([h_gill[k]       for k in k_sorted])
    tau_arr    = np.array([h_tau[k]        for k in k_sorted])
    corr_arr   = np.array([h_corr[k]       for k in k_sorted])
    delta_arr  = np.array([delta_h_corr[k] for k in k_sorted])
 
    # --- Panel 1: Raw curves ---
    ax1 = axes[0]
    ax1.plot(k_arr, gill_arr, 'o-', color='royalblue', linewidth=2,
             markersize=6, label=f'H_gillespie(k)')
    ax1.plot(k_arr, tau_arr,  's-', color='tomato',    linewidth=2,
             markersize=6, label=f'H_tau(k)  [tau={tau}]')
    ax1.axvline(x=r, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, label=f'k=r={r}')
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Hellinger distance', fontsize=12)
    ax1.set_title('Panel 1: Raw curves', fontsize=13)
    ax1.set_xticks(k_arr)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
 
    # --- Panel 2: Corrected curve ---
    ax2 = axes[1]
    ax2.plot(k_arr, corr_arr, 'o-', color='forestgreen', linewidth=2, markersize=6)
    ax2.axvline(x=r, color='grey',  linestyle='--', linewidth=1.5, alpha=0.7, label=f'k=r={r}')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.5, label='H=0')
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('H_tau(k) - H_gillespie(k)', fontsize=12)
    ax2.set_title('Panel 2: Corrected curve', fontsize=13)
    ax2.set_xticks(k_arr)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
 
    # --- Panel 3: Incremental gain ---
    ax3 = axes[2]
    bar_colors = ['tomato' if k <= r else 'lightgrey' for k in k_sorted]
    ax3.bar(k_arr, delta_arr, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax3.axvline(x=r, color='grey',  linestyle='--', linewidth=1.5, alpha=0.7, label=f'k=r={r}')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('Delta H_corrected(k)', fontsize=12)
    ax3.set_title('Panel 3: Incremental gain', fontsize=13)
    ax3.set_xticks(k_arr)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
 
    fig.suptitle(
        f'Marginal Hellinger vs k  —  t={t_actual:.4f},  tau={tau},  r={r}',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    return h_gill, h_tau, h_corr



# ==============================================================================
# TEST 8: bootstrap_method_comparison
# ==============================================================================
 
def bootstrap_method_comparison(
    samples,
    L,
    k=4,
    n_methods=4,
    extra_methods=('gillespie', 'tau_leap_0.5'),
    method_filter='informed_corrector',
    n_bootstrap=200,
    ci_level=0.95,
    subset_seed=0,
    method_seed=0,
    plot_filename='bootstrap_method_comparison.png',
):
    """
    Bootstrap the marginal Hellinger distance (order k) for a randomly chosen
    set of corrector methods plus reference methods, to assess whether the
    differences between methods are statistically significant.
 
    For each method and each checkpoint:
      - Resample the n_mc forward particles and the n_mc reverse particles with
        replacement, n_bootstrap times.
      - Recompute the marginal Hellinger (on a single fixed random subset of k
        dimensions) for each resample.
      - Report mean and a (ci_level*100)% confidence interval.
 
    Requires a FULL DiffusionSamples object (raw particles), not lightweight.
 
    Parameters
    ----------
    samples : DiffusionSamples
        Full samples object with raw particles for forward and all methods.
    L : int
        Vocabulary size.
    k : int
        Marginal order (number of dimensions in the subset).
    n_methods : int
        Number of corrector methods to randomly select from those matching
        method_filter.
    extra_methods : tuple of str
        Reference methods always included, e.g. ('gillespie', 'tau_leap_0.5').
    method_filter : str
        Substring that selectable corrector methods must contain.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level, e.g. 0.95.
    subset_seed : int
        Seed for choosing the fixed random subset of k dimensions.
    method_seed : int
        Seed for randomly selecting which corrector methods to include.
    plot_filename : str
        Path to save the output plot.
 
    Returns
    -------
    chosen_methods : list of str
        The full list of methods that were bootstrapped.
    boot_results : dict[str, dict]
        boot_results[method] = {'mean': arr, 'ci_lo': arr, 'ci_hi': arr}
        each array indexed by checkpoint.
    """
    if isinstance(samples, dict):
        raise ValueError("bootstrap_method_comparison requires a FULL "
                         "DiffusionSamples object (raw particles), not a "
                         "lightweight dict.")
 
    print("=" * 70)
    print("TEST: Bootstrap Method Comparison (marginal Hellinger)")
    print("=" * 70)
 
    times      = np.asarray(samples.times)
    n_times    = len(times)
    n_mc       = samples.metadata['n_mc']
    N          = samples.metadata['N']
    alpha      = 1.0 - ci_level
 
    # ------------------------------------------------------------------
    # 1. Select methods
    # ------------------------------------------------------------------
    available = samples.list_methods()
    candidates = [m for m in available if method_filter in m]
 
    m_rng = np.random.default_rng(method_seed)
    if len(candidates) <= n_methods:
        chosen_corr = candidates
    else:
        idx = m_rng.choice(len(candidates), size=n_methods, replace=False)
        chosen_corr = [candidates[i] for i in sorted(idx)]
 
    chosen_methods = list(extra_methods) + chosen_corr
 
    print(f"\nMarginal order k={k},  n_bootstrap={n_bootstrap},  CI={ci_level}")
    print(f"Reference methods: {list(extra_methods)}")
    print(f"Randomly chosen corrector methods ({len(chosen_corr)}):")
    for m in chosen_corr:
        print(f"  - {m}")
 
    # ------------------------------------------------------------------
    # 2. Fixed random subset of k dimensions
    # ------------------------------------------------------------------
    s_rng = np.random.default_rng(subset_seed)
    subset = sorted(s_rng.choice(N, size=k, replace=False).tolist())
    print(f"\nFixed subset of dimensions: {subset}")
 
    # ------------------------------------------------------------------
    # 3. Bootstrap (fast: resample from multinomial counts)
    # ------------------------------------------------------------------
    # Instead of resampling particles and rebuilding the PMF each time
    # (expensive), we count each unique k-dim state ONCE per checkpoint,
    # then bootstrap by drawing multinomial counts. A bootstrap resample of
    # n_mc particles is exactly a Multinomial(n_mc, empirical_probs) draw.
    #
    # Both forward and reverse are encoded onto a common set of vocab codes
    # so the two probability vectors are aligned index-by-index, making the
    # Hellinger distance a simple vectorized formula.
 
    def _encode(arr):
        """Encode (n, k) integer states into a single integer per row (base L)."""
        codes = np.zeros(arr.shape[0], dtype=np.int64)
        for j in range(arr.shape[1]):
            codes = codes * L + arr[:, j].astype(np.int64)
        return codes
 
    boot_results = {}
    boot_rng = np.random.default_rng(method_seed + 12345)
 
    for method in chosen_methods:
        print(f"\nBootstrapping '{method}'...")
        boot_matrix = np.zeros((n_bootstrap, n_times))
 
        for ti, t in enumerate(times):
            fwd = samples.forward[float(t)][:, subset]                  # (n_mc, k)
            rev = samples.reverse_methods[method][float(t)][:, subset]  # (n_mc, k)
 
            # Encode to integer codes and align onto a shared support
            fwd_codes = _encode(fwd)
            rev_codes = _encode(rev)
            all_codes = np.concatenate([fwd_codes, rev_codes])
            uniq, inv = np.unique(all_codes, return_inverse=True)
            S = len(uniq)
 
            inv_f = inv[:len(fwd_codes)]
            inv_r = inv[len(fwd_codes):]
 
            # Empirical counts over the shared support
            count_f = np.bincount(inv_f, minlength=S).astype(float)
            count_r = np.bincount(inv_r, minlength=S).astype(float)
            p_f = count_f / count_f.sum()
            p_r = count_r / count_r.sum()
 
            # Bootstrap: each resample is a multinomial draw of n_mc particles
            boot_f = boot_rng.multinomial(n_mc, p_f, size=n_bootstrap) / n_mc  # (B, S)
            boot_r = boot_rng.multinomial(n_mc, p_r, size=n_bootstrap) / n_mc  # (B, S)
 
            # Vectorized Hellinger across all bootstraps at once
            # H = sqrt( 0.5 * sum_s (sqrt(p_s) - sqrt(q_s))^2 )
            diff = np.sqrt(boot_f) - np.sqrt(boot_r)
            boot_matrix[:, ti] = np.sqrt(0.5 * np.sum(diff * diff, axis=1))
 
        mean_h = boot_matrix.mean(axis=0)
        ci_lo  = np.quantile(boot_matrix, alpha / 2,       axis=0)
        ci_hi  = np.quantile(boot_matrix, 1.0 - alpha / 2, axis=0)
        boot_results[method] = {'mean': mean_h, 'ci_lo': ci_lo, 'ci_hi': ci_hi}
 
    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
 
    colors = plt.cm.tab10(np.linspace(0, 1, len(chosen_methods)))
 
    for color, method in zip(colors, chosen_methods):
        res = boot_results[method]
        # Shorten long corrector names for the legend
        label = method if len(method) < 40 else method[:37] + '...'
        ax.plot(times, res['mean'], '-', color=color, linewidth=2, label=label)
        ax.fill_between(times, res['ci_lo'], res['ci_hi'], color=color, alpha=0.15)
 
    ax.set_xlabel('Physical time t', fontsize=12)
    ax.set_ylabel(f'Marginal Hellinger (k={k})', fontsize=12)
    ax.set_title(
        f'Bootstrap comparison — {int(ci_level*100)}% CI over {n_bootstrap} resamples',
        fontsize=13
    )
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()
 
    # ------------------------------------------------------------------
    # 5. Table at t=0
    # ------------------------------------------------------------------
    t0_idx = int(np.argmin(np.abs(times - 0.0)))
 
    print("\n" + "=" * 70)
    print(f"MARGINAL HELLINGER AT t={times[t0_idx]:.4f}  ({int(ci_level*100)}% CI)")
    print("=" * 70)
 
    col_w = 14
    header = f"{'method':<50}{'mean':>{col_w}}{'CI_lo':>{col_w}}{'CI_hi':>{col_w}}"
    print(header)
    print("-" * len(header))
 
    for method in chosen_methods:
        res = boot_results[method]
        short = method if len(method) < 48 else method[:45] + '...'
        print(f"{short:<50}{res['mean'][t0_idx]:>{col_w}.4f}"
              f"{res['ci_lo'][t0_idx]:>{col_w}.4f}{res['ci_hi'][t0_idx]:>{col_w}.4f}")
 
    print("=" * 70)
 
    return chosen_methods, boot_results


"""


Function to plot marginal Hellinger distance vs NFE budget
for different tau values and optionally corrector configurations.

Bootstrap confidence intervals are computed using the multinomial
resampling trick (efficient, no need to re-run the diffusion process).
"""

# =============================================================================
# Internal helpers
# =============================================================================

def _encode(arr):
    n, k = arr.shape
    L    = int(arr.max()) + 2
    base = L ** np.arange(k - 1, -1, -1)
    return arr @ base


def _bootstrap_hellinger(fwd_particles, rev_particles, n_bootstrap, alpha, rng):
    n_mc = len(fwd_particles)

    fwd_codes = _encode(fwd_particles)
    rev_codes = _encode(rev_particles)

    all_codes        = np.concatenate([fwd_codes, rev_codes])
    uniq, inv        = np.unique(all_codes, return_inverse=True)
    S                = len(uniq)
    inv_f, inv_r     = inv[:n_mc], inv[n_mc:]

    count_f = np.bincount(inv_f, minlength=S).astype(float)
    count_r = np.bincount(inv_r, minlength=S).astype(float)
    p_f     = count_f / count_f.sum()
    p_r     = count_r / count_r.sum()

    boot_f  = rng.multinomial(n_mc, p_f, size=n_bootstrap) / n_mc
    boot_r  = rng.multinomial(n_mc, p_r, size=n_bootstrap) / n_mc
    diff    = np.sqrt(boot_f) - np.sqrt(boot_r)
    boot_h  = np.sqrt(0.5 * np.sum(diff * diff, axis=1))

    return (float(boot_h.mean()),
            float(np.quantile(boot_h, alpha / 2)),
            float(np.quantile(boot_h, 1.0 - alpha / 2)))


def _infer_method_type(method_name):
    """Infer curve grouping type from method name."""
    if method_name == 'gillespie':
        return 'gillespie'
    if 'informed_corrector' in method_name:
        return 'informed_corrector'
    if 'random_masking' in method_name:
        return 'random_masking'
    if 'PRISM' in method_name:
        return 'PRISM'
    if 'DPC' in method_name:
        return 'DPC'
    if 'tau_leap' in method_name:
        return 'tau_leap'
    return 'unknown'


def _compute_and_record(samples, method_name, curve_label,
                        t_actual, subset, n_bootstrap, alpha, rng, results):
    """Compute bootstrap Hellinger for one method and append to results."""
    nfe = samples.nfe.get(method_name, None)
    if nfe is None:
        print(f"  WARNING: NFE not found for {method_name}, skipping.")
        return

    fwd = samples.forward[t_actual][:, list(subset)]
    rev = samples.reverse_methods[method_name][t_actual][:, list(subset)]

    mean_h, ci_lo, ci_hi = _bootstrap_hellinger(
        fwd, rev, n_bootstrap, alpha, rng
    )

    print(f"  NFE={nfe:4d}  H={mean_h:.4f}  "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  curve='{curve_label}'")

    results.append({
        'curve_label': curve_label,   # legend entry + curve grouping
        'method_name': method_name,   # full method name for table
        'nfe':         nfe,
        'mean_h':      mean_h,
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
    })


def _assign_colors_and_markers(curve_labels):
    """
    Assign a color and marker to each unique curve label.

    Color philosophy:
    - Tau-leap: cool slate blues (baseline family)
    - PRISM:    warm amber/orange family
    - Informed: teal/jade family
    - Random:   violet/purple family
    - DPC:      rose/coral family
    - Gillespie: near-black charcoal (reference)

    Within each family, variants are spaced across a perceptually
    uniform range so individual curves are distinguishable.
    """
    # ── Single-entry fixed colors (exact label match) ──────────────────────
    exact_colors = {
        'gillespie':          '#1a1a2e',   # deep charcoal
        'Gillespie':          '#1a1a2e',
        'tau_leap':           '#4361ee',   # vivid blue
        'Tau-leap':           '#4361ee',
    }
    exact_markers = {
        'gillespie':  '*',
        'Gillespie':  '*',
        'tau_leap':   'o',
        'Tau-leap':   'o',
    }

    # ── Per-family colour sequences (perceptually spaced) ──────────────────
    # Tau-leap family  — slate blues
    tau_seq = [
        '#4361ee', '#3a86ff', '#4cc9f0', '#2d6a9f', '#023e8a',
        '#0077b6', '#48cae4', '#0096c7', '#00b4d8', '#90e0ef',
    ]
    # PRISM family  — amber → deep orange
    prism_seq = [
        '#f77f00', '#fcbf49', '#e76f51', '#d62828', '#f4a261',
        '#e9c46a', '#fb8500', '#ffb703', '#e63946', '#c77dff',
    ]
    # Informed corrector  — teal → jade
    ic_seq = [
        '#2a9d8f', '#52b788', '#57cc99', '#38a3a5', '#22577a',
        '#80b918', '#55a630', '#007f5f', '#1b4332', '#40916c',
    ]
    # Random masking  — violet → plum
    rm_seq = [
        '#7b2d8b', '#9b5de5', '#c77dff', '#e0aaff', '#7209b7',
        '#560bad', '#480ca8', '#3a0ca3', '#3f37c9', '#b5179e',
    ]
    # DPC  — rose → coral
    dpc_seq = [
        '#e63946', '#ff4d6d', '#ff758f', '#c9184a', '#ff0054',
        '#ef233c', '#d90429', '#ff6b6b', '#ee9b00', '#ca6702',
    ]

    tau_idx = prism_idx = ic_idx = rm_idx = dpc_idx = 0

    color_map  = {}
    marker_map = {}

    for label in curve_labels:
        if label in exact_colors:
            color_map[label]  = exact_colors[label]
            marker_map[label] = exact_markers.get(label, 'o')

        elif 'PRISM' in label:
            color_map[label]  = prism_seq[prism_idx % len(prism_seq)]
            marker_map[label] = 's'
            prism_idx += 1

        elif 'informed' in label.lower() or label.upper().startswith('IC'):
            color_map[label]  = ic_seq[ic_idx % len(ic_seq)]
            marker_map[label] = '^'
            ic_idx += 1

        elif 'random' in label.lower() or label.upper().startswith('RM'):
            color_map[label]  = rm_seq[rm_idx % len(rm_seq)]
            marker_map[label] = 'D'
            rm_idx += 1

        elif 'DPC' in label or 'dpc' in label:
            color_map[label]  = dpc_seq[dpc_idx % len(dpc_seq)]
            marker_map[label] = 'P'
            dpc_idx += 1

        elif 'tau' in label.lower():
            color_map[label]  = tau_seq[tau_idx % len(tau_seq)]
            marker_map[label] = 'o'
            tau_idx += 1

        else:
            color_map[label]  = '#adb5bd'   # neutral gray for unknowns
            marker_map[label] = 'x'

    return color_map, marker_map


def _make_plot(results, k, t_eval, title_extra='', alpha=0.05,
               n_bootstrap=None, filename=None, figsize=(10, 6)):
    """
    Shared plotting logic used by both plot_nfe_vs_hellinger
    and plot_from_samples.
    """
    results_sorted = sorted(results, key=lambda r: r['nfe'])

    # Group by curve_label
    by_curve = defaultdict(list)
    for r in results_sorted:
        by_curve[r['curve_label']].append(r)

    unique_labels = list(by_curve.keys())
    color_map, marker_map = _assign_colors_and_markers(unique_labels)

    # ── Style setup ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=figsize, facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    # --- Non-Gillespie curves ---
    for curve_label, group in by_curve.items():
        if curve_label in ('gillespie', 'Gillespie'):
            continue

        color  = color_map[curve_label]
        marker = marker_map[curve_label]

        nfes   = [r['nfe']    for r in group]
        means  = [r['mean_h'] for r in group]
        ci_los = [r['ci_lo']  for r in group]
        ci_his = [r['ci_hi']  for r in group]

        # Main line
        ax.plot(nfes, means,
                color=color, marker=marker,
                linewidth=2.2, markersize=8,
                markeredgecolor='white', markeredgewidth=0.8,
                label=curve_label, zorder=3, solid_capstyle='round')

        # CI shading
        ax.fill_between(nfes, ci_los, ci_his,
                        color=color, alpha=0.12, zorder=2)

        # CI whiskers
        for nfe, lo, hi in zip(nfes, ci_los, ci_his):
            ax.plot([nfe, nfe], [lo, hi],
                    color=color, linewidth=1.2, alpha=0.5, zorder=4)

    # --- Gillespie horizontal reference ---
    gillespie_result = next(
        (r for r in results if r['curve_label'] in ('gillespie', 'Gillespie')),
        None
    )
    non_gil = [r for r in results if r['curve_label'] not in ('gillespie','Gillespie')]
    if gillespie_result and non_gil:
        nfe_min = min(r['nfe'] for r in non_gil) - 0.5
        nfe_max = max(r['nfe'] for r in non_gil) + 0.5
        ax.axhline(y=gillespie_result['mean_h'],
                   color='#1a1a2e', linestyle=(0, (5, 3)),
                   linewidth=2.0, alpha=0.85,
                   label='Gillespie (reference)', zorder=5)
        ax.fill_between(
            [nfe_min, nfe_max],
            gillespie_result['ci_lo'],
            gillespie_result['ci_hi'],
            color='#1a1a2e', alpha=0.06, zorder=1
        )

    ci_pct = int((1 - alpha) * 100)
    bs_str = f", B={n_bootstrap}" if n_bootstrap else ""

    ax.set_xlabel("NFE (Number of Function Evaluations)",
                  fontsize=12, labelpad=8, color='#333333')
    ax.set_ylabel(f"Marginal Hellinger  H  (k={k})",
                  fontsize=12, labelpad=8, color='#333333')
    ax.set_title(
        f"Hellinger vs NFE{title_extra}\n"
        f"k={k},  t={t_eval},  Bootstrap {ci_pct}% CI{bs_str}",
        fontsize=11, pad=12, color='#111111'
    )

    ax.tick_params(colors='#555555', labelsize=10)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    ax.legend(fontsize=10, framealpha=0.9, edgecolor='#dddddd',
              fancybox=False, labelspacing=0.4)
    ax.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.axhline(y=0, color='#bbbbbb', linestyle=':', linewidth=0.8)

    plt.tight_layout(pad=1.5)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    plt.show()

    # --- Summary table: Hellinger values ---
    print(f"\n{'=' * 80}")
    print(f"HELLINGER SUMMARY — t={t_eval}, k={k}")
    print(f"{'=' * 80}")
    print(f"{'Curve':<25} {'NFE':>6} {'H mean':>10} {'CI lo':>10} {'CI hi':>10}")
    print("-" * 80)
    for r in results_sorted:
        if r['curve_label'] in ('gillespie', 'Gillespie'):
            continue
        print(f"{r['curve_label']:<25} {r['nfe']:>6} "
              f"{r['mean_h']:>10.6f} {r['ci_lo']:>10.6f} {r['ci_hi']:>10.6f}")
    if gillespie_result:
        print(f"\n{'Gillespie (ref)':<25} {gillespie_result['nfe']:>6} "
              f"{gillespie_result['mean_h']:>10.6f} "
              f"{gillespie_result['ci_lo']:>10.6f} "
              f"{gillespie_result['ci_hi']:>10.6f}")
    print("=" * 80)

    # --- Detail table: full method names at each NFE ---
    print(f"\n{'=' * 80}")
    print(f"METHOD DETAILS — full method name at each NFE")
    print(f"{'=' * 80}")
    print(f"{'Curve':<25} {'NFE':>6}  {'Method name'}")
    print("-" * 80)
    for r in results_sorted:
        print(f"{r['curve_label']:<25} {r['nfe']:>6}  {r['method_name']}")
    print("=" * 80)

    return fig, ax


# =============================================================================
# Main function: run + plot
# =============================================================================

def plot_nfe_vs_hellinger(
    w,
    mu,
    beta,
    T,
    k,
    t_eval,
    tau_values=None,
    corrector_configs=None,
    samples=None,
    n_mc=None,
    checkpoint_times=None,
    include_gillespie=True,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
    filename=None,
    figsize=(10, 6),
):
    """
    Run methods and plot marginal Hellinger vs NFE with bootstrap CIs.

    Parameters
    ----------
    w, mu, beta, T : mixture parameters
    k : int — marginal order
    t_eval : float — time point to evaluate at
    tau_values : list of float — tau-leaping step sizes (no corrector)
    corrector_configs : list of dict — each with keys:
        'tau', 'corrector_method', 'corrector_start',
        'corrector_hyperparameters', 'label' (curve name for legend)
    samples : DiffusionSamples or None
        If None, runs forward + Gillespie from scratch.
    n_mc, checkpoint_times : required if samples is None
    include_gillespie : bool
    n_bootstrap, alpha, seed, filename, figsize : as usual

    Returns
    -------
    results : list of dict
    samples : DiffusionSamples
    """
    rng = np.random.default_rng(seed)
    N   = mu.shape[1]

    # --- Forward + Gillespie ---
    if samples is None:
        if n_mc is None or checkpoint_times is None:
            raise ValueError("n_mc and checkpoint_times required when samples=None")
        print("=" * 60)
        print("Running forward process...")
        samples = run_diffusion_experiment(w, mu, beta, T, n_mc, checkpoint_times)
        if include_gillespie:
            print("Running Gillespie...")
            add_gillespie_reverse(samples, w, mu, beta, T)
    else:
        print("=" * 60)
        print(f"Reusing existing samples. Methods: {samples.list_methods()}")

    # --- Fixed subset ---
    all_subsets = list(combinations(range(N), k))
    subset      = all_subsets[rng.integers(0, len(all_subsets))]
    print(f"Fixed subset: {subset}")

    # --- Closest checkpoint ---
    times    = samples.times
    t_actual = float(times[int(np.argmin(np.abs(times - t_eval)))])
    print(f"t_eval={t_eval} → checkpoint t={t_actual:.6f}\n")

    results = []

    # --- Gillespie ---
    if 'gillespie' in samples.reverse_methods:
        print("Gillespie bootstrap...")
        _compute_and_record(samples, 'gillespie', 'gillespie',
                            t_actual, subset, n_bootstrap, alpha, rng, results)

    # --- Tau-leaping ---
    if tau_values:
        print(f"\nTau-leaping ({len(tau_values)} values)...")
        for tau in tau_values:
            method_name = f"tau_leap_{tau}"
            if method_name not in samples.reverse_methods:
                add_tau_leap_reverse(samples, w, mu, beta, T,
                                     tau=tau, corrector=False)
            _compute_and_record(samples, method_name, f"tau_leap",
                                t_actual, subset, n_bootstrap, alpha, rng, results)

    # --- Correctors ---
    if corrector_configs:
        print(f"\nCorrectors ({len(corrector_configs)} configs)...")
        for cfg in corrector_configs:
            tau               = cfg['tau']
            corrector_method  = cfg['corrector_method']
            corrector_start   = cfg['corrector_start']
            corrector_hparams = cfg['corrector_hyperparameters']
            curve_label       = cfg.get('label', f"{corrector_method} tau={tau}")

            methods_before = set(samples.nfe.keys())
            add_tau_leap_reverse(
                samples, w, mu, beta, T,
                tau=tau,
                corrector=True,
                corrector_method=corrector_method,
                corrector_start=corrector_start,
                corrector_hyperparameters=corrector_hparams
            )
            methods_after = set(samples.nfe.keys())
            new_methods   = methods_after - methods_before

            if len(new_methods) == 0:
                # Method already existed — find it by most recent nfe key
                method_name = list(samples.nfe.keys())[-1]
            else:
                method_name = new_methods.pop()

            _compute_and_record(samples, method_name, curve_label,
                                t_actual, subset, n_bootstrap, alpha, rng, results)

    # --- Plot ---
    N_val = mu.shape[1]; L_val = mu.shape[2]
    title_extra = f" — N={N_val}, L={L_val}, r={len(w)}, β={beta}, T={T}"
    fig, ax = _make_plot(results, k, t_eval,
                         title_extra=title_extra,
                         alpha=alpha, n_bootstrap=n_bootstrap,
                         filename=filename, figsize=figsize)

    return results, samples


# =============================================================================
# Plot from existing samples (no re-running)
# =============================================================================

def plot_from_samples(
    samples,
    k,
    t_eval,
    methods=None,
    curve_labels=None,   # optional dict: method_name -> curve_label
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
    filename=None,
    figsize=(10, 6),
):
    """
    Plot NFE vs Hellinger from a DiffusionSamples object.

    Can be called after reloading samples — no re-running needed.

    Parameters
    ----------
    samples : DiffusionSamples
    k : int
    t_eval : float
    methods : list of str, optional — which methods to plot (None = all)
    curve_labels : dict, optional
        Maps method_name -> curve_label for legend grouping.
        E.g. {'tau_leap_0.4_corrector_PRISM_start_2.0_eta_0.5': 'PRISM eta=0.5'}
        If None, _infer_method_type is used to group automatically.
    n_bootstrap, alpha, seed, filename, figsize : as usual

    Returns
    -------
    results : list of dict
    fig, ax : Figure and Axes

    Example
    -------
    # Group PRISM variants by eta:
    labels = {m: f"PRISM eta={re.search(r'eta_([0-9.]+)', m).group(1)}"
              for m in samples.list_methods() if 'PRISM' in m}
    results, fig, ax = plot_from_samples(samples, k=5, t_eval=0.0,
                                         curve_labels=labels)
    """
    rng = np.random.default_rng(seed)
    N   = samples.metadata['N']

    if methods is None:
        methods = samples.list_methods()

    available = set(samples.reverse_methods.keys())
    missing   = [m for m in methods if m not in available]
    if missing:
        raise ValueError(f"Methods not found: {missing}")

    # Fixed subset
    all_subsets = list(combinations(range(N), k))
    subset      = all_subsets[rng.integers(0, len(all_subsets))]
    print(f"Fixed subset: {subset}")

    times    = samples.times
    t_actual = float(times[int(np.argmin(np.abs(times - t_eval)))])
    print(f"t_eval={t_eval} → checkpoint t={t_actual:.6f}\n")

    results = []

    for method_name in methods:
        # Determine curve label
        if curve_labels and method_name in curve_labels:
            curve_label = curve_labels[method_name]
        else:
            curve_label = _infer_method_type(method_name)

        _compute_and_record(samples, method_name, curve_label,
                            t_actual, subset, n_bootstrap, alpha, rng, results)

    if not results:
        raise ValueError("No results computed.")

    N_val = samples.metadata['N']
    L_val = samples.metadata['L']
    nmc   = samples.metadata['n_mc']
    title_extra = f" — N={N_val}, L={L_val}, n_mc={nmc:,}"

    fig, ax = _make_plot(results, k, t_eval,
                         title_extra=title_extra,
                         alpha=alpha, n_bootstrap=n_bootstrap,
                         filename=filename, figsize=figsize)

    return results, fig, ax




def plot_method_comparison(samples, methods=None, filename='diffusion_comparison_all_methods.png',
                          figsize=(12, 6), show_annotations=True, return_distances=False,
                          time_start=None, time_end=None,
                          marginal_hellinger=False, k=None, subset_seed=0):
    """
    Plot Hellinger distance comparison for selected reverse methods.

    Works with both full DiffusionSamples objects and lightweight PMF dicts.

    Parameters
    ----------
    samples : DiffusionSamples or dict
        Container with forward and reverse samples (full)
        OR dict with forward/reverse PMFs (lightweight)
    methods : list of str or None
        List of method names to plot. If None, plots all available methods.
    filename : str
        Output filename for the plot
    figsize : tuple
        Figure size (width, height)
    show_annotations : bool
        Whether to show t=0 and t=T annotations
    return_distances : bool
        Whether to return the distances dictionary
    time_start : float or None
        Start time for plot x-axis
    time_end : float or None
        End time for plot x-axis
    marginal_hellinger : bool
        If False (default), compute joint Hellinger distance.
        If True, compute marginal Hellinger of order k (requires k to be set).
    k : int or None
        Marginal order to use when marginal_hellinger=True.
    subset_seed : int
        Seed for choosing the random subset of k dimensions (M=1).

    Returns
    -------
    all_distances : dict (only if return_distances=True)
        Dictionary of Hellinger distances for each method
    """
    if marginal_hellinger and k is None:
        raise ValueError("k must be specified when marginal_hellinger=True.")

    # Detect if lightweight or full
    is_lightweight = isinstance(samples, dict) and samples.get('lightweight', False)

    if is_lightweight:
        # Lightweight format with PMFs
        times = samples['times']
        available_methods = list(samples['reverse_pmfs'].keys())

        if methods is None:
            methods = available_methods
        else:
            invalid = [m for m in methods if m not in available_methods]
            if invalid:
                raise ValueError(f"Methods not found: {invalid}. Available: {available_methods}")

        # Determine N from the first stored forward PMF
        first_t = float(times[0])
        N = samples['forward_pmfs'][first_t]['states'].shape[1]

        # Choose one random subset of size k (shared across all methods/times)
        if marginal_hellinger:
            rng = np.random.default_rng(subset_seed)
            subset = sorted(rng.choice(N, size=k, replace=False).tolist())
            print(f"Marginal Hellinger (k={k}), subset of dimensions: {subset}")

        # Compute Hellinger distances from joint PMFs
        print("Computing Hellinger distances from joint PMFs...")
        all_distances = {}
        for method in available_methods:
            all_distances[method] = {}
            for t in times:
                forward_pmf = samples['forward_pmfs'][float(t)]
                reverse_pmf = samples['reverse_pmfs'][method][float(t)]

                if marginal_hellinger:
                    f_states, f_probs = marginalize_joint_pmf(
                        forward_pmf['states'], forward_pmf['probs'], subset)
                    r_states, r_probs = marginalize_joint_pmf(
                        reverse_pmf['states'], reverse_pmf['probs'], subset)
                else:
                    f_states, f_probs = forward_pmf['states'], forward_pmf['probs']
                    r_states, r_probs = reverse_pmf['states'], reverse_pmf['probs']

                all_distances[method][float(t)] = compute_hellinger_from_joint_pmfs(
                    f_states, f_probs, r_states, r_probs
                )

    else:
        # Full DiffusionSamples object
        times = samples.times
        available_methods = samples.list_methods()

        if methods is None:
            methods = available_methods
        else:
            invalid = [m for m in methods if m not in available_methods]
            if invalid:
                raise ValueError(f"Methods not found: {invalid}. Available: {available_methods}")

        # Compute distances for all methods
        print("Computing Hellinger distances from particles...")
        if marginal_hellinger:
            all_distances = {}
            for method in available_methods:
                raw = samples.compute_marginal_hellinger(method, k=k, M=1, seed=subset_seed)
                all_distances[method] = {t: raw[t]['h'] for t in raw}
        else:
            all_distances = samples.compute_all_hellinger_distances()

    # Filter to only selected methods
    selected_distances = {m: all_distances[m] for m in methods}

    # Filter times based on time_start and time_end
    if time_start is None:
        time_start = times[0]
    if time_end is None:
        time_end = times[-1]

    # Get times within the specified range
    time_mask = (times >= time_start) & (times <= time_end)
    plot_times = times[time_mask]

    if len(plot_times) == 0:
        raise ValueError(f"No checkpoints in range [{time_start}, {time_end}]")

    # Plot comparisons
    hellinger_label = f'marginal H (k={k})' if marginal_hellinger else 'joint H'
    print(f"\nGenerating plot for methods: {methods}")
    print(f"Time range: [{time_start:.3f}, {time_end:.3f}]  ({hellinger_label})")
    plt.figure(figsize=figsize)

    for method_name in methods:
        distances = selected_distances[method_name]
        # Only plot times in the specified range
        plot_distances = [distances[float(t)] for t in plot_times]
        plt.plot(plot_times, plot_distances,
                 'o-', label=method_name, linewidth=2, markersize=4)

    plt.xlabel('Physical time t', fontsize=12)
    plt.ylabel(f'Hellinger distance  ({hellinger_label})', fontsize=12)
    plt.title('Forward vs Reverse Process Comparison\nComparing Different Methods', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Set x-axis limits
    plt.xlim(time_start, time_end)

    # Annotate key points for each method
    if show_annotations:
        t0 = plot_times[0]   # First time in plot range
        tT = plot_times[-1]  # Last time in plot range

        for i, method_name in enumerate(methods):
            distances = selected_distances[method_name]

            # Only annotate if these times are in the plot range
            if float(t0) in distances:
                # Annotate start of range
                y_offset = 0.05 + i * 0.03  # Stagger annotations if multiple methods
                plt.annotate(f'{method_name} t={t0:.2f}: H={distances[float(t0)]:.3f}',
                             xy=(t0, distances[float(t0)]),
                             xytext=(time_start + 0.1 * (time_end - time_start), distances[float(t0)] + y_offset),
                             arrowprops=dict(arrowstyle='->', alpha=0.7),
                             fontsize=9)

            if float(tT) in distances:
                # Annotate end of range
                plt.annotate(f'{method_name} t={tT:.2f}: H={distances[float(tT)]:.3f}',
                             xy=(tT, distances[float(tT)]),
                             xytext=(time_start + 0.8 * (time_end - time_start), distances[float(tT)] + y_offset),
                             arrowprops=dict(arrowstyle='->', alpha=0.7),
                             fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.show()

    if return_distances:
        return all_distances
    
# =============================================================================
# Clean interface: dict of {line_name -> [method_names]}
# =============================================================================

def plot_nfe_from_dict(
    samples,
    lines,
    k,
    t_eval,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
    filename=None,
    figsize=(10, 6),
):
    """
    Plot marginal Hellinger vs NFE where each line is defined explicitly.

    Parameters
    ----------
    samples : DiffusionSamples
        Full samples object.
    lines : dict  {str -> list of str}
        Keys   = line name (appears in legend)
        Values = list of method names that form that line.
                 Each method becomes one point (x=NFE, y=Hellinger).
        Special key 'gillespie' (or any value containing 'gillespie')
        is plotted as a horizontal dashed reference line.

        Example
        -------
        lines = {
            'tau=2':        ['tau_leap_2'],
            'tau=0.6':      ['tau_leap_0.6'],
            'PRISM eta=1':  [
                'tau_leap_0.6_corrector_PRISM_start_0.5_eta_1',
                'tau_leap_0.6_corrector_PRISM_start_1.0_eta_1',
                'tau_leap_0.6_corrector_PRISM_start_2.0_eta_1',
            ],
            'gillespie':    ['gillespie'],
        }

    k : int
        Marginal order for Hellinger distance.
    t_eval : float
        Time point to evaluate at.
    n_bootstrap : int
        Bootstrap resamples for CI.
    alpha : float
        CI significance level (0.05 → 95% CI).
    seed : int, optional
    filename : str, optional
    figsize : tuple

    Returns
    -------
    results : list of dict
        One entry per method with keys:
        line_name, method_name, nfe, mean_h, ci_lo, ci_hi
    fig, ax : Figure and Axes
    """
    rng = np.random.default_rng(seed)
    N   = samples.metadata['N']

    # ── Validate all methods exist ────────────────────────────────────────
    all_requested = [m for methods in lines.values() for m in methods]
    available     = set(samples.reverse_methods.keys())
    missing       = [m for m in all_requested if m not in available]
    if missing:
        raise ValueError(
            f"Methods not found in samples:\n" +
            "\n".join(f"  - {m}" for m in missing) +
            f"\n\nAvailable:\n" +
            "\n".join(f"  - {m}" for m in sorted(available))
        )

    # ── Fixed random subset of k dimensions ──────────────────────────────
    all_subsets = list(combinations(range(N), k))
    subset      = all_subsets[rng.integers(0, len(all_subsets))]
    print(f"Fixed subset of {k} dimensions: {subset}")

    # ── Closest checkpoint to t_eval ─────────────────────────────────────
    times    = samples.times
    t_actual = float(times[int(np.argmin(np.abs(times - t_eval)))])
    print(f"t_eval={t_eval} → checkpoint t={t_actual:.6f}\n")

    fwd_particles_full = samples.forward[t_actual][:, list(subset)]

    # ── Compute bootstrap Hellinger for every method ──────────────────────
    results = []

    for line_name, method_list in lines.items():
        print(f"Line '{line_name}':")
        for method_name in method_list:
            nfe = samples.nfe.get(method_name, None)
            if nfe is None:
                print(f"  WARNING: NFE not found for '{method_name}', skipping.")
                continue

            rev = samples.reverse_methods[method_name][t_actual][:, list(subset)]

            mean_h, ci_lo, ci_hi = _bootstrap_hellinger(
                fwd_particles_full, rev, n_bootstrap, alpha, rng
            )

            print(f"  NFE={nfe:4d}  H={mean_h:.4f}  "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}]  ({method_name})")

            results.append({
                'line_name':   line_name,
                'method_name': method_name,
                'nfe':         nfe,
                'mean_h':      mean_h,
                'ci_lo':       ci_lo,
                'ci_hi':       ci_hi,
            })

    if not results:
        raise ValueError("No results computed — check method names and NFE dict.")

    # ── Assign colors and markers ─────────────────────────────────────────
    unique_lines  = list(lines.keys())
    color_map, marker_map = _assign_colors_and_markers(unique_lines)

    # ── Plot ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=figsize, facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    # Separate gillespie from other lines
    gil_lines  = [ln for ln in unique_lines if 'gillespie' in ln.lower()]
    plot_lines = [ln for ln in unique_lines if 'gillespie' not in ln.lower()]

    # ── Non-gillespie curves ──────────────────────────────────────────────
    for line_name in plot_lines:
        line_results = sorted(
            [r for r in results if r['line_name'] == line_name],
            key=lambda r: r['nfe']
        )
        if not line_results:
            continue

        color  = color_map[line_name]
        marker = marker_map[line_name]

        nfes   = [r['nfe']    for r in line_results]
        means  = [r['mean_h'] for r in line_results]
        ci_los = [r['ci_lo']  for r in line_results]
        ci_his = [r['ci_hi']  for r in line_results]

        ax.plot(nfes, means,
                color=color, marker=marker,
                linewidth=2.2, markersize=8,
                markeredgecolor='white', markeredgewidth=0.8,
                label=line_name, zorder=3,
                solid_capstyle='round')

        ax.fill_between(nfes, ci_los, ci_his,
                        color=color, alpha=0.12, zorder=2)

        for nfe, lo, hi in zip(nfes, ci_los, ci_his):
            ax.plot([nfe, nfe], [lo, hi],
                    color=color, linewidth=1.2, alpha=0.5, zorder=4)

    # ── Gillespie horizontal reference ────────────────────────────────────
    all_nfes = [r['nfe'] for r in results
                if 'gillespie' not in r['line_name'].lower()]

    for gil_line in gil_lines:
        gil_results = [r for r in results if r['line_name'] == gil_line]
        if not gil_results:
            continue
        gil_r = gil_results[0]   # one point expected

        if all_nfes:
            x_lo = min(all_nfes) - 0.3
            x_hi = max(all_nfes) + 0.3
        else:
            x_lo, x_hi = ax.get_xlim()

        ax.axhline(y=gil_r['mean_h'],
                   color='#1a1a2e', linestyle=(0, (5, 3)),
                   linewidth=2.0, alpha=0.85,
                   label='Gillespie (reference)', zorder=5)
        ax.fill_between(
            [x_lo, x_hi],
            gil_r['ci_lo'], gil_r['ci_hi'],
            color='#1a1a2e', alpha=0.06, zorder=1
        )

    # ── Labels, style ─────────────────────────────────────────────────────
    ci_pct = int((1 - alpha) * 100)
    N_val  = samples.metadata['N']
    L_val  = samples.metadata['L']
    nmc    = samples.metadata['n_mc']

    ax.set_xlabel("NFE (Number of Function Evaluations)",
                  fontsize=12, labelpad=8, color='#333333')
    ax.set_ylabel(f"Marginal Hellinger  H  (k={k})",
                  fontsize=12, labelpad=8, color='#333333')
    ax.set_title(
        f"Hellinger vs NFE  —  N={N_val}, L={L_val}, n_mc={nmc:,}\n"
        f"k={k},  t={t_eval},  Bootstrap {ci_pct}% CI  (B={n_bootstrap})",
        fontsize=11, pad=12, color='#111111'
    )

    ax.tick_params(colors='#555555', labelsize=10)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.legend(fontsize=10, framealpha=0.9, edgecolor='#dddddd',
              fancybox=False, labelspacing=0.4)
    ax.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)

    # Let matplotlib auto-scale y axis — no forced y=0
    ax.autoscale(axis='y')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(0, y_min * 0.95), y_max * 1.05)

    plt.tight_layout(pad=1.5)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")

    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: (r['line_name'], r['nfe']))

    print(f"\n{'=' * 90}")
    print(f"HELLINGER SUMMARY  —  t={t_eval}, k={k}")
    print(f"{'=' * 90}")
    print(f"{'Line':<25} {'NFE':>6} {'H mean':>10} {'CI lo':>10} "
          f"{'CI hi':>10}  Method")
    print("-" * 90)
    for r in results_sorted:
        print(f"{r['line_name']:<25} {r['nfe']:>6} "
              f"{r['mean_h']:>10.6f} {r['ci_lo']:>10.6f} {r['ci_hi']:>10.6f}"
              f"  {r['method_name']}")
    print("=" * 90)

    return results, fig, ax


def build_empirical_distributions(
    method_names,
    k,
    methods_dir='methods',
    t=0.0,
    seed=None,
):
    """
    For a list of methods, sample a fixed set of k coordinates once,
    then compute the empirical distribution of each method's particles
    projected onto those coordinates at time t.

    Loads and frees each method file from RAM one at a time.

    Parameters
    ----------
    method_names : list of str
        Names of methods to include (without .pkl extension).
    k : int
        Number of coordinates to project onto.
    methods_dir : str
        Path to the methods/ folder.
    t : float
        Time point to evaluate at (must match a key in the particles dict).
    seed : int or None
        Random seed for coordinate selection.

    Returns
    -------
    result : dict with keys:
        'subset'  : tuple of k ints — the coordinates chosen
        'k'       : int
        't'       : float
        'forward' : dict with 'states' (S,k), 'counts' (S,), 'n' int
        'methods' : dict[method_name -> dict with 'states', 'counts', 'n', 'nfe']
    """

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(methods_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    N        = meta['forward'][0.0].shape[1]
    nfe_dict = meta.get('nfe', {})

    # ── Sample fixed k coordinates ────────────────────────────────────────────
    rng    = np.random.default_rng(seed)
    subset = tuple(int(i) for i in rng.choice(N, size=k, replace=False))
    print(f"Selected {k} coordinates: {subset}")

    # ── Empirical distribution of forward at time t ───────────────────────────
    fwd_particles = meta['forward'][float(t)][:, list(subset)]
    fwd_states, fwd_counts = np.unique(fwd_particles, axis=0, return_counts=True)
    fwd_n = len(fwd_particles)

    del meta
    gc.collect()

    result = {
        'subset':  subset,
        'k':       k,
        't':       float(t),
        'forward': {
            'states': fwd_states,   # shape (S_fwd, k)
            'counts': fwd_counts,   # shape (S_fwd,)
            'n':      fwd_n,
        },
        'methods': {},
    }

    # ── Process each method ───────────────────────────────────────────────────
    for method_name in method_names:
        fpath = os.path.join(methods_dir, f"{method_name}.pkl")

        if not os.path.exists(fpath):
            print(f"  WARNING: {method_name}.pkl not found, skipping.")
            continue

        with open(fpath, 'rb') as f:
            particles_dict = pickle.load(f)

        if float(t) not in particles_dict:
            print(f"  WARNING: t={t} not found in {method_name}, skipping.")
            del particles_dict
            gc.collect()
            continue

        rev_particles = particles_dict[float(t)][:, list(subset)]
        rev_states, rev_counts = np.unique(rev_particles, axis=0, return_counts=True)
        rev_n = len(rev_particles)

        del particles_dict, rev_particles
        gc.collect()

        nfe = nfe_dict.get(method_name, None)

        result['methods'][method_name] = {
            'states': rev_states,   # shape (S_rev, k)
            'counts': rev_counts,   # shape (S_rev,)
            'n':      rev_n,
            'nfe':    nfe,
        }

        # print(f"  Loaded {method_name}  |  nfe={nfe}  |  unique states={len(rev_states)}")

    return result




CORRECTOR_NAMES = {'random_masking', 'informed_corrector', 'DPC', 'PRISM'}


def _bootstrap_hellinger_from_counts(states_f, counts_f, n_f,
                                      states_r, counts_r, n_r,
                                      n_bootstrap, alpha, rng):
    """Bootstrap Hellinger from precomputed (states, counts) pairs."""
    if states_f.ndim == 1:
        states_f = states_f.reshape(-1, 1)
        states_r = states_r.reshape(-1, 1)

    k    = states_f.shape[1]
    L    = int(max(states_f.max(), states_r.max())) + 2
    base = L ** np.arange(k - 1, -1, -1)

    codes_f     = states_f @ base
    codes_r     = states_r @ base
    all_codes   = np.union1d(codes_f, codes_r)
    S           = len(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    p_f = np.zeros(S)
    for code, count in zip(codes_f, counts_f):
        p_f[code_to_idx[code]] = count
    p_f /= p_f.sum()

    p_r = np.zeros(S)
    for code, count in zip(codes_r, counts_r):
        p_r[code_to_idx[code]] = count
    p_r /= p_r.sum()

    boot_f = rng.multinomial(n_f, p_f, size=n_bootstrap) / n_f
    boot_r = rng.multinomial(n_r, p_r, size=n_bootstrap) / n_r
    diff   = np.sqrt(boot_f) - np.sqrt(boot_r)
    boot_h = np.sqrt(0.5 * np.sum(diff * diff, axis=1))

    return (float(boot_h.mean()),
            float(np.quantile(boot_h, alpha / 2)),
            float(np.quantile(boot_h, 1.0 - alpha / 2)))


def compute_hellinger_at_t0(
    result,
    corrector_present,
    k,
    corrector_name=None,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
):
    """
    Using precomputed empirical distributions from build_empirical_distributions(),
    compute the marginal Hellinger distance with bootstrap CIs for each method.

    Parameters
    ----------
    result : dict
        Output of build_empirical_distributions().
    corrector_present : bool
        If False, only plain tau-leap methods (no corrector in name).
        If True, only methods containing corrector_name.
    k : int
        Marginal order (must match the k used in build_empirical_distributions).
    corrector_name : str or None
        Required if corrector_present=True.
        One of 'random_masking', 'informed_corrector', 'DPC', 'PRISM'.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        CI significance level (0.05 → 95% CI).
    seed : int or None

    Returns
    -------
    df : pd.DataFrame
        Columns: method_name, nfe, hellinger_mean, ci_lo, ci_hi
        Sorted by nfe.
    """

    # ── Validate ──────────────────────────────────────────────────────────────
    if corrector_present:
        if corrector_name is None:
            raise ValueError("corrector_name must be provided when corrector_present=True")
        if corrector_name not in CORRECTOR_NAMES:
            raise ValueError(
                f"corrector_name must be one of {CORRECTOR_NAMES}, got '{corrector_name}'"
            )

    assert k == result['k'], (
        f"k={k} does not match result['k']={result['k']}. "
        f"Rebuild distributions with the correct k."
    )

    rng = np.random.default_rng(seed)
    fwd = result['forward']

    # ── Filter methods ────────────────────────────────────────────────────────
    all_methods = list(result['methods'].keys())

    if corrector_present:
        methods_to_eval = [m for m in all_methods if corrector_name in m]
    else:
        methods_to_eval = [
            m for m in all_methods
            if 'tau_leap' in m and not any(c in m for c in CORRECTOR_NAMES)
        ]

    print(f"Found {len(methods_to_eval)} matching methods.")

    # ── Compute bootstrap Hellinger for each method ───────────────────────────
    rows = []

    for method_name in sorted(methods_to_eval):
        m   = result['methods'][method_name]
        nfe = m['nfe']

        if nfe is None:
            print(f"  WARNING: NFE not found for '{method_name}', skipping.")
            continue

        mean_h, ci_lo, ci_hi = _bootstrap_hellinger_from_counts(
            fwd['states'], fwd['counts'], fwd['n'],
            m['states'],   m['counts'],   m['n'],
            n_bootstrap, alpha, rng
        )

        # print(f"  {method_name}  |  nfe={nfe}  |  H={mean_h:.4f}  "
        #       f"[{ci_lo:.4f}, {ci_hi:.4f}]")

        rows.append({
            'method_name':    method_name,
            'nfe':            nfe,
            'hellinger_mean': mean_h,
            'ci_lo':          ci_lo,
            'ci_hi':          ci_hi,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('nfe').reset_index(drop=True)

    return df





# =============================================================================
# Internal helpers
# =============================================================================

def _encode(arr):
    n, k = arr.shape
    L    = int(arr.max()) + 2
    base = L ** np.arange(k - 1, -1, -1)
    return arr @ base


def _bootstrap_hellinger_from_counts(states_f, counts_f, n_f,
                                      states_r, counts_r, n_r,
                                      n_bootstrap, alpha, rng):
    """
    Bootstrap Hellinger distance from pre-computed empirical distributions
    stored as (states, counts) pairs — no need to reload raw particles.
    """
    # Build a common state space by encoding states as integers
    # Encode both state arrays
    if states_f.ndim == 1:
        states_f = states_f.reshape(-1, 1)
        states_r = states_r.reshape(-1, 1)

    k  = states_f.shape[1]
    L  = int(max(states_f.max(), states_r.max())) + 2
    base = L ** np.arange(k - 1, -1, -1)

    codes_f = states_f @ base   # shape (S_f,)
    codes_r = states_r @ base   # shape (S_r,)

    # Merge into a common vocabulary
    all_codes      = np.union1d(codes_f, codes_r)
    S              = len(all_codes)
    code_to_idx    = {c: i for i, c in enumerate(all_codes)}

    p_f = np.zeros(S)
    for code, count in zip(codes_f, counts_f):
        p_f[code_to_idx[code]] = count
    p_f /= p_f.sum()

    p_r = np.zeros(S)
    for code, count in zip(codes_r, counts_r):
        p_r[code_to_idx[code]] = count
    p_r /= p_r.sum()

    # Parametric multinomial bootstrap
    boot_f = rng.multinomial(n_f, p_f, size=n_bootstrap) / n_f
    boot_r = rng.multinomial(n_r, p_r, size=n_bootstrap) / n_r
    diff   = np.sqrt(boot_f) - np.sqrt(boot_r)
    boot_h = np.sqrt(0.5 * np.sum(diff * diff, axis=1))

    return (float(boot_h.mean()),
            float(np.quantile(boot_h, alpha / 2)),
            float(np.quantile(boot_h, 1.0 - alpha / 2)))


def _assign_colors_and_markers(curve_labels):
    exact_colors = {
        'gillespie': '#1a1a2e',
        'Gillespie': '#1a1a2e',
        'tau_leap':  '#4361ee',
        'Tau-leaping':  '#4361ee',
    }
    exact_markers = {
        'gillespie': '*',
        'Gillespie': '*',
        'tau_leap':  'o',
        'Tau-leaping':  'o',
    }
    tau_seq   = ['#4361ee','#3a86ff','#4cc9f0','#2d6a9f','#023e8a',
                 '#0077b6','#48cae4','#0096c7','#00b4d8','#90e0ef']
    prism_seq = ['#f77f00','#fcbf49','#e76f51','#d62828','#f4a261',
                 '#e9c46a','#fb8500','#ffb703','#e63946','#c77dff']
    ic_seq    = ['#80b918','#55a630','#007f5f','#1b4332','#40916c','#2a9d8f','#52b788','#57cc99','#38a3a5','#22577a',
                 ]
    ic_no_margin_seq = ['#e63946','#ca6702','#ff4d6d','#ff758f','#c9184a','#ff0054',
                 '#ef233c','#d90429','#ff6b6b','#ee9b00','#ca6702']
    rm_seq    = ['#7b2d8b','#ff6b6b','#b5179e','#9b5de5','#c77dff','#e0aaff','#7209b7',
                 '#560bad','#480ca8','#3a0ca3','#3f37c9']


    tau_idx = prism_idx = ic_idx = ic_no_margin_idx = rm_idx  = 0
    color_map  = {}
    marker_map = {}

    for label in curve_labels:
        if label in exact_colors:
            color_map[label]  = exact_colors[label]
            marker_map[label] = exact_markers.get(label, 'o')
        elif 'PRISM' in label:
            color_map[label]  = prism_seq[prism_idx % len(prism_seq)]
            marker_map[label] = 's'
            prism_idx += 1

        elif 'informed' in label.lower() and 'no margin' in label.lower():
            color_map[label]  = ic_no_margin_seq[ic_no_margin_idx % len(ic_no_margin_seq)]
            marker_map[label] = 'v'
            ic_no_margin_idx += 1
        elif 'informed' in label.lower():
            color_map[label]  = ic_seq[ic_idx % len(ic_seq)]
            marker_map[label] = '^'
            ic_idx += 1
        elif 'random' in label.lower() or 'RM' in label:
            color_map[label]  = rm_seq[rm_idx % len(rm_seq)]
            marker_map[label] = 'D'
            rm_idx += 1
        # elif 'DPC' in label or 'dpc' in label:
        #     color_map[label]  = dpc_seq[dpc_idx % len(dpc_seq)]
        #     marker_map[label] = 'P'
        #     dpc_idx += 1
        elif 'tau' in label.lower():
            color_map[label]  = tau_seq[tau_idx % len(tau_seq)]
            marker_map[label] = 'o'
            tau_idx += 1
        else:
            color_map[label]  = '#adb5bd'
            marker_map[label] = 'x'

    return color_map, marker_map


# =============================================================================
# Build empirical distributions
# =============================================================================

def build_empirical_distributions(
    method_names,
    k,
    methods_dir='methods',
    t=0.0,
    seed=None,
):
    """
    For a list of methods, sample a fixed set of k coordinates once,
    then compute the empirical distribution of each method's particles
    projected onto those coordinates at time t.

    Parameters
    ----------
    method_names : list of str
    k : int
    methods_dir : str
    t : float
    seed : int or None

    Returns
    -------
    result : dict with keys:
        'subset'  : tuple of k ints
        'k'       : int
        't'       : float
        'forward' : {'states', 'counts', 'n'}
        'methods' : {method_name -> {'states', 'counts', 'n', 'nfe'}}
    """
    meta_path = os.path.join(methods_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    N        = meta['forward'][0.0].shape[1]
    nfe_dict = meta.get('nfe', {})

    rng    = np.random.default_rng(seed)
    subset = tuple(int(i) for i in rng.choice(N, size=k, replace=False))
    print(f"Selected {k} coordinates: {subset}")

    fwd_particles          = meta['forward'][float(t)][:, list(subset)]
    fwd_states, fwd_counts = np.unique(fwd_particles, axis=0, return_counts=True)
    fwd_n                  = len(fwd_particles)

    del meta
    gc.collect()

    result = {
        'subset':  subset,
        'k':       k,
        't':       float(t),
        'forward': {'states': fwd_states, 'counts': fwd_counts, 'n': fwd_n},
        'methods': {},
    }

    for method_name in method_names:
        fpath = os.path.join(methods_dir, f"{method_name}.pkl")
        if not os.path.exists(fpath):
            print(f"  WARNING: {method_name}.pkl not found, skipping.")
            continue

        with open(fpath, 'rb') as f:
            particles_dict = pickle.load(f)

        if float(t) not in particles_dict:
            print(f"  WARNING: t={t} not in {method_name}, skipping.")
            del particles_dict
            gc.collect()
            continue

        rev_particles          = particles_dict[float(t)][:, list(subset)]
        rev_states, rev_counts = np.unique(rev_particles, axis=0, return_counts=True)
        rev_n                  = len(rev_particles)

        del particles_dict, rev_particles
        gc.collect()

        nfe = nfe_dict.get(method_name, None)
        result['methods'][method_name] = {
            'states': rev_states,
            'counts': rev_counts,
            'n':      rev_n,
            'nfe':    nfe,
        }
        # print(f"  Loaded {method_name}  |  nfe={nfe}  |  unique states={len(rev_states)}")

    return result


# =============================================================================
# Plot from precomputed empirical distributions
# =============================================================================

def plot_nfe_from_distributions(
    result,
    lines,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
    filename=None,
    figsize=(10, 6),
):
    """
    Plot marginal Hellinger vs NFE using precomputed empirical distributions
    from build_empirical_distributions().

    Parameters
    ----------
    result : dict
        Output of build_empirical_distributions().
    lines : dict {str -> list of str}
        Keys   = line name (legend label)
        Values = list of method names forming that line.
        Methods with 'gillespie' in the line name are plotted as a
        horizontal dashed reference.

        Example
        -------
        lines = {
            'tau=0.6':     ['tau_leap_0.6'],
            'tau=1.0':     ['tau_leap_1.0'],
            'PRISM eta=0.2': [
                'tau_leap_0.6_corrector_PRISM_start_1.0_eta_0.2',
                'tau_leap_1.0_corrector_PRISM_start_1.0_eta_0.2',
            ],
            'gillespie':   ['gillespie'],
        }

    n_bootstrap : int
    alpha : float
    seed : int or None
    filename : str or None
    figsize : tuple

    Returns
    -------
    results : list of dict
        One entry per method with keys:
        line_name, method_name, nfe, mean_h, ci_lo, ci_hi
    fig, ax
    """
    rng      = np.random.default_rng(seed)
    k        = result['k']
    t        = result['t']
    fwd      = result['forward']
    methods  = result['methods']

    # ── Compute bootstrap Hellinger for each method ───────────────────────────
    results = []

    for line_name, method_list in lines.items():
        print(f"Line '{line_name}':")
        for method_name in method_list:
            if method_name not in methods:
                print(f"  WARNING: '{method_name}' not in result, skipping.")
                continue

            m   = methods[method_name]
            nfe = m['nfe']
            if nfe is None:
                print(f"  WARNING: NFE not found for '{method_name}', skipping.")
                continue

            mean_h, ci_lo, ci_hi = _bootstrap_hellinger_from_counts(
                fwd['states'], fwd['counts'], fwd['n'],
                m['states'],   m['counts'],   m['n'],
                n_bootstrap, alpha, rng
            )

            print(f"  NFE={nfe:4d}  H={mean_h:.4f}  "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}]  ({method_name})")

            results.append({
                'line_name':   line_name,
                'method_name': method_name,
                'nfe':         nfe,
                'mean_h':      mean_h,
                'ci_lo':       ci_lo,
                'ci_hi':       ci_hi,
            })

    if not results:
        raise ValueError("No results computed — check method names.")

    # ── Assign colors and markers ─────────────────────────────────────────────
    unique_lines           = list(lines.keys())
    color_map, marker_map  = _assign_colors_and_markers(unique_lines)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    gil_lines  = [ln for ln in unique_lines if 'gillespie' in ln.lower()]
    plot_lines = [ln for ln in unique_lines if 'gillespie' not in ln.lower()]

    # Non-gillespie curves
    for line_name in plot_lines:
        line_results = sorted(
            [r for r in results if r['line_name'] == line_name],
            key=lambda r: r['nfe']
        )
        if not line_results:
            continue

        color  = color_map[line_name]
        marker = marker_map[line_name]
        nfes   = [r['nfe']    for r in line_results]
        means  = [r['mean_h'] for r in line_results]
        ci_los = [r['ci_lo']  for r in line_results]
        ci_his = [r['ci_hi']  for r in line_results]

        ax.plot(nfes, means, color=color, marker=marker,
                linewidth=2.2, markersize=8,
                markeredgecolor='white', markeredgewidth=0.8,
                label=line_name, zorder=3, solid_capstyle='round')
        ax.fill_between(nfes, ci_los, ci_his, color=color, alpha=0.12, zorder=2)
        for nfe, lo, hi in zip(nfes, ci_los, ci_his):
            ax.plot([nfe, nfe], [lo, hi], color=color,
                    linewidth=1.2, alpha=0.5, zorder=4)

    # Gillespie horizontal reference
    all_nfes = [r['nfe'] for r in results if 'gillespie' not in r['line_name'].lower()]
    for gil_line in gil_lines:
        gil_results = [r for r in results if r['line_name'] == gil_line]
        if not gil_results:
            continue
        gil_r = gil_results[0]
        x_lo  = (min(all_nfes) - 0.3) if all_nfes else 0
        x_hi  = (max(all_nfes) + 0.3) if all_nfes else 1
        ax.axhline(y=gil_r['mean_h'], color='#1a1a2e',
                   linestyle=(0, (5, 3)), linewidth=2.0, alpha=0.85,
                   label='Gillespie (baseline)', zorder=5)
        ax.fill_between([x_lo, x_hi], gil_r['ci_lo'], gil_r['ci_hi'],
                        color='#1a1a2e', alpha=0.06, zorder=1)

    ci_pct = int((1 - alpha) * 100)
    ax.set_xlabel("NFE (Number of Function Evaluations)",
                  fontsize=15, labelpad=8, color='#333333')
    ax.set_ylabel(f"Marginal Hellinger  H  ($s={k}$)",
                  fontsize=15, labelpad=8, color='#333333')
    # ax.set_title(
    #     f"Hellinger vs NFE\n"
    #     f"k={k},  t={t},  Bootstrap {ci_pct}% CI  (B={n_bootstrap})",
    #     fontsize=11, pad=12, color='#111111'
    # )
    ax.tick_params(colors='#555555', labelsize=10)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.legend(fontsize=15, framealpha=0.9, edgecolor='#dddddd',
              fancybox=False, labelspacing=0.4)
    ax.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.autoscale(axis='y')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(0, y_min * 0.95), y_max * 1.05)

    plt.tight_layout(pad=1.5)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

    plt.show()


    return results, fig, ax

def plot_vocabulary_distribution_from_result(
    result,
    indices,
    methods,
    L,
    labels=None,
    plot_filename='vocabulary_distribution.png',
):
    """
    For each index in `indices`, plot the empirical marginal distribution over
    vocabulary values {0, ..., L-1} as side-by-side bars for forward and each
    reverse method, using precomputed empirical distributions from
    build_empirical_distributions().

    Parameters
    ----------
    result : dict
        Output of build_empirical_distributions().
    indices : list of int
        Positions within result['subset'] to plot (0 to k-1).
    methods : list of str
        Method names to include. Must be keys in result['methods'].
    L : int
        Vocabulary size.
    labels : dict or None
        Optional mapping from method name to display label in legend.
        e.g. {'gillespie': 'Gillespie', 'tau_leap_0.6': 'Tau-leap'}
        'forward' can also be relabelled: {'forward': 'Target'}
        If None, full method name is used.
    plot_filename : str
        Path to save the output plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    subset       = result['subset']
    k            = result['k']
    t            = result['t']
    fwd          = result['forward']
    methods_data = result['methods']

    assert max(indices) < k, (
        f"Index {max(indices)} out of range for k={k}. "
        f"Indices must be positions within the subset {subset}."
    )

    # All labels: forward + requested methods
    all_method_names = ['forward'] + methods
    vocab_values     = np.arange(L)

    # Resolve display labels
    def get_label(name):
        if labels and name in labels:
            return labels[name]
        return name

    cmap      = plt.cm.tab10
    colors    = [cmap(i) for i in range(len(all_method_names))]
    bar_width = 0.8 / len(all_method_names)

    n_indices = len(indices)
    fig, axes = plt.subplots(1, n_indices, figsize=(5 * n_indices, 5), squeeze=False)

    for col, idx in enumerate(indices):
        ax       = axes[0, col]
        orig_dim = subset[idx]

        for b, (name, color) in enumerate(zip(all_method_names, colors)):
            if name == 'forward':
                states = fwd['states']
                counts = fwd['counts']
                n      = fwd['n']
            else:
                if name not in methods_data:
                    print(f"WARNING: '{name}' not in result['methods'], skipping.")
                    continue
                m      = methods_data[name]
                states = m['states']
                counts = m['counts']
                n      = m['n']

            col_vals = states if states.ndim == 1 else states[:, idx]

            probs = np.zeros(L)
            for v, c in zip(col_vals, counts):
                if 0 <= v < L:
                    probs[v] += c
            probs /= n

            offset = (b - len(all_method_names) / 2 + 0.5) * bar_width
            ax.bar(vocab_values + offset, probs, width=bar_width,
                   color=color, label=get_label(name), alpha=0.85)

        ax.set_xlabel('Token value', fontsize=11)
        ax.set_ylabel('Empirical probability', fontsize=11)
        ax.set_title(f'Dimension {orig_dim}', fontsize=12)
        ax.set_xticks(vocab_values)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # fig.suptitle(
    #     f'Vocabulary distribution at t={t:.4f}\nSubset: {subset}',
    #     fontsize=13, y=1.02
    # )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()

import os
import pickle
import gc
import numpy as np
import matplotlib.pyplot as plt


def _hellinger_direct(fwd, rev):
    """Compute Hellinger distance directly without bootstrap."""
    n_mc  = len(fwd)
    L_val = int(max(fwd.max(), rev.max())) + 2
    k_val = fwd.shape[1]
    base  = L_val ** np.arange(k_val - 1, -1, -1)

    codes_f   = fwd @ base
    codes_r   = rev @ base
    all_codes = np.concatenate([codes_f, codes_r])
    uniq, inv = np.unique(all_codes, return_inverse=True)
    S         = len(uniq)

    p_f  = np.bincount(inv[:n_mc], minlength=S) / n_mc
    p_r  = np.bincount(inv[n_mc:], minlength=S) / n_mc
    diff = np.sqrt(p_f) - np.sqrt(p_r)
    return float(np.sqrt(0.5 * np.sum(diff ** 2)))


def _bootstrap_hellinger_from_particles(fwd, rev, n_bootstrap, alpha, rng):
    """Bootstrap Hellinger from raw particle arrays."""
    n_mc  = len(fwd)
    L_val = int(max(fwd.max(), rev.max())) + 2
    k_val = fwd.shape[1]
    base  = L_val ** np.arange(k_val - 1, -1, -1)

    codes_f   = fwd @ base
    codes_r   = rev @ base
    all_codes = np.concatenate([codes_f, codes_r])
    uniq, inv = np.unique(all_codes, return_inverse=True)
    S         = len(uniq)
    inv_f, inv_r = inv[:n_mc], inv[n_mc:]

    count_f = np.bincount(inv_f, minlength=S).astype(float)
    count_r = np.bincount(inv_r, minlength=S).astype(float)
    p_f     = count_f / count_f.sum()
    p_r     = count_r / count_r.sum()

    boot_f = rng.multinomial(n_mc, p_f, size=n_bootstrap) / n_mc
    boot_r = rng.multinomial(n_mc, p_r, size=n_bootstrap) / n_mc
    diff   = np.sqrt(boot_f) - np.sqrt(boot_r)
    boot_h = np.sqrt(0.5 * np.sum(diff * diff, axis=1))

    return (float(boot_h.mean()),
            float(np.quantile(boot_h, alpha / 2)),
            float(np.quantile(boot_h, 1.0 - alpha / 2)))


def plot_hellinger_across_times(
    method_names,
    k,
    methods_dir='methods',
    bootstrap=True,
    n_bootstrap=200,
    alpha=0.05,
    t_range=None,
    seed=None,
    filename=None,
    figsize=(12, 6),
):
    """
    Load each method, compute marginal Hellinger at checkpoint times,
    plot H vs time, then delete data from memory.

    Parameters
    ----------
    method_names : list of str
        Method names to plot (without .pkl). Each becomes one line.
    k : int
        Marginal order — number of random dimensions to project onto.
    methods_dir : str
        Path to the methods/ folder.
    bootstrap : bool
        If True, compute bootstrap CIs. If False, direct Hellinger (faster).
    n_bootstrap : int
        Number of bootstrap resamples (only used if bootstrap=True).
    alpha : float
        CI significance level (only used if bootstrap=True).
    t_range : tuple of (float, float) or None
        If provided, only plot times in [t_range[0], t_range[1]].
        e.g. t_range=(0.0, 2.0)
    seed : int or None
    filename : str or None
    figsize : tuple
    """
    rng = np.random.default_rng(seed)

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(methods_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    N        = meta['forward'][0.0].shape[1]
    all_times = sorted(meta['forward'].keys())
    nfe_dict  = meta.get('nfe', {})
    forward   = meta['forward']

    # ── Filter times ──────────────────────────────────────────────────────────
    if t_range is not None:
        times = [t for t in all_times if t_range[0] <= t <= t_range[1]]
        print(f"Time range: [{t_range[0]}, {t_range[1]}] → {len(times)} checkpoints")
    else:
        times = all_times
        print(f"All {len(times)} checkpoints")

    # ── Sample fixed k coordinates ────────────────────────────────────────────
    subset = tuple(int(i) for i in rng.choice(N, size=k, replace=False))
    print(f"Selected {k} coordinates: {subset}")

    del meta
    gc.collect()

    # ── Color and marker assignment ───────────────────────────────────────────
    cmap       = plt.cm.tab10
    colors     = {m: cmap(i) for i, m in enumerate(method_names)}
    markers    = ['o', 's', '^', 'D', 'P', '*', 'v', 'X', 'h', '+']
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(method_names)}

    # ── Plot setup ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    # ── Process each method ───────────────────────────────────────────────────
    for method_name in method_names:
        fpath = os.path.join(methods_dir, f"{method_name}.pkl")
        if not os.path.exists(fpath):
            print(f"WARNING: {method_name}.pkl not found, skipping.")
            continue

        print(f"\nProcessing {method_name}...")
        with open(fpath, 'rb') as f:
            particles_dict = pickle.load(f)

        nfe   = nfe_dict.get(method_name, None)
        label = f"{method_name}  (NFE={nfe})" if nfe else method_name

        means, ci_los, ci_his = [], [], []

        for t in times:
            fwd_proj = forward[t][:, list(subset)]
            rev_proj = particles_dict[t][:, list(subset)]

            if bootstrap:
                mean_h, ci_lo, ci_hi = _bootstrap_hellinger_from_particles(
                    fwd_proj, rev_proj, n_bootstrap, alpha, rng
                )
            else:
                mean_h = _hellinger_direct(fwd_proj, rev_proj)
                ci_lo  = mean_h
                ci_hi  = mean_h

            means.append(mean_h)
            ci_los.append(ci_lo)
            ci_his.append(ci_hi)

        del particles_dict
        gc.collect()

        color  = colors[method_name]
        marker = marker_map[method_name]

        ax.plot(times, means, color=color, marker=marker,
                linewidth=2.0, markersize=5, label=label,
                markeredgecolor='white', markeredgewidth=0.6,
                zorder=3, solid_capstyle='round')

        if bootstrap:
            ax.fill_between(times, ci_los, ci_his,
                            color=color, alpha=0.12, zorder=2)

        print(f"  Done. NFE={nfe}")

    del forward
    gc.collect()

    # ── Style ─────────────────────────────────────────────────────────────────
    ci_str = f"Bootstrap {int((1-alpha)*100)}% CI (B={n_bootstrap})" if bootstrap else "No CI"
    t_str  = f"t ∈ [{t_range[0]}, {t_range[1]}]" if t_range else "all times"

    ax.set_xlabel("Time t", fontsize=12, labelpad=8, color='#333333')
    ax.set_ylabel(f"Marginal Hellinger  H  ($s={k}$)", fontsize=12, labelpad=8, color='#333333')
    # ax.set_title(
    #     f"Hellinger distance across time  ({t_str})\nk={k},  {ci_str}",
    #     fontsize=11, pad=12, color='#111111'
    # )
    ax.tick_params(colors='#555555', labelsize=10)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='#dddddd',
              fancybox=False, labelspacing=0.4)
    ax.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)

    if t_range:
        ax.set_xlim(t_range[0], t_range[1])
    else:
        ax.set_xlim(left=0)

    plt.tight_layout(pad=1.5)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")

    plt.show()

    return fig, ax




def _hellinger_direct(fwd, rev):
    n_mc  = len(fwd)
    L_val = int(max(fwd.max(), rev.max())) + 2
    k_val = fwd.shape[1]
    base  = L_val ** np.arange(k_val - 1, -1, -1)
    codes_f   = fwd @ base
    codes_r   = rev @ base
    all_codes = np.concatenate([codes_f, codes_r])
    uniq, inv = np.unique(all_codes, return_inverse=True)
    S         = len(uniq)
    p_f  = np.bincount(inv[:n_mc], minlength=S) / n_mc
    p_r  = np.bincount(inv[n_mc:], minlength=S) / n_mc
    diff = np.sqrt(p_f) - np.sqrt(p_r)
    return float(np.sqrt(0.5 * np.sum(diff ** 2)))


def _bootstrap_hellinger(fwd, rev, n_bootstrap, alpha, rng):
    n_mc  = len(fwd)
    L_val = int(max(fwd.max(), rev.max())) + 2
    k_val = fwd.shape[1]
    base  = L_val ** np.arange(k_val - 1, -1, -1)
    codes_f   = fwd @ base
    codes_r   = rev @ base
    all_codes = np.concatenate([codes_f, codes_r])
    uniq, inv = np.unique(all_codes, return_inverse=True)
    S         = len(uniq)
    inv_f, inv_r = inv[:n_mc], inv[n_mc:]
    count_f = np.bincount(inv_f, minlength=S).astype(float)
    count_r = np.bincount(inv_r, minlength=S).astype(float)
    p_f     = count_f / count_f.sum()
    p_r     = count_r / count_r.sum()
    boot_f  = rng.multinomial(n_mc, p_f, size=n_bootstrap) / n_mc
    boot_r  = rng.multinomial(n_mc, p_r, size=n_bootstrap) / n_mc
    diff    = np.sqrt(boot_f) - np.sqrt(boot_r)
    boot_h  = np.sqrt(0.5 * np.sum(diff * diff, axis=1))
    return (float(boot_h.mean()),
            float(np.quantile(boot_h, alpha / 2)),
            float(np.quantile(boot_h, 1.0 - alpha / 2)))


# def _assign_colors_and_markers(line_names):
#     tau_seq   = ['#4361ee','#3a86ff','#4cc9f0','#2d6a9f','#023e8a']
#     prism_seq = ['#f77f00','#fcbf49','#e76f51','#d62828','#f4a261']
#     ic_seq    = ['#2a9d8f','#52b788','#57cc99','#38a3a5','#22577a']
#     rm_seq    = ['#7b2d8b','#ff6b6b','#b5179e','#e0aaff','#9b5de5','#c77dff','#7209b7']
#     dpc_seq   = ['#e63946','#ff4d6d','#ff758f','#c9184a','#ff0054']

#     tau_idx = prism_idx = ic_idx = rm_idx = dpc_idx = 0
#     color_map  = {}
#     marker_map = {}

#     for label in line_names:
#         if 'gillespie' in label.lower():
#             color_map[label]  = '#1a1a2e'
#             marker_map[label] = '*'
#         elif 'PRISM' in label:
#             color_map[label]  = prism_seq[prism_idx % len(prism_seq)]
#             marker_map[label] = 's'
#             prism_idx += 1
#         elif 'informed' in label.lower():
#             color_map[label]  = ic_seq[ic_idx % len(ic_seq)]
#             marker_map[label] = '^'
#             ic_idx += 1
#         elif 'random' in label.lower() or 'RM' in label:
#             color_map[label]  = rm_seq[rm_idx % len(rm_seq)]
#             marker_map[label] = 'D'
#             rm_idx += 1
#         elif 'DPC' in label or 'dpc' in label:
#             color_map[label]  = dpc_seq[dpc_idx % len(dpc_seq)]
#             marker_map[label] = 'P'
#             dpc_idx += 1
#         elif 'tau' in label.lower():
#             color_map[label]  = tau_seq[tau_idx % len(tau_seq)]
#             marker_map[label] = 'o'
#             tau_idx += 1
#         else:
#             color_map[label]  = '#adb5bd'
#             marker_map[label] = 'x'

#     return color_map, marker_map


def plot_nfe_from_folder(
    lines,
    k,
    methods_dir='methods',
    t=0.0,
    bootstrap=True,
    n_bootstrap=200,
    alpha=0.05,
    seed=None,
    filename=None,
    figsize=(12, 6),
):
    """
    Load methods from methods/ folder, compute marginal Hellinger at t=0.0,
    and plot H vs NFE. Each key in `lines` becomes one curve.

    Parameters
    ----------
    lines : dict {str -> list of str}
        Keys   = line name (legend label)
        Values = list of method names forming that line.
        Use 'gillespie' key for the horizontal reference line.

        Example
        -------
        lines = {
            'RM start=1.0': [
                'tau_leap_0.8',
                'tau_leap_0.8_corrector_random_masking_start_1.0_ncorr_1_tauc_0.1',
                'tau_leap_0.8_corrector_random_masking_start_1.0_ncorr_4_tauc_0.1',
                'tau_leap_0.8_corrector_random_masking_start_1.0_ncorr_8_tauc_0.1',
                'tau_leap_0.8_corrector_random_masking_start_1.0_ncorr_12_tauc_0.1',
            ],
            'gillespie': ['gillespie'],
        }

    k : int
        Marginal order.
    methods_dir : str
        Path to methods/ folder.
    t : float
        Time point to evaluate at (default 0.0).
    bootstrap : bool
        If True compute bootstrap CIs, if False direct Hellinger.
    n_bootstrap : int
    alpha : float
    seed : int or None
    filename : str or None
    figsize : tuple

    Returns
    -------
    results : list of dict
    fig, ax
    """
    rng = np.random.default_rng(seed)

    # ── Load meta ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(methods_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    N        = meta['forward'][0.0].shape[1]
    nfe_dict = meta.get('nfe', {})
    forward_t = meta['forward'][float(t)]

    del meta
    gc.collect()

    # ── Sample fixed k coordinates ────────────────────────────────────────────
    subset = tuple(int(i) for i in rng.choice(N, size=k, replace=False))
    print(f"Selected {k} coordinates: {subset}")
    fwd_proj = forward_t[:, list(subset)]

    del forward_t
    gc.collect()

    # ── Compute Hellinger for each method ─────────────────────────────────────
    results = []

    for line_name, method_list in lines.items():
        print(f"\nLine '{line_name}':")
        for method_name in method_list:
            fpath = os.path.join(methods_dir, f"{method_name}.pkl")
            if not os.path.exists(fpath):
                print(f"  WARNING: {method_name}.pkl not found, skipping.")
                continue

            with open(fpath, 'rb') as f:
                particles_dict = pickle.load(f)

            rev_proj = particles_dict[float(t)][:, list(subset)]
            del particles_dict
            gc.collect()

            nfe = nfe_dict.get(method_name, None)
            if nfe is None:
                print(f"  WARNING: NFE not found for '{method_name}', skipping.")
                continue

            if bootstrap:
                mean_h, ci_lo, ci_hi = _bootstrap_hellinger(
                    fwd_proj, rev_proj, n_bootstrap, alpha, rng
                )
            else:
                mean_h = _hellinger_direct(fwd_proj, rev_proj)
                ci_lo  = mean_h
                ci_hi  = mean_h

            print(f"  NFE={nfe:4d}  H={mean_h:.4f}  ({method_name})")

            results.append({
                'line_name':   line_name,
                'method_name': method_name,
                'nfe':         nfe,
                'mean_h':      mean_h,
                'ci_lo':       ci_lo,
                'ci_hi':       ci_hi,
            })

    if not results:
        raise ValueError("No results computed — check method names.")

    del fwd_proj
    gc.collect()

    # ── Plot ──────────────────────────────────────────────────────────────────
    unique_lines          = list(lines.keys())
    color_map, marker_map = _assign_colors_and_markers(unique_lines)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    gil_lines  = [ln for ln in unique_lines if 'gillespie' in ln.lower()]
    plot_lines = [ln for ln in unique_lines if 'gillespie' not in ln.lower()]

    # Non-gillespie curves
    for line_name in plot_lines:
        line_results = sorted(
            [r for r in results if r['line_name'] == line_name],
            key=lambda r: r['nfe']
        )
        if not line_results:
            continue

        color  = color_map[line_name]
        marker = marker_map[line_name]
        nfes   = [r['nfe']    for r in line_results]
        means  = [r['mean_h'] for r in line_results]
        ci_los = [r['ci_lo']  for r in line_results]
        ci_his = [r['ci_hi']  for r in line_results]

        ax.plot(nfes, means, color=color, marker=marker,
                linewidth=2.2, markersize=8, label=line_name,
                markeredgecolor='white', markeredgewidth=0.8,
                zorder=3, solid_capstyle='round')
        if bootstrap:
            ax.fill_between(nfes, ci_los, ci_his, color=color, alpha=0.12, zorder=2)
            for nfe, lo, hi in zip(nfes, ci_los, ci_his):
                ax.plot([nfe, nfe], [lo, hi], color=color,
                        linewidth=1.2, alpha=0.5, zorder=4)

    # Gillespie horizontal reference
    all_nfes = [r['nfe'] for r in results if 'gillespie' not in r['line_name'].lower()]
    for gil_line in gil_lines:
        gil_results = [r for r in results if r['line_name'] == gil_line]
        if not gil_results:
            continue
        gil_r = gil_results[0]
        x_lo  = (min(all_nfes) - 0.5) if all_nfes else 0
        x_hi  = (max(all_nfes) + 0.5) if all_nfes else 1
        ax.axhline(y=gil_r['mean_h'], color='#1a1a2e',
                   linestyle=(0, (5, 3)), linewidth=2.0, alpha=0.85,
                   label='Gillespie (baseline)', zorder=5)
        if bootstrap:
            ax.fill_between([x_lo, x_hi], gil_r['ci_lo'], gil_r['ci_hi'],
                            color='#1a1a2e', alpha=0.06, zorder=1)

    ci_str = f"Bootstrap {int((1-alpha)*100)}% CI (B={n_bootstrap})" if bootstrap else "No CI"
    ax.set_xlabel("NFE (Number of Function Evaluations)", fontsize=15, labelpad=8)
    ax.set_ylabel(f"Marginal Hellinger  H  ($s={k}$)", fontsize=15, labelpad=8)
    # ax.set_title(
    #     f"Hellinger vs NFE  —  t={t}\nk={k},  {ci_str}",
    #     fontsize=11, pad=12
    # )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=15, framealpha=0.9, edgecolor='#dddddd',
              fancybox=False, labelspacing=0.4)
    ax.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)

    plt.tight_layout(pad=1.5)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")

    plt.show()

    return results, fig, ax