import matplotlib.pyplot as plt


def plot_method_comparison(samples, methods=None, filename='diffusion_comparison_all_methods.png', 
                          figsize=(12, 6), show_annotations=True, return_distances=False,
                          time_start=None, time_end=None):
    """
    Plot Hellinger distance comparison for selected reverse methods.
    
    Parameters
    ----------
    samples : DiffusionSamples
        Container with forward and reverse samples
    methods : list of str or None
        List of method names to plot. If None, plots all available methods.
        Example: ['gillespie', 'tau_leap_0.01']
    filename : str
        Output filename for the plot
    figsize : tuple
        Figure size (width, height)
    show_annotations : bool
        Whether to show t=0 and t=T annotations
    return_distances : bool
        Whether to return the distances dictionary
    time_start : float or None
        Start time for plot x-axis. If None, uses minimum time.
    time_end : float or None
        End time for plot x-axis. If None, uses maximum time.
    
    Returns
    -------
    all_distances : dict (only if return_distances=True)
        Dictionary of Hellinger distances for each method
    """
    # Determine which methods to plot
    if methods is None:
        methods = samples.list_methods()
    else:
        # Validate that requested methods exist
        available = samples.list_methods()
        invalid = [m for m in methods if m not in available]
        if invalid:
            raise ValueError(f"Methods not found: {invalid}. Available: {available}")
    
    # Compute distances for all methods
    all_distances = samples.compute_all_hellinger_distances()
    
    # Filter to only selected methods
    selected_distances = {m: all_distances[m] for m in methods}
    
    # Filter times based on time_start and time_end
    if time_start is None:
        time_start = samples.times[0]
    if time_end is None:
        time_end = samples.times[-1]
    
    # Get times within the specified range
    time_mask = (samples.times >= time_start) & (samples.times <= time_end)
    plot_times = samples.times[time_mask]
    
    if len(plot_times) == 0:
        raise ValueError(f"No checkpoints in range [{time_start}, {time_end}]")
    
    # Plot comparisons
    print(f"\nGenerating plot for methods: {methods}")
    print(f"Time range: [{time_start:.3f}, {time_end:.3f}]")
    plt.figure(figsize=figsize)
    
    for method_name in methods:
        distances = selected_distances[method_name]
        # Only plot times in the specified range
        plot_distances = [distances[t] for t in plot_times]
        plt.plot(plot_times, plot_distances, 
                 'o-', label=method_name, linewidth=2, markersize=4)
    
    plt.xlabel('Physical time t', fontsize=12)
    plt.ylabel('Hellinger distance', fontsize=12)
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
            if t0 in distances:
                # Annotate start of range
                y_offset = 0.05 + i * 0.03  # Stagger annotations if multiple methods
                plt.annotate(f'{method_name} t={t0:.2f}: H={distances[t0]:.3f}', 
                             xy=(t0, distances[t0]), 
                             xytext=(time_start + 0.1 * (time_end - time_start), distances[t0] + y_offset),
                             arrowprops=dict(arrowstyle='->', alpha=0.7),
                             fontsize=9)
            
            if tT in distances:
                # Annotate end of range
                plt.annotate(f'{method_name} t={tT:.2f}: H={distances[tT]:.3f}', 
                             xy=(tT, distances[tT]), 
                             xytext=(time_start + 0.8 * (time_end - time_start), distances[tT] + y_offset),
                             arrowprops=dict(arrowstyle='->', alpha=0.7),
                             fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.show()
    
    if return_distances:
        return all_distances  # Return all distances, not just selected ones
    
# ============================================================================
### How to use the plotting function
# ============================================================================

        # **Plot all methods**

        # plot_method_comparison(samples_2)

        # **Plot only specific methods**

        # plot_method_comparison(samples_2, methods=['gillespie', 'tau_leap_0.01'])

        # **Compare just two tau values**

        # plot_method_comparison(samples_2, methods=['tau_leap_0.01', 'tau_leap_0.1'])

        # **Single method**

        # plot_method_comparison(samples_2, methods=['gillespie'], 
        #                       filename='gillespie_only.png')

        # **Multiple methods with custom settings**

        # plot_method_comparison(
        #     samples_2, 
        #     methods=['gillespie', 'tau_leap_0.01', 'tau_leap_0.05'],
        #     filename='selected_methods.png',
        #     figsize=(15, 8),
        #     show_annotations=False
        # )

        # **Check what methods are available first**

        # print("Available methods:", samples_2.list_methods())
        # plot_method_comparison(samples_2, methods=['gillespie', 'tau_leap_0.1'])

        # **Set return_dstances=True to get the distance dict for all methods**``