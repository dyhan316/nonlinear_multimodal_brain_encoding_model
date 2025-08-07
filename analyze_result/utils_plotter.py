import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import numpy as np
from math import pi

def plot_histogram_by_measure(measure_type, measure_values, perf_dominance, bins=None, num_bins=None, exclude_bins=[], sig_digits=2, fontsize=18):
    """
    Plots a stacked histogram of perf_dominance divided by measure values.

    Parameters:
    - measure_type: str, the type of measure (e.g., 'cc_max', 'cc_norm')
    - measure_values: array-like, the values of the measure
    - perf_dominance: array-like, the perf_dominance values
    - bins: list, the bin edges for dividing measure_values (optional if num_bins is provided)
    - num_bins: int, number of equally spaced bins (optional if bins is provided)
    - exclude_bins: list, the indices of bins to exclude from the plot
    - sig_digits: int, number of significant digits for legend values (default is 2)
    """
    if num_bins is not None:
        # Generate equally spaced bins based on num_bins
        bins = np.linspace(min(measure_values), max(measure_values), num_bins + 1)
    
    if bins is None:
        raise ValueError("Either 'bins' or 'num_bins' must be provided.")
    
    # Create masks for each bin range
    masks = [(measure_values >= bins[i]) & (measure_values <= bins[i + 1]) if i == 0 else (measure_values > bins[i]) & (measure_values <= bins[i + 1]) for i in range(len(bins) - 1)]
    
    # Filter out the bins to exclude
    masks = [mask for i, mask in enumerate(masks) if i not in exclude_bins]
    labels = [f'{bins[i]:.{sig_digits}g}-{bins[i+1]:.{sig_digits}g}' for i in range(len(bins) - 1) if i not in exclude_bins]
    
    # Divide perf_dominance into groups based on the masks
    perf_dominance_groups = [perf_dominance[mask] for mask in masks]
    
    # Assign colors for each group
    colors = plt.cm.viridis(np.linspace(0, 1, len(perf_dominance_groups)))
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(perf_dominance_groups, bins=50, color=colors, edgecolor='black', alpha=0.7, stacked=True, label=labels, density=False)
    
    # Add labels and title
    plt.xlabel(f'Fraction of timepoints where LLAMA is better \n (w.r.t. |error|) (for each voxel)', fontsize=fontsize)
    # plt.ylabel('Density (%)', fontsize=fontsize)
    plt.ylabel('Number of voxels', fontsize=fontsize)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid()
    plt.xlim(0, 1)
    plt.legend(title=f'{measure_type} bins', loc='upper right', fontsize=fontsize, title_fontsize=fontsize)
    
    # Show plot
    plt.show()
# Example usage


def plot_baseline_value(criteria, whisper = False, vertical = False, ax_or_plt = None): 
    if whisper : 
        if criteria == 'mean_cc_norm_mean':
            baseline_value = 0.251981
        elif criteria == 'mean_r2_mean':
            baseline_value = 0.027710
        label = 'whisper_large_final'
    else : 
        if criteria == 'mean_cc_norm_mean':
            baseline_value = 0.291187
        elif criteria == 'mean_r2_mean':
            baseline_value = 0.036620
        label = 'LLAMA30B_18'

    if ax_or_plt is not None:
        if vertical:
            ax_or_plt.axvline(x=baseline_value, color='black', linestyle='--', label=label)
        else:
            ax_or_plt.axhline(y=baseline_value, color='black', linestyle='--', label=label)
    else:
        if vertical:
            plt.axvline(x=baseline_value, color='black', linestyle='--', label=label)
        else:
            plt.axhline(y=baseline_value, color='black', linestyle='--', label=label)


###some other stuff
def plot_scatter_with_regression(x_to_plot, y_to_plot, xlabel="cc_max", 
                                 ylabel = "$\Delta$ r", title="", ylim=(-0.3, 0.3), 
                                 xlim=(-.01, 1), show_mean_line=False, ax=None):
    plt.scatter(x_to_plot, y_to_plot, s=1)
    slope, intercept, r_value, p_value, std_err = linregress(x_to_plot, y_to_plot)
    plt.plot(x_to_plot, slope*x_to_plot + intercept, color='red', label=f"y={slope:.2e}x+{intercept:.2e}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ax is not None:
        plt.axhline(ax, color='black', linestyle='--', label=f"y={ax:.2e}")
    if show_mean_line:
        plt.axhline(y_to_plot.mean(), color='green', linestyle='--', label=f"mean={y_to_plot.mean():.2e}")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.show()

def plot_mean_in_bins_with_ci(x, y, num_bins=10, exclude_bins=[], sig_digits=2,
                              xlabel='x', ylabel='y', x_min=None, x_max=None,
                              ax=None, color=None, title=f'Mean of y values in each bin with 95% CI', label=None):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max
    bins = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
    
    masks = [(x >= bins[i]) & (x <= bins[i + 1]) if i == 0 else (x > bins[i]) & (x <= bins[i + 1]) for i in range(len(bins) - 1)]
    masks = [mask for i, mask in enumerate(masks) if i not in exclude_bins]
    bin_centers = [center for i, center in enumerate(bin_centers) if i not in exclude_bins]
    
    y_groups = [y[mask] for mask in masks]
    mean_values = [group.mean() for group in y_groups]
    ci_values = [1.96 * group.std() / np.sqrt(len(group)) for group in y_groups]  # 95% confidence interval
    
    if ax is None:
        ax = plt.gca()
    
    ax.errorbar(bin_centers, mean_values, yerr=ci_values, fmt='o', capsize=5, color=color, label=label)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y')
    ax.set_title(title, fontsize=18)
    
    return ax


def plot_mean_in_rois_with_ci(roi_dict, sig_digits=2, 
                              xlabel='ROI', ylabel='mean $\Delta$ r', 
                              ax=None, color=None, title = f'Mean of y values in each bin with 95% CI',label=None):
    roi_names = list(roi_dict.keys())
    mean_values = [np.mean(roi_dict[roi]) for roi in roi_names]
    ci_values = [1.96 * np.std(roi_dict[roi]) / np.sqrt(len(roi_dict[roi])) for roi in roi_names]  # 95% confidence interval
    
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.plot(roi_names, mean_values, color=color, label=label)
    ax.fill_between(roi_names, np.subtract(mean_values, ci_values), np.add(mean_values, ci_values), color=color, alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=90, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(title, fontsize=18)
    
    # Draw vertical lines from each tick
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='gray', linestyle='--', linewidth=0.3)
    
    if ax is None:
        plt.show()

def plot_mean_in_rois_with_ci_radar(roi_dict, sig_digits=2, 
                                    xlabel='ROI', ylabel='mean $\Delta$ r', 
                                    ax=None, color=None, title='Mean of y values in each bin with 95% CI', label=None):
    # Extract ROI names, mean values, and confidence intervals
    roi_names = list(roi_dict.keys())
    mean_values = [np.mean(roi_dict[roi]) for roi in roi_names]
    ci_values = [1.96 * np.std(roi_dict[roi]) / np.sqrt(len(roi_dict[roi])) for roi in roi_names]  # 95% CI

    # Number of variables
    num_vars = len(roi_names)

    # The radar chart requires the variables to be arranged in a circle, so we repeat the first value at the end
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop for radar

    # Append the first value to the end for a complete radar plot
    mean_values += mean_values[:1]
    ci_values += ci_values[:1]  # Fix: Append first CI value to close the loop

    # Compute the upper and lower bounds for the CI
    ci_values_lower = np.subtract(mean_values, ci_values)
    ci_values_upper = np.add(mean_values, ci_values)

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot the mean values
    ax.plot(angles, mean_values, color=color, linewidth=2, linestyle='solid', label=label)
    ax.fill_between(angles, ci_values_lower, ci_values_upper, color=color, alpha=0.4)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])  # Skip the last repeated angle
    ax.set_xticklabels(roi_names, fontsize=12)

    # Optional labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16, labelpad=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16, labelpad=30)
    ax.set_title(title, size=18, y=1.1)

    # Optionally show the plot if ax is not provided
    if ax is None:
        plt.show()




##without fill_between        
""" 
def plot_mean_in_rois_with_ci(roi_dict, sig_digits=2, 
                              xlabel='ROI', ylabel='mean $\Delta$ r', 
                              ax=None, color=None, capsize=5, title = f'Mean of y values in each bin with 95% CI',label=None):
    roi_names = list(roi_dict.keys())
    mean_values = [np.mean(roi_dict[roi]) for roi in roi_names]
    ci_values = [1.96 * np.std(roi_dict[roi]) / np.sqrt(len(roi_dict[roi])) for roi in roi_names]  # 95% confidence interval
    
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.errorbar(roi_names, mean_values, yerr=ci_values, fmt='o', color=color, capsize=capsize, label = label)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=90, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    # ax.grid(axis='y')
    ax.set_title(title, fontsize=18)
    
    # Draw vertical lines from each tick
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='light gray', linestyle='--', linewidth=0.3)
    
    if ax is None:
        plt.show()
"""