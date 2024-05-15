import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd

custom_colors = {"S11_RA110_S3_32": 'red',
                 "S3_RA110_S1_11": 'blue',
                 "S3_RA110_S1_12": "orange",
                 "S5_RA001_S310": "green",
                 "S5_RA001_S210": "purple",
                 "S9_RA110_S2_21": "brown"}

# Adjusting the first plot with specified font sizes
gb_latex_dict = {
    "S5_RA001_S310": r'$\Sigma5$[001]$(310)$',
    "S5_RA001_S210": r'$\Sigma5$[001]$(210)$',
    "S11_RA110_S3_32": r'$\Sigma11$[110]$(3\overline{3}2)$',
    "S3_RA110_S1_12": r'$\Sigma3$[110]$(1\overline{1}2)$',
    "S3_RA110_S1_11": r'$\Sigma3$[110]$(1\overline{1}1)$',
    "S9_RA110_S2_21": r'$\Sigma9$[110]$(2\overline{2}1)$'
}

bulk_df = pd.read_csv("bulk_df.csv")

def plot_minEseg_prop(df,
                      y_prop="E_seg",
                      x_prop="Z",
                      ylabel=r"$\rm{min}(E_{\rm{seg}})$ (eV)",
                      figsize=(20, 12),
                      shift_xticks=False,
                      xlabel_fontsize=24,
                      xtick_yshift=0,
                      ylabel_fontsize=24,
                      xtick_fontsize=20,
                      ytick_fontsize=24,
                      legend_fontsize=20,
                      xtick_labels = bulk_df.element.values,
                      xtick_posns = bulk_df.Z.values):
    # Create a plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Looping over each unique "GB" group
    for idx, (gb, group) in enumerate(df.dropna(subset=[y_prop]).groupby("GB")):
        color = custom_colors[gb]  # Custom color for each group

        Eseg_col = "E_seg"
        # For each "GB" group, find the minimum "E_seg"
        min_eseg_per_element = group.groupby("element").apply(lambda x: x.nsmallest(1, Eseg_col).iloc[0])
        min_eseg_per_element = min_eseg_per_element[min_eseg_per_element[Eseg_col] <= 0]
        # Sorting values by 'Z' for consistent plotting
        min_eseg_per_element = min_eseg_per_element.sort_values(by='Z')

        # Plotting
        x_values = min_eseg_per_element[x_prop]
        y_values = min_eseg_per_element[y_prop]

        line1, = ax1.plot(x_values, y_values, color=color, linestyle='--', marker="o", linewidth=3, markersize=6)
        ax1.axhline(0, color=color)

        # Creating legends
        gb_legends = [(line1, f'{gb_latex_dict[gb]}') for gb in custom_colors]

    # Set labels and axis
    ax1.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if xtick_labels is not None:
        ax1.set_xticks(xtick_posns)
        ax1.set_xticklabels(xtick_labels, fontsize=xtick_fontsize, rotation=90,va='center')
    ax1.tick_params(axis='y', labelsize=ytick_fontsize)

    if shift_xticks:
        shifts = [-0.01, 0.04, 0.09, 0.14]  # Define y-shift values for three lines
        shifts = [-0.01, 0.04, 0.09]  # Define y-shift values for three lines
        for i, label in enumerate(ax1.get_xticklabels()):
            label.set_y(shifts[i % 3] + xtick_yshift)
    # Manually adding gridlines at specified intervals (1, 4, 7, 10, ..., up to 92)
    ax1.grid(False)  # Turn off grid
    gridline_positions = np.arange(1, 93, 3)  # Generate positions
    # Draw vertical lines for specified positions
    for pos in gridline_positions:
        ax1.axvline(x=pos, linestyle='-', linewidth='0.5', color='grey', alpha=0.75)  # Adjust alpha for visibility if needed

    # Creating a custom legend
    gb_legend = plt.legend(*zip(*gb_legends), bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=legend_fontsize)
    plt.gca().add_artist(gb_legend)

    return fig, ax1


def plot_pivot_table(df,
                 colormap_thresholds=[None, None],
                 figsize=(18, 30),
                 colormap='bwr',
                 colormap_label='E$_{\\rm{seg}}$ (eV)',
                 color_label_fontsize=20,
                 colormap_tick_fontsize=12,
                 xtick_fontsize=18,
                 ytick_fontsize=12,
                 threshold_low=None,
                 threshold_high=None,
                 transpose_axes=False):
    """
    Plot a heatmap with custom parameters.

    Parameters:
    - df: DataFrame to plot.
    - colormap_thresholds: List with [vmin, vmax] for the colormap, default is [-1, 1].
    - figsize: Tuple for figure size (width, height), default is (18, 30).
    - colormap: String for the colormap name, default is 'bwr'.
    - colormap_label: Label for the colorbar, default is 'E$_{\\rm{seg}}$ (eV)'.
    - fontsize: Font size for the colorbar label, default is 20.
    - xtick_fontsize: Font size for x-axis tick labels, default is 18.
    - ytick_fontsize: Font size for y-axis tick labels, default is 12.
    - threshold_low: Lower threshold to filter data; defaults to None.
    - threshold_high: Higher threshold to filter data; defaults to None.
    - transpose_axes: Boolean to transpose the DataFrame; defaults to False.
    """
    if threshold_low is not None or threshold_high is not None:
        df = df.copy()
        df[(df < threshold_low) | (df > threshold_high)] = np.nan
    
    if transpose_axes:
        df = df.T

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    cmap = plt.cm.get_cmap(colormap)
    cmap.set_bad('k')
    if colormap_thresholds == [None, None]:
        vmax = max(abs(np.nanmin(df.max())), abs(np.nanmin(df.min())))
        vmin = -vmax
    else:
        vmin, vmax = colormap_thresholds
    im = axs.imshow(df, cmap=cmap, vmax=vmax, vmin=vmin)
    cm = plt.colorbar(im, ax=axs, shrink=0.3, location='right', pad=0.01)
    cm.set_label(colormap_label, rotation=270, labelpad=15, fontsize=color_label_fontsize)
    # cm.ax.tick_params(labelsize=colormap_tick_fontsize)  # Set colorbar tick label size
    if colormap_thresholds != [None, None]:
        ticks = cm.get_ticks()
        if len(ticks) > 1:  # Check to ensure there are ticks to modify
            tick_labels = [f"$<{vmin}$" if i == 0 else f"$>{vmax}$" if i == len(ticks)-1 else str(tick) for i, tick in enumerate(ticks)]
            cm.set_ticks(ticks)  # Set the ticks back if they were changed
            cm.set_ticklabels(tick_labels, fontsize=colormap_tick_fontsize)  # Set the modified tick labels
    else:
        cm.set_ticklabels(cm.get_ticks(), fontsize=colormap_tick_fontsize)  # Set the modified tick labels

    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=0, fontsize=xtick_fontsize)
    plt.yticks(np.arange(len(df.index)), df.index, fontsize=ytick_fontsize)

    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axs.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    axs.tick_params(axis='both', which='major', width=1.5, length=4)
    axs.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    return fig, axs