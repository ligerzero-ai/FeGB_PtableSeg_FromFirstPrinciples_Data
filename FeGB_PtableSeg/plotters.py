import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches

import numpy as np
import pandas as pd

import os

from pymatgen.core import Structure

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

gb_marker_dict = {
    "S3_RA110_S1_11": '*',
    "S3_RA110_S1_12": 'P',
    "S5_RA001_S210": '^',
    "S5_RA001_S310": 'o',
    "S9_RA110_S2_21": 'X',
    "S11_RA110_S3_32": 's',
}

module_path = os.path.dirname(os.path.abspath(__file__))
bulk_df = pd.read_csv(os.path.join(module_path, 'bulk_df.csv'))

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

def periodic_table_plot(plot_df, 
                        property="Eseg_min",
                        count_min=None,
                        count_max=None,
                        center_cm_zero=False,
                        center_point=None,  # New parameter for arbitrary centering
                        property_name=None,
                        cmap=cm.Blues,
                        element_font_color = "darkgoldenrod"
):
    module_path = os.path.dirname(os.path.abspath(__file__))
    bulk_df = pd.read_csv(os.path.join(module_path, 'bulk_df.csv'))
    bulk_df.index = bulk_df['element'].values
    #elem_tracker = bulk_df['count']
    bulk_df = bulk_df[bulk_df['Z'] <= 92]  # Cap at element 92

    n_row = bulk_df['row'].max()
    n_column = bulk_df['column'].max()

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = bulk_df['row']
    columns = bulk_df['column']
    symbols = bulk_df['element']
    rw = 0.9  # rectangle width
    rh = rw    # rectangle height

    if count_min is None:
        count_min = plot_df[property].min()
    if count_max is None:
        count_max = plot_df[property].max()

    # Adjust normalization based on centering preference
    if center_cm_zero:
        cm_threshold = max(abs(count_min), abs(count_max))
        norm = Normalize(-cm_threshold, cm_threshold)
    elif center_point is not None:
        # Adjust normalization to center around the arbitrary point
        max_diff = max(center_point - count_min, count_max - center_point)
        norm = Normalize(center_point - max_diff, center_point + max_diff)
    else:
        norm = Normalize(vmin=count_min, vmax=count_max)

    for row, column, symbol in zip(rows, columns, symbols):
        row = bulk_df['row'].max() - row
        if symbol in plot_df.element.unique():
            count = plot_df[plot_df["element"] == symbol][property].values[0]
            # Check for NaN and adjust color and skip text accordingly
            if pd.isna(count):
                color = 'grey'  # Set color to none for NaN values
                count = ''  # Avoid displaying text for NaN values
            else:
                color = cmap(norm(count))
        else:
            count = ''
            color = 'none'

        if row < 3:
            row += 0.5
        rect = patches.Rectangle((column, row), rw, rh,
                                linewidth=1.5,
                                edgecolor='gray',
                                facecolor=color,
                                alpha=1)

        # Element symbol
        plt.text(column + rw / 2, row + rh / 2 + 0.2, symbol,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=22,  # Adjusted for visibility
                fontweight='semibold',
                color=element_font_color)

        # Property value - Added below the symbol
        if count:  # Only display if count is not empty (including not NaN)
            plt.text(column + rw / 2, row + rh / 2 - 0.25, f"{count:.2f}",  # Formatting count to 2 decimal places
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,  # Smaller font size for the count value
                    fontweight='semibold',
                    color=element_font_color)

        ax.add_patch(rect)
    # Generate the color bar
    granularity = 20
    colormap_array = np.linspace(norm.vmin, norm.vmax, granularity) if center_point is None else np.linspace(center_point - max_diff, center_point + max_diff, granularity)
    
    for i, value in enumerate(colormap_array):
        color = cmap(norm(value))
        color = 'silver' if value == 0 else color
        length = 9
        x_offset = 3.5
        y_offset = 7.8
        x_loc = i / granularity * length + x_offset
        width = length / granularity
        height = 0.35
        rect = patches.Rectangle((x_loc, y_offset), width, height,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)

        if i in [0, granularity//4, granularity//2, 3*granularity//4, granularity-1]:
            plt.text(x_loc + width / 2, y_offset - 0.4, f'{value:.1f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='semibold',
                     fontsize=20, color='k')

        ax.add_patch(rect)

    if property_name is None:
        property_name = property
    plt.text(x_offset + length / 2, y_offset + 1.0,
             property_name,
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='semibold',
             fontsize=20, color='k')
    ax.set_ylim(-0.15, n_row + .1)
    ax.set_xlim(0.85, n_column + 1.1)

    ax.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.close()
    return fig, ax

def get_GB_area(df_seg, GB):
    df_GB = df_seg[df_seg["GB"] == GB]
    struct = Structure.from_str(df_GB.structure.iloc[0], fmt="json")
    area  = struct.volume/struct.lattice.c*0.01 # Area in nm^2
    return area

def plot_coverage_vs_temperature(df,
                                 df_spectra,
                                 alloy_conc,
                                 temperature_range,
                                 element,
                                 close_fig=True,
                                 save_path="/mnt/c/Users/liger/Koofr/Fe-PtableTrends-Manuscript/Figures",
                                 xlims=None,
                                 ylims=None,
                                 figsize=(12, 9)):
    fig, ax1 = plt.subplots(figsize=figsize)
    legend_elements = []  # List to hold the legend handles
    df_ele = df_spectra[df_spectra["element"] == element]
    plot_data = []

    # Add a special legend entry for headers
    header_label = "      GB            min(E$_{\mathrm{seg}}$)"
    legend_elements.append(plt.Line2D([0], [0], color='none', label=header_label))

    for GB, GB_df in df_ele.groupby(by="GB"):
        temp_concs = []
        for temp in temperature_range:
            site_concentrations = calc_C_GB(temp, alloy_conc * 0.01, np.array(GB_df.total_spectra.values[0]))
            temp_concs.append(site_concentrations.sum() / get_GB_area(df, GB=GB))
        plot_data.append((temperature_range, temp_concs, GB))
        
        min_total_spectra = min(GB_df.total_spectra.values[0])
        # Format the label to just include the GB, no min(E_seg)
        min_total_spectra = min(GB_df.total_spectra.values[0])
        label = f"{gb_latex_dict[GB]} {min_total_spectra:.2f} eV"
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                          markerfacecolor=custom_colors[GB], markersize=15, linestyle='None'))
    legend_elements.append(legend_elements.pop(1))  # Adjust order to keep header at the top

    # Plot data
    for data in plot_data:
        color = custom_colors[data[2]]
        ax1.plot(data[0], data[1], linewidth=8, color=color)

    # Axis settings and text
    ax1.set_xlabel("Temperature (K)", fontsize=24)
    ax1.set_ylabel("Atomic coverage (atoms/nm$^2$)", fontsize=24)
    ax1.tick_params(labelsize=24)
    ax1.grid()

    # Set xlims and ylims if specified
    if xlims is not None:
        ax1.set_xlim([max(0, xlims[0]), xlims[1] if xlims[1] is not None else ax1.get_xlim()[1]])
    else:
        ax1.set_xlim(left=0)

    if ylims is not None:
        ax1.set_ylim([max(0, ylims[0]), ylims[1] if ylims[1] is not None else ax1.get_ylim()[1]])
    else:
        ax1.set_ylim(bottom=0)

    # Place text for element and bulk concentration
    ax1.text(0.05, 0.05, f'{element}\n{alloy_conc:.2f} at.%', transform=ax1.transAxes, fontsize=40, verticalalignment='bottom', horizontalalignment='left')

    # Legend setup
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor = (0.99, 0.99), fontsize=22, frameon=True, handletextpad=0.01, borderpad=0.05, labelspacing=0.3)

    # Additional x-axis for Celsius
    kelvin_ticks = [73, 273, 473, 673, 873, 1073, 1273]
    celsius_ticks = [k - 273 for k in kelvin_ticks]
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(kelvin_ticks)
    ax2.set_xticklabels([f"{c}°C" for c in celsius_ticks])
    ax2.set_xlabel("Temperature (°C)", fontsize=24)
    ax2.tick_params(axis='x', labelsize=24)

    # Save and close figure
    # fig.savefig(f'{save_path}/WhiteCoghlan/WhiteCoghlan_{element}_{alloy_conc}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    if close_fig:
        plt.close(fig)  # Close the figure to avoid display
        
    return fig, ax1

def calc_C_GB(Temperature,c_bulk,E_seg):
    """
    Calculate the grain boundary concentration.

    :param Temperature: Temperature in Kelvin
    :param c_bulk: Bulk concentration
    :param E_seg: Segregation energy array for a given element at different sites
    :return: Grain boundary concentration
    """
    c_GB = np.divide(c_bulk * np.exp(-E_seg/(8.6173303e-05*Temperature))\
        ,(1 - c_bulk + c_bulk * np.exp(-E_seg/(8.6173303e-05*Temperature))))
    return c_GB
class GB_symmetries():
    def __init__(self):
        # S3 S111
        studied_list = [20, 22, 24, 26, 28, 30, 32, 34, 36]
        # 0.5-1ML available
        symmetry = [[21, 52, 53],\
                    [23, 50, 51],\
                    [25, 48, 49],\
                    [27, 46, 47],\
                    [29, 44, 45],\
                    [31, 42, 43],\
                    [33, 40, 41],\
                    [35, 38, 39],\
                    [37]]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S3_1_symmetrydict = dict(zip(studied_list,symmetry))
        
        # S3 S112
        studied_list = [12, 14, 16, 18, 20, 22, 24]
        # 0.5-1ML available
        symmetry = [[13, 36, 37],\
                    [15, 34, 35],\
                    [17, 32, 33],\
                    [19, 30, 31],\
                    [21, 28, 29],\
                    [23, 26, 27],\
                    [25]]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S3_2_symmetrydict = dict(zip(studied_list,symmetry))
        
        # S9
        studied_list = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # only 0-1 ML available
        symmetry = [[47],\
                [46],\
                [45],\
                [44],\
                [43],\
                [42],\
                [41],\
                [40],\
                [39],\
                [38],\
                [37],\
                [],\
                [],\
                []]
        # When the site is on the GB plane, we don't need to calculate values on both sides
        self.S9_symmetrydict = dict(zip(studied_list,symmetry))
        
        # S11
        studied_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # only 0-1 ML available
        symmetry = [[32],\
                    [31],\
                    [30],\
                    [29],\
                    [28],\
                    [27],\
                    [26],\
                    [25],\
                    [24],\
                    [23],\
                    [],\
                    []]
        self.S11_symmetrydict = dict(zip(studied_list,symmetry))
        
        #S5 210
        studied_list = [24, 27, 29, 31, 33, 35, 37]
        # ML not well defined, should be 0,0.5,1 ML but middle plane has two inequivalent sites
        symmetry = [[25] + [46, 47],\
                    [26] + [44, 45],\
                    [28] + [42, 43],\
                    [30] + [40, 41],\
                    [32] + [38, 39],\
                    [34],\
                    [36]]
        self.S5_2_symmetrydict = dict(zip(studied_list,symmetry))

        # S5 310
        # 0/0.25/0.5/0.75/1 ML
        studied_list = [23, 27, 33, 37, 40]
        symmetry = [[22, 24, 25] + [54, 55, 56, 57],\
                    [26, 28, 29] + [50, 51 ,52, 53],\
                    [30, 31, 32] + [46, 47, 48, 49],\
                    [34, 35, 36] + [42, 43, 44, 45],\
                    [38, 39, 41]]
        self.S5_3_symmetrydict = dict(zip(studied_list,symmetry))
        
        name_list = ['S9_RA110_S2_21',
        'S11_RA110_S3_32',
        'S3_RA110_S1_12',
        'S3_RA110_S1_11',
        'S5_RA001_S310',
        'S5_RA001_S210']
        
        self.symmetry_dict_all = dict(zip(name_list,
                                          [self.S9_symmetrydict,
                                           self.S11_symmetrydict,
                                           self.S3_2_symmetrydict,
                                           self.S3_1_symmetrydict,
                                           self.S5_3_symmetrydict,
                                           self.S5_2_symmetrydict])
                                     )

def plot_x_y_whist_spectra(df, x="R_wsep_lst", y="R_ANSBO_lst",
                           xlabel=r"$\rm{R}_{\rm{W_{\rm{sep}}}}$", ylabel=r"$\rm{R}_{\rm{ANSBO}}$",
                           xlabel_fontsize=24, ylabel_fontsize=24, legend_fontsize=12,
                           bin_width_x=0.02, bin_width_y=0.02, close_fig=True, mask_limits=None,
                           hist_ticksize=20, scatter_ticksize=20, title=None, title_fontsize=24):
    fig = plt.figure(figsize=(10, 10))
    ax_scatter = plt.axes([0.1, 0.1, 0.65, 0.65])
    ax_histx = plt.axes([0.1, 0.77, 0.65, 0.2], sharex=ax_scatter)
    ax_histy = plt.axes([0.77, 0.1, 0.2, 0.65], sharey=ax_scatter)

    ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    all_x = []
    all_y = []
    
    df[f"full_multiplicity_{x}"] = [[row[x]] * row.site_multiplicity for _, row in df.iterrows()]
    df[f"full_multiplicity_{y}"] = [[row[y]] * row.site_multiplicity for _, row in df.iterrows()]

    # Collect all data first to define global bin edges
    for gb_type, marker in gb_marker_dict.items():
        gb_df = df[df['GB'] == gb_type].dropna()

        plot_x = np.concatenate(gb_df[f"full_multiplicity_{x}"].explode().dropna().apply(np.ravel).tolist())
        plot_y = np.concatenate(gb_df[f"full_multiplicity_{y}"].explode().dropna().apply(np.ravel).tolist())

        if mask_limits:
            mask = (plot_x >= mask_limits[0]) & (plot_x <= mask_limits[1]) & (plot_y >= mask_limits[0]) & (plot_y <= mask_limits[1])
            plot_x = plot_x[mask]
            plot_y = plot_y[mask]

        all_x.extend(plot_x)
        all_y.extend(plot_y)

    # Calculate bins for x and y histograms with alignment logic
    min_x = np.floor(min(all_x) / bin_width_x) * bin_width_x
    max_x = np.ceil(max(all_x) / bin_width_x) * bin_width_x
    binsx = np.arange(min_x, max_x + bin_width_x, bin_width_x)
    
    min_y = np.floor(min(all_y) / bin_width_y) * bin_width_y
    max_y = np.ceil(max(all_y) / bin_width_y) * bin_width_y
    binsy = np.arange(min_y, max_y + bin_width_y, bin_width_y)

    # Initialize bottom arrays for stacked bars
    bottom_x = np.zeros(len(binsx) - 1)
    bottom_y = np.zeros(len(binsy) - 1)

    # Re-loop to plot
    for gb_type, marker in gb_marker_dict.items():
        plot_x = np.concatenate(df[df['GB'] == gb_type][f"full_multiplicity_{x}"].explode().dropna().apply(np.ravel).tolist())
        plot_y = np.concatenate(df[df['GB'] == gb_type][f"full_multiplicity_{y}"].explode().dropna().apply(np.ravel).tolist())
        
        if mask_limits:
            mask = (plot_x >= mask_limits[0]) & (plot_x <= mask_limits[1]) & (plot_y >= mask_limits[0]) & (plot_y <= mask_limits[1])
            plot_x = plot_x[mask]
            plot_y = plot_y[mask]

        counts_x, _ = np.histogram(plot_x, bins=binsx)
        counts_y, _ = np.histogram(plot_y, bins=binsy)

        ax_scatter.scatter(plot_x, plot_y, alpha=0.9, marker=marker, s=200, label=gb_latex_dict[gb_type], c=custom_colors[gb_type])

        ax_histx.bar(binsx[:-1], counts_x, width=np.diff(binsx), bottom=bottom_x, align='edge', label=gb_type, color=custom_colors[gb_type])
        bottom_x += counts_x
        ax_histy.barh(binsy[:-1], counts_y, height=np.diff(binsy), left=bottom_y, align='edge', label=gb_type, color=custom_colors[gb_type])
        bottom_y += counts_y

    # Extend axes limits by adding/subtracting one bin width
    ax_scatter.set_xlim(min_x , max_x)
    ax_scatter.set_ylim(min_y , max_y)
    
    ax_scatter.tick_params(axis='both', which='major', labelsize=scatter_ticksize)
    ax_histx.tick_params(axis='both', which='major', labelsize=hist_ticksize)
    ax_histy.tick_params(axis='both', which='major', labelsize=hist_ticksize)

    ax_scatter.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax_scatter.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax_scatter.legend(bbox_to_anchor=(1.02, 1.02), loc="lower left", fontsize=legend_fontsize)
    ax_scatter.grid(which="both", alpha=0.5)
    # Add user-specified title on the top left corner
    if title:
        ax_scatter.text(0.01, 0.99, title, transform=ax_scatter.transAxes, fontsize=title_fontsize, verticalalignment='top', horizontalalignment='left')

    if close_fig:
        plt.close(fig)

    return fig, ax_scatter

## Figure 10
def plot_property(df,
                  x_prop,
                  y_prop,
                  figsize=(20,16),
                  x_label = r"E$_{\rm{seg}}$ (eV)",
                  y_label = r"W$_{\rm{sep}}^{\rm{RGS}}$ (J/m$^2$)",
                  element_groups = None,
                  text_labels = None,
                  legend_posn = (0.001,0.681),
                  xlims=None,  # New parameter for x-axis limits
                  ylims=None,  # New parameter for y-axis limits
                  savefig_path = None):
    
    fig = plt.figure(figsize=figsize)
    ax1 = plt.gca()
    gb_legends = []

    df_plt = df.dropna(subset=[y_prop])
    if element_groups is not None:
        df_plt = df_plt[df_plt["ele_group"].isin(element_groups)]

    # df_plt = df_plt[df_plt["eta_Wsep_RGS"] > 0]
    # Looping over each unique "GB" group
    for idx, (gb, group) in enumerate(df_plt.dropna(subset=[y_prop]).groupby("GB")):
        # Assigning custom color for each group
        color = custom_colors[gb]

        # Plotting
        x_values = group[x_prop]  # "Z" values for x-axis
        y_values = group[y_prop]  # Corresponding "min_wsep_rigid" values
        elements = group.element  # Element values for text labels

        # Scatter plot for "min_wsep_rigid" on the primary y-axis
        line1 = ax1.scatter(x_values, y_values, color=color, marker="x", s=100, linewidths=3, alpha=1.0)
        
        if text_labels is not None:
            # Adding text labels for each marker
            for x, y, element in zip(x_values, y_values, elements):
                ax1.text(x, y, element, color=color, fontsize=20)

        # Creating legends
        gb_legends.append((line1, f'{gb_latex_dict[gb]}'))
        # ax1.axhline(df_coh_sub[df_coh_sub["GB"] == gb].iloc[0].pure_Wsep_min, color=color, linewidth=2, linestyle="--")
    
    gb_legends.append(gb_legends.pop(0))
    # Modify marker size in legend by plotting empty lists
    for line, label in gb_legends:
        plt.scatter([], [], s=200, color=line.get_facecolor()[0], label=label)

    gb_legend = plt.legend(fontsize=30,
                           loc="lower left",
                           bbox_to_anchor=legend_posn,
                           scatterpoints=1,
                           frameon=True,
                           handletextpad=0.1, # Reduces space between the marker and text
                            borderpad=0.1,    # Reduces space between the text and legend border
                            labelspacing=0.15)

    ax1.tick_params(axis='y', labelsize=40, rotation=90)
    ax1.tick_params(axis='x', which='both', labelsize=40)
    ax1.grid(which="both")

    plt.xlabel(x_label, fontsize=40)
    plt.ylabel(y_label, fontsize=40)
    
    # Setting x and y limits if provided
    if xlims is not None:
        ax1.set_xlim(xlims)
    if ylims is not None:
        ax1.set_ylim(ylims)
        
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0.1)
    return fig, ax1