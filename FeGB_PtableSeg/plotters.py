import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors  # Import for LogNorm

import numpy as np
import pandas as pd

import warnings
import os

from pymatgen.core import Structure
from pymatgen.core import Element

module_path = os.path.dirname(os.path.abspath(__file__))
ptable = pd.read_csv(os.path.join(module_path, "bulk_df.csv"))

def get_element_number(symbol):
    try:
        return Element(symbol).Z
    except ValueError:
        warnings.warn(f"Warning: Symbol '{symbol}' was not found.")
        return np.nan


def get_element_symbol(element_number):
    row = ptable[ptable["Z"] == element_number]
    if not row.empty:
        return row["element"].values[0]
    else:
        warnings.warn(f"Warning: Element with Z:{element_number} was not found.")
        return np.nan

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

module_path = os.path.dirname(os.path.abspath(__file__))
bulk_df = pd.read_csv(os.path.join(module_path, 'bulk_df.csv'))


#%% Fig 3
def plot_minEseg_prop_vs_Z(df,
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

    # To store legend handles for each GB
    legend_handles = []

    # Looping over each unique "GB" group
    for gb, group in df.dropna(subset=[y_prop]).groupby("GB"):
        color = custom_colors.get(gb, 'black')  # Get color for each group, default to 'black' if not found

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

        # Collect legend entries if not already added
        if gb not in [h.get_label() for h in legend_handles]:  # Avoid duplicates
            legend_handles.append(line1)

    # Set labels and axis
    ax1.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if xtick_labels is not None:
        ax1.set_xticks(xtick_posns)
        ax1.set_xticklabels(xtick_labels, fontsize=xtick_fontsize, rotation=90, va='center')
    ax1.tick_params(axis='y', labelsize=ytick_fontsize)

    if shift_xticks:
        shifts = [-0.01, 0.04, 0.09]  # Define y-shift values for three lines
        for i, label in enumerate(ax1.get_xticklabels()):
            label.set_y(shifts[i % 3] + xtick_yshift)

    # Manually adding gridlines at specified intervals (1, 4, 7, 10, ..., up to 92)
    ax1.grid(False)  # Turn off grid
    gridline_positions = np.arange(1, 93, 3)  # Generate positions
    # Draw vertical lines for specified positions
    for pos in gridline_positions:
        ax1.axvline(x=pos, linestyle='-', linewidth='0.5', color='grey', alpha=0.75)  # Adjust alpha for visibility if needed

    # Creating a custom legend with unique handles and labels, moving the first entry to the end
    gb_legend_labels = [gb_latex_dict.get(gb, gb) for gb in custom_colors.keys()]

    # Rearrange legend handles and labels
    legend_handles = legend_handles[1:] + legend_handles[:1]
    gb_legend_labels = gb_legend_labels[1:] + gb_legend_labels[:1]

    plt.legend(legend_handles, gb_legend_labels, bbox_to_anchor=(0.63, 0.05), loc="lower left", fontsize=legend_fontsize)

    return fig, ax1

#%% Fig 5

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

#%% Fig 6

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
        # print(get_GB_area(df, GB=GB))
        for temp in temperature_range:
            site_concentrations = calc_C_GB(temp, alloy_conc * 0.01, np.array(GB_df.full_seg_spectra.values[0]))
            temp_concs.append(site_concentrations.sum() / get_GB_area(df, GB=GB))
        plot_data.append((temperature_range, temp_concs, GB))
        min_total_spectra = min(GB_df.full_seg_spectra.values[0])
        # Format the label to just include the GB, no min(E_seg)
        min_total_spectra = min(GB_df.full_seg_spectra.values[0])
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
    ax2.set_xticklabels([f"{c}" for c in celsius_ticks])
    ax2.set_xlabel("Temperature (°C)", fontsize=24)
    ax2.tick_params(axis='x', labelsize=24)

    # Save and close figure
    # fig.savefig(f'{save_path}/WhiteCoghlan/WhiteCoghlan_{element}_{alloy_conc}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    if close_fig:
        plt.close(fig)  # Close the figure to avoid display
        
    return fig, ax1
#%% Fig 7a
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
    cmap = plt.get_cmap(colormap)
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

def get_GB_area(df_seg, GB):
    df_GB = df_seg[df_seg["GB"] == GB]
    struct = Structure.from_str(df_GB.structure.iloc[0], fmt="json")
    area  = struct.volume/struct.lattice.c*0.01 # Area in nm^2
    return area

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

def plot_interfacial_coverage_vs_minEseg(df, df_spectra, elements_to_plot, atomic_pct_conc=0.1, temp=300, nan_segregation=[], gb_latex_dict={}, custom_colors={}):
    df_copy = df_spectra.copy()

    interf_conc_300 = []
    interf_conc_300_lst = []


    for _, row in df_copy.iterrows():
        interfactial_conc_persite = calc_C_GB(temp, atomic_pct_conc*0.01, np.array(row.full_seg_spectra))
        interfacial_conc = interfactial_conc_persite.sum() / get_GB_area(df, row.GB)
        interf_conc_300.append(interfacial_conc)
        interf_conc_300_lst.append(interfactial_conc_persite)

    df_copy["interf_conc_300"] = interf_conc_300
    df_copy["interf_conc_300_persite"] = interf_conc_300_lst
    df_copy["min_Eseg"] = df_copy.full_seg_spectra.apply(min)
    df_copy["GB_element"] = [f"{row.GB}_{int(row.Z)}" for _, row in df_copy.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 9))

    for gb, gb_df in df_copy[df_copy["element"].isin(elements_to_plot)].groupby("GB"):
        plot_df = gb_df[gb_df["min_Eseg"] > -1]
        plot_df = plot_df[plot_df["min_Eseg"] < 0]
        plot_df = plot_df[~plot_df["GB_element"].isin(nan_segregation)]
        ax.scatter(plot_df["min_Eseg"], plot_df["interf_conc_300"], label=gb_latex_dict.get(gb, gb), c=custom_colors.get(gb, 'blue'), s=100)

        # Add text labels for each point
        for idx, row in plot_df.iterrows():
            ax.text(row["min_Eseg"], row["interf_conc_300"], row["element"], fontsize=20)

    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Move the first legend entry to the end
    handles.append(handles.pop(0))
    labels.append(labels.pop(0))

    # Create the new legend
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.01, 1.015), fontsize=16, handletextpad=0.1, borderpad=0.2)

    ax.set_ylabel(r"Interfacial coverage (atoms/nm$^2$)", fontsize=24)
    ax.set_xlabel(r"min(E$_{\rm{seg}}$) (eV)", fontsize=24)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    ax.text(0.0, 0.0, f"T: {temp} K\n" + r"C$_{\rm{b}}$:" + f" {atomic_pct_conc} at.%", 
            transform=ax.transAxes, fontsize=30, verticalalignment='bottom', horizontalalignment="left")
    ax.set_ylim([0, None])
    ax.set_xlim([None, 0])
    return fig, ax

#%%% Fig 7b
def calculate_effective_temperature_eseg(y, T, cB, kB=8.6173303e-05):
    """
    Calculate the effective segregation energy Eseg of the spectra from the given parameters.

    Parameters:
    y (float): The term from the equation.
    T (float): The temperature in Kelvin.
    cB (float): The concentration of B.
    kB (float): The Boltzmann constant, default 1.38e-23 J/K.

    Returns:
    float: The calculated segregation energy Eseg in Joules.
    """
    if y * cB - cB == 0:
        return np.nan
        #raise ValueError("The denominator becomes zero, adjust your input values.")
    
    numerator = y * cB - y
    denominator = y * cB - cB
    Eseg = -kB * T * np.log(numerator / denominator)
    return Eseg

def plot_Eseg_vs_temperature(df_spectra, element_to_plot, gb_latex_dict, custom_colors, alloy_conc=0.01 * 0.087, temp_range=(100, 1000), temp_step=20, legend_loc='center left', legend_bbox_to_anchor=None):
    temperatures = np.arange(temp_range[0], temp_range[1] + temp_step, temp_step)
    
    df_ele = df_spectra[df_spectra["element"] == element_to_plot]
    fig, ax1 = plt.subplots(figsize=(12, 9))

    legend_elements = []  # Moved outside the loop to accumulate all entries

    for GB, GB_df in df_ele.groupby(by="GB"):
        temp_effective_Esegs = []
        plot_data = []

        for temp in temperatures:
            site_concentrations = calc_C_GB(temp, alloy_conc, np.array(GB_df.full_seg_spectra.values[0]))
            temp_effective_Eseg = calculate_effective_temperature_eseg(site_concentrations.mean(), T=temp, cB=alloy_conc)
            temp_effective_Esegs.append(temp_effective_Eseg)

        plot_data.append((temperatures, temp_effective_Esegs, GB))

        for data in plot_data:
            color = custom_colors.get(data[2], 'blue')
            ax1.plot(data[0], data[1], linewidth=8, color=color)
            if data[2] not in [entry.get_label() for entry in legend_elements]:  # Prevent duplicate labels
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                  label=gb_latex_dict.get(data[2], data[2]),
                                                  markerfacecolor=color, markersize=15, linestyle='None'))

    text_str = f'{element_to_plot}\n{alloy_conc*100:.3f} at.%'
    ax1.text(0.05, 0.05, text_str, transform=ax1.transAxes, fontsize=40, verticalalignment='bottom', horizontalalignment='left')

    ax1.set_xlabel("Temperature (K)", fontsize=24)
    ax1.set_ylabel("Effective segregation energy  (eV)", fontsize=24)
    ax1.tick_params(labelsize=24)
    ax1.grid()

    kelvin_ticks = [73, 273, 473, 673, 873, 1073]
    celsius_ticks = [k - 273 for k in kelvin_ticks]

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(kelvin_ticks)
    ax2.set_xticklabels([f"{c}" for c in celsius_ticks])
    ax2.set_xlabel("Temperature (°C)", fontsize=24)
    ax2.tick_params(axis='x', labelsize=24)

    # Move the first entry to the end of the legend
    if legend_elements:
        first_entry = legend_elements.pop(0)
        legend_elements.append(first_entry)

    # Add the legend with customizable location and bbox_to_anchor
    ax1.legend(handles=legend_elements, fontsize=16, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    return fig, ax1

#%% Fig 9a
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
    
    areas = {}
    for GB, GB_df in df.groupby("GB"):
        structure = Structure.from_str(GB_df.iloc[0].structure, fmt="json")
        areas[GB] = structure.volume/structure.lattice.c/100 # nm^2 not Ang^2
    
    df[f"full_multiplicity_{x}"] = [[row[x]] * int(np.round(row.site_multiplicity / areas[row.GB])) for _, row in df.iterrows()]
    df[f"full_multiplicity_{y}"] = [[row[y]] * int(np.round(row.site_multiplicity / areas[row.GB])) for _, row in df.iterrows()]
    
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

#%% Fig 9b
def plot_single_stacked_histogram(df, 
                                  x="E_seg",
                                  xlabel=r"$\rm{R}_{\rm{W_{\rm{sep}}}}$", 
                                  xlabel_fontsize=24,
                                  ylabel="count",
                                  ylabel_fontsize=24,
                                  legend_fontsize=12,
                                  legend_loc=(0.0, 0.8),
                                  bin_width=0.02,
                                  figsize=(14, 8),
                                  hist_ticksize=20,
                                  custom_colors=None,
                                  gb_latex_dict=None, 
                                  title=None,
                                  title_fontsize=24):
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for plotting by expanding based on site_multiplicity
    df[f"full_multiplicity_{x}"] = [[row[x]] * int(np.round(row.site_multiplicity)) for _, row in df.iterrows()]

    # Calculate global bin edges based on all data
    all_x = np.concatenate(df[f"full_multiplicity_{x}"].explode().dropna().apply(np.ravel).tolist())
    min_x = np.floor(min(all_x) / bin_width) * bin_width
    max_x = np.ceil(max(all_x) / bin_width) * bin_width
    bins = np.arange(min_x, max_x + bin_width, bin_width)

    # Initialize bottom array for stacking
    bottom = np.zeros(len(bins) - 1)

    # Plot stacked histogram for each GB type
    for gb_type, color in custom_colors.items():
        gb_df = df[df['GB'] == gb_type]
        
        if gb_df.empty:
            continue  # Skip if there are no rows for this GB type

        plot_x = np.concatenate(gb_df[f"full_multiplicity_{x}"].explode().dropna().apply(np.ravel).tolist())
        
        if len(plot_x) == 0:
            continue  # Skip if plot_x is empty

        # Get histogram counts for the current GB group
        counts, _ = np.histogram(plot_x, bins=bins)

        # Plot the current GB group's data as a segment of the stacked histogram
        ax.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom, align='edge', 
               color=color, edgecolor='black', label=gb_latex_dict.get(gb_type, gb_type))
        bottom += counts  # Update the bottom array for stacking

    # Format the x-axis histogram
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=hist_ticksize)
    ax.grid(True, which="both", alpha=0.5)

    # Extract handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        # Reorder by popping the first entry and appending it to the end
        first_handle = handles.pop(0)
        first_label = labels.pop(0)
        handles.append(first_handle)
        labels.append(first_label)
        
        # Add reordered legend
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=legend_loc, 
                   bbox_transform=ax.transAxes, fontsize=legend_fontsize)

    # Add a title if specified
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    plt.tight_layout()

    return fig, ax

# Define the function to calculate c_GB
def calc_C_GB(Temperature, c_bulk, E_seg):
    """
    Calculate the grain boundary concentration.

    :param Temperature: Temperature in Kelvin
    :param c_bulk: Bulk concentration
    :param E_seg: Segregation energy array for a given element at different sites
    :return: Grain boundary concentration
    """
    c_GB = np.divide(c_bulk * np.exp(-E_seg/(8.6173303e-05 * Temperature)),
                     (1 - c_bulk + c_bulk * np.exp(-E_seg/(8.6173303e-05 * Temperature))))
    return c_GB

def plot_gb_histogram_with_cGB(df,
                               element,
                               c_bulk,
                               temperatures,
                               ylims_hist=(0, 25),
                               ylims_cGB=(0, 1.0),
                               custom_colors=None,
                               gb_latex_dict=None,
                               xlabel=r"E$_{\rm{seg}}$ (eV)",
                               xlabel_fontsize=20,
                               ylabel=r"Available GB interface sites (atoms/nm$^2$)",
                               ylabel_fontsize=20,
                               ylabel2="Langmuir-McLean isotherm predicted probability of GB site occupation",
                               ylabel2_fontsize=20,
                               legend_fontsize=20,
                               legend_loc=(0.0, 0.68),
                               bin_width=0.05,
                               figsize=(16, 12),
                               hist_ticksize=20,
                               title=None,
                               title_fontsize=24):
    """
    Plot a stacked histogram for GB data and overlay Langmuir-McLean isotherm predictions.
    
    :param df: DataFrame containing GB data.
    :param element: Element to filter the DataFrame.
    :param c_bulk: Bulk concentration as a decimal.
    :param temperatures: List of temperatures in Kelvin for c_GB calculations.
    :param ylabel_2: Label for the secondary y-axis.
    :param ylims_hist: Tuple for y-axis limits of the histogram.
    :param ylims_cGB: Tuple for y-axis limits of the c_GB curves.
    :param custom_colors: Dictionary of colors for different GB types.
    :param gb_latex_dict: Dictionary for LaTeX labels for different GB types.
    :param xlabel: Label for the x-axis.
    :param xlabel_fontsize: Font size for the x-axis label.
    :param ylabel: Label for the y-axis.
    :param ylabel_fontsize: Font size for the y-axis label.
    :param legend_fontsize: Font size for the legend.
    :param legend_loc: Location of the legend.
    :param bin_width: Width of the bins for the histogram.
    :param figsize: Size of the figure.
    :param hist_ticksize: Font size for the tick labels.
    :param title: Title of the plot.
    :param title_fontsize: Font size for the title.
    """
    # Filter data for the specified element
    filtered_df = df[df["E_seg"] < 0]
    filtered_df = filtered_df[filtered_df["element"] == element]
    filtered_df["structure"] = filtered_df.structure.apply(lambda x: Structure.from_str(x, fmt="json"))
    filtered_df["area"] = filtered_df.structure.apply(lambda x: x.volume / x.lattice.c)
    filtered_df["site_multiplicity"] = [row.site_multiplicity / (0.01 * row.area) for _, row in filtered_df.iterrows()]

    # Plot the stacked histogram
    fig, ax = plot_single_stacked_histogram(
        filtered_df,
        x="E_seg",
        xlabel=xlabel,
        xlabel_fontsize=xlabel_fontsize,
        ylabel=ylabel,
        ylabel_fontsize=ylabel_fontsize,
        legend_fontsize=legend_fontsize,
        legend_loc=legend_loc,
        bin_width=bin_width,
        figsize=figsize,
        hist_ticksize=hist_ticksize,
        custom_colors=custom_colors,
        gb_latex_dict=gb_latex_dict,
        title=title,
        title_fontsize=title_fontsize
    )
    ax.set_ylim(*ylims_hist)  # Set y-axis limits for histogram

    # Create a secondary y-axis for c_GB curves
    ax2 = ax.twinx()

    # Define different line styles for the c_GB lines
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # Prepare data for c_GB curves
    E_seg = np.linspace(filtered_df.E_seg.min(), 0, 500)

    # Plot c_GB curves for each temperature with different line styles
    for i, T in enumerate(temperatures):
        c_GB = calc_C_GB(T, c_bulk, E_seg)
        ax2.plot(E_seg, c_GB, label=r'c${_{\rm{GB}}}$ @' + f'T={T} K',
                 linewidth=10, linestyle=line_styles[i % len(line_styles)], color='black')

    # Label and format secondary axis
    ax2.set_ylabel(ylabel2, color='k', fontsize=ylabel2_fontsize)
    ax2.tick_params(axis='y', labelsize=20, labelcolor='black')  # Set tick font size and label color
    ax2.set_ylim(*ylims_cGB)  # Set y-axis limits for c_GB curves

    # Add a legend for the c_GB lines only
    #ax2.legend(loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=18, frameon=True)

    fig.tight_layout()
    plt.grid(True)
    return fig, ax, ax2


#%% Fig 9c
def create_prop_vs_temp_plot(plot_data, file_name, legend_elements, xlims, element_text, alloy_conc, ylabel_text, custom_colors, gb_latex_dict, legend=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    for data in plot_data:
        color = custom_colors.get(data[2], 'grey')
        ax.plot(data[0], data[1], label=gb_latex_dict.get(data[2], 'Unknown'), linewidth=8, color=color)

    ax.set_xlabel(r"Temperature (K)", fontsize=24)
    ax.set_ylabel(ylabel_text, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(True)
    
    if xlims:
        ax.set_xlim([max(0, xlims[0]), xlims[1]])
    if legend:
        ax.legend(handles=legend_elements, fontsize=22, frameon=True, handletextpad=0.01, borderpad=0.05, labelspacing=0.3)

    # Additional text
    text_str = f'{element_text}\n{alloy_conc*100:.2f} at.%'
    ax.text(0.02, 0.10, text_str, transform=ax.transAxes, fontsize=40, verticalalignment='bottom', horizontalalignment='left')
    
    # fig.savefig(f'{fig_dir}/TempEffectiveCohesion/{file_name}_{element_text}_{alloy_conc*100:.2f}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)
    return fig, ax

def plot_cohesion_vs_temp(df_spectra,
                          element_to_plot,
                          alloy_conc,
                          temp_range,
                          temp_step,
                          custom_colors,
                          gb_latex_dict,
                          xlims,
                          ylabel_text="",
                          cohesion_type="ANSBO",
                          ylabel_pad=20,
                          top_xlabel_pad=20):
    temperature_range = np.arange(temp_range[0], temp_range[1] + temp_step, temp_step)
    plot_data = []
    legend_elements = []

    for element, ele_df in df_spectra.groupby("element"):
        if element != element_to_plot:
            continue

        for _, row in ele_df.iterrows():
            temp_concs = []
            for temp in temperature_range:
                concentration = calc_C_GB(temp, alloy_conc, np.array(row.full_seg_spectra))
                eff_coh_eff = np.array(row[f'eta_coh_{cohesion_type}_spectra']) * concentration
                temp_concs.append(eff_coh_eff.sum())
            plot_data.append((temperature_range, temp_concs, row.GB))
            color = custom_colors.get(row.GB, 'grey')
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              label=gb_latex_dict.get(row.GB, 'Unknown'),
                                              markerfacecolor=color, markersize=15, linestyle='None'))
        legend_elements.append(legend_elements.pop(0))

    fig, ax = create_prop_vs_temp_plot(
        plot_data, f"EffectiveCohesion_vs_Temp_{cohesion_type}", legend_elements, 
        xlims, element_to_plot, alloy_conc, ylabel_text, custom_colors, gb_latex_dict, legend=False
    )

    # Set consistent tick label sizes on the main axes.
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    # Set the ylabel with the specified padding.
    ax.set_ylabel(ax.get_ylabel(), labelpad=ylabel_pad, fontsize=40)

    # Add a secondary x-axis on top showing temperature in Celsius instead of Kelvin.
    kelvin_ticks = [73, 273, 473, 673, 873, 1073]
    celsius_ticks = [k - 273 for k in kelvin_ticks]

    ax2 = ax.twiny()  # create secondary x-axis
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(kelvin_ticks)
    ax2.set_xticklabels([f"{c}" for c in celsius_ticks])
    ax2.set_xlabel("Temperature (°C)", fontsize=24, labelpad=top_xlabel_pad)
    # Set tick label sizes on the secondary x-axis.
    ax2.tick_params(axis='x', labelsize=24)

    return fig, ax

#%% Figure 10
def plot_prop_vs_prop(df,
                  x_prop,
                  y_prop,
                  figsize=(20,16),
                  x_label = r"E$_{\rm{seg}}$ (eV)",
                  x_label_fontsize = 30,
                  xtick_fontsize = 30,
                  y_label = r"W$_{\rm{sep}}^{\rm{RGS}}$ (J/m$^2$)",
                  y_label_fontsize = 30,
                  ytick_fontsize = 30,
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

    ax1.tick_params(axis='y', labelsize=xtick_fontsize, rotation=90)
    ax1.tick_params(axis='x', which='both', labelsize=ytick_fontsize)
    ax1.grid(which="both")

    plt.xlabel(x_label, fontsize=x_label_fontsize)
    plt.ylabel(y_label, fontsize=y_label_fontsize)
    
    # Setting x and y limits if provided
    if xlims is not None:
        ax1.set_xlim(xlims)
    if ylims is not None:
        ax1.set_ylim(ylims)
        
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0.1)
    return fig, ax1

def plot_prop_vs_prop_with_2d_histograms(x_values,
                                         y_values,
                                         figsize=(20, 16),
                                         x_label=r"E$_{\rm{seg}}$ (eV)",
                                         x_label_fontsize=30,
                                         xtick_fontsize=30,
                                         y_label=r"W$_{\rm{sep}}^{\rm{RGS}}$ (J/m$^2$)",
                                         y_label_fontsize=30,
                                         ytick_fontsize=30,
                                         hist_tick_fontsize=20,  # Added histograms' tick font size
                                         colorbar_tick_fontsize=20,  # Added colorbar tick font size
                                         colorbar_size=[0.92, 0.15, 0.02, 0.7],  # Colorbar size [left, bottom, width, height]
                                         legend_posn=(0.001, 0.681),
                                         xlims=None,
                                         ylims=None,
                                         x_bin_width=0.1,  # Bin width for the histograms
                                         y_bin_width=0.05,
                                         colormap='viridis',  # Colormap for the heatmap
                                         savefig_path=None,
                                         range_xy=None):  # Added range parameter

    # Create a figure with a custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4, wspace=0.3, hspace=0.3)
    
    # Main 2D histogram (heatmap)
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    
    # Histograms
    ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)

    # Drop NaN values in x and y data
    valid_mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_values = np.array(x_values)[valid_mask]
    y_values = np.array(y_values)[valid_mask]
    if range_xy is None:
        # Calculate the range for uniform binning and floor to align with zero
        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)

        # Floor the minimum values to the nearest lower integer and ceil the maximum values to ensure alignment
        x_min = np.floor(x_min)
        x_max = np.ceil(x_max)
        y_min = np.floor(y_min)
        y_max = np.ceil(y_max)

        # Ensure that zero is aligned as a bin edge
        x_min = min(x_min, 0)
        y_min = min(y_min, 0)
        range_xy = [(x_min, x_max), (y_min, y_max)]
        
    # Calculate aligned bins to have a bin edge at 0 based on the bin width
    x_min, x_max = range_xy[0]
    y_min, y_max = range_xy[1]
    
    # Calculate bin edges based on the specified bin width
    x_bins = np.arange(x_min, x_max + x_bin_width, x_bin_width)
    y_bins = np.arange(y_min, y_max + y_bin_width, y_bin_width)

    # 2D Histogram with log scale
    heatmap = ax_scatter.hist2d(x_values, y_values, bins=[x_bins, y_bins], cmap=colormap, 
                                norm=mcolors.LogNorm(), cmin=1, range=range_xy)

    # Add a colorbar in a new axis to avoid layout issues
    cbar_ax = fig.add_axes(colorbar_size)  # Positioning [left, bottom, width, height]
    cbar = plt.colorbar(heatmap[3], cax=cbar_ax)
    cbar.set_label('Count (log scale)', fontsize=20)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)  # Set colorbar tick font size

    # Set labels and ticks for scatter plot
    ax_scatter.set_xlabel(x_label, fontsize=x_label_fontsize)
    ax_scatter.set_ylabel(y_label, fontsize=y_label_fontsize)
    ax_scatter.tick_params(axis='y', labelsize=ytick_fontsize, rotation=90)
    ax_scatter.tick_params(axis='x', which='both', labelsize=xtick_fontsize)
    ax_scatter.grid(which="both", alpha=0.3)

    # Set limits if provided
    if xlims is not None:
        ax_scatter.set_xlim(xlims)
    if ylims is not None:
        ax_scatter.set_ylim(ylims)

    # Plot histograms
    ax_hist_x.hist(x_values, bins=x_bins, color="grey", alpha=0.7)
    ax_hist_y.hist(y_values, bins=y_bins, color="grey", alpha=0.7, orientation='horizontal')
    ax_hist_x.grid()
    ax_hist_y.grid()
    # Set tick params for the histograms with custom font size
    ax_hist_x.tick_params(axis='y', labelsize=hist_tick_fontsize)
    ax_hist_y.tick_params(axis='x', labelsize=hist_tick_fontsize)

    # Hide x tick labels for top histogram and y tick labels for right histogram
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)
    plt.setp(ax_hist_y.get_yticklabels(), visible=False)

    # Save figure if path is provided
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0.1)

    return fig, ax_scatter


#%% Fig 11

def plot_prop_vs_prop_GB(df, gb_to_plot='S5_RA001_S210', y_prop="R_DDEC6_ANSBO", x_prop="E_seg", 
                         custom_colors=None, gb_latex_dict=None, fig_dir='.', 
                         ylabel=r"R$_{\rm{ANSBO}}$= ANSBO$_{\rm{seg}}$/ANSBO$_{\rm{pure}}$", 
                         xlabel=r"E$_{\rm{seg}}$ (eV)", figsize=(12, 12), 
                         xlabel_fontsize=36, ylabel_fontsize=36, xtick_fontsize=20, 
                         ytick_fontsize=24, point_label_fontsize=24, padding_fraction=0.001,
                         label_left=None, label_right=None, label_top=None, label_bottom=None,
                         horizontal_padding=0.001, vertical_padding=0.001):
    
    # Initialize label position lists if None
    label_left = label_left or []
    label_right = label_right or []
    label_top = label_top or []
    label_bottom = label_bottom or []

    # Filter and prepare data
    filtered_df = df.copy()

    # Get the range of x-axis and y-axis
    x_range = filtered_df[x_prop].max() - filtered_df[x_prop].min()
    y_range = filtered_df[y_prop].max() - filtered_df[y_prop].min()

    # Calculate the padding in axes coordinates
    padding_axes_x = horizontal_padding * x_range
    padding_axes_y = vertical_padding * y_range

    # Filter the DataFrame for the specified GB
    filtered_df = filtered_df[filtered_df["GB"] == gb_to_plot]

    if filtered_df.empty:
        print(f"No data available for GB: {gb_to_plot}")
        return None, None

    fig, ax1 = plt.subplots(figsize=figsize, dpi=80)
    color = custom_colors.get(gb_to_plot, 'grey')  # Use custom colors from dictionary

    # Minimum E_seg calculation per element
    min_eseg_per_element = filtered_df.groupby("element").apply(lambda x: x.nsmallest(1, x_prop).iloc[0])
    min_eseg_per_element = min_eseg_per_element[min_eseg_per_element[x_prop] <= 0]
    min_eseg_per_element = min_eseg_per_element.sort_values(by='Z')

    # Scatter plot with annotations
    for _, row in min_eseg_per_element.iterrows():
        x_value = row[x_prop]
        y_value = row[y_prop]
        element = row["element"]

        # Plot the scatter point
        ax1.scatter(x_value, y_value, color="r", marker="o", s=40)

        # Custom label positioning based on user-defined lists
        if element in label_left:
            ax1.text(x_value - padding_axes_x, y_value, element, color="k", 
                     verticalalignment='center', horizontalalignment='right', 
                     fontsize=point_label_fontsize)
        elif element in label_right:
            ax1.text(x_value + padding_axes_x, y_value, element, color="k", 
                     verticalalignment='center', horizontalalignment='left', 
                     fontsize=point_label_fontsize)
        elif element in label_top:
            ax1.text(x_value, y_value + padding_axes_y, element, color="k", 
                     verticalalignment='bottom', horizontalalignment='center', 
                     fontsize=point_label_fontsize)
        elif element in label_bottom:
            ax1.text(x_value, y_value - padding_axes_y, element, color="k", 
                     verticalalignment='top', horizontalalignment='center', 
                     fontsize=point_label_fontsize)
        else:
            # Default positioning if not specified
            ax1.text(x_value-padding_axes_x, y_value + padding_axes_y, element, color="k", 
                     verticalalignment='center', horizontalalignment='right', 
                     fontsize=point_label_fontsize)

    # Text in the top left corner with large font size
    ax1.text(0.02, 0.98, gb_latex_dict.get(gb_to_plot, gb_to_plot), color="k", fontsize=40, 
             verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)

    # Axes customization
    ax1.axhline(1.00, color='r', linewidth=2, linestyle="--")
    ax1.axvline(0, color='r', linewidth=2, linestyle="--")
    
    ax1.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax1.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    
    ax1.tick_params(axis='x', labelsize=xtick_fontsize)
    ax1.tick_params(axis='y', labelsize=ytick_fontsize)

    ax1.grid(True)
    # fig.savefig(f'{fig_dir}/SegregationEngineering_min_{x_prop}_vs_{y_prop}_{gb_to_plot}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)

    return fig, ax1

#%% Fig 12

#%%
def periodic_table_dual_plot(plot_df, 
                        property1="Eseg_min1",
                        property2="Eseg_min2",  # New property
                        count_min1=None,
                        count_max1=None,
                        count_min2=None,
                        count_max2=None,
                        center_cm_zero1=False,
                        center_cm_zero2=False,
                        center_point1=None,  # New parameter for arbitrary centering
                        center_point2=None,
                        property_name1=None,
                        property_name2=None,
                        cmap1=plt.cm.Blues,  # Colormap for the first property
                        cmap2=plt.cm.Reds,  # Colormap for the second property
                        element_font_color="darkgoldenrod"):
    module_path = os.path.dirname(os.path.abspath(__file__))
    ptable = pd.read_csv(os.path.join(module_path, 'periodic_table.csv'))
    ptable.index = ptable['symbol'].values
    elem_tracker = ptable['count']
    ptable = ptable[ptable['Z'] <= 92]  # Cap at element 92

    n_row = ptable['row'].max()
    n_column = ptable['column'].max()

    fig, ax = plt.subplots(figsize=(n_column, n_row))
    rows = ptable['row']
    columns = ptable['column']
    symbols = ptable['symbol']
    rw = 0.9  # rectangle width
    rh = rw    # rectangle height

    if count_min1 is None or count_min2 is None or count_max1 is None or count_max2 is None:
        show_symbols = False
    else:
        show_symbols = True
    
    if count_min1 is None:
        count_min1 = plot_df[property1].min()
    if count_max1 is None:
        count_max1 = plot_df[property1].max()

    # Adjust normalization based on centering preference
    if center_cm_zero1:
        cm_threshold1 = max(abs(count_min1), abs(count_max1))
        norm1 = Normalize(-cm_threshold1, cm_threshold1)
    elif center_point1 is not None:
        # Adjust normalization to center around the arbitrary point
        max_diff = max(center_point1 - count_min1, count_max1 - center_point1)
        norm1 = Normalize(center_point1 - max_diff, center_point1 + max_diff)
    else:
        norm1 = Normalize(vmin=count_min1, vmax=count_max1)

    if count_min2 is None:
        count_min2 = plot_df[property2].min()
    if count_max2 is None:
        count_max2 = plot_df[property2].max()

    # Adjust normalization based on centering preference for the second property
    if center_cm_zero2:
        cm_threshold2 = max(abs(count_min2), abs(count_max2))
        norm2 = Normalize(-cm_threshold2, cm_threshold2)
    elif center_point2 is not None:
        # Adjust normalization to center around the arbitrary point for the second property
        max_diff2 = max(center_point2 - count_min2, count_max2 - center_point2)
        norm2 = Normalize(center_point2 - max_diff2, center_point2 + max_diff2)
    else:
        norm2 = Normalize(vmin=count_min2, vmax=count_max2)

    for row, column, symbol in zip(rows, columns, symbols):
        row = ptable['row'].max() - row
        # Initial color set to 'none' for both properties
        color1, color2 = 'none', 'none'

        if symbol in plot_df.element.unique():
            element_data = plot_df[plot_df["element"] == symbol]
            if property1 in element_data and not element_data[property1].isna().all():
                value1 = element_data[property1].values[0]
                color1 = cmap1(norm1(value1))
            if property2 in element_data and not element_data[property2].isna().all():
                value2 = element_data[property2].values[0]
                color2 = cmap2(norm2(value2))

        # Draw upper right triangle for property1
        triangle1 = patches.Polygon([(column, row), (column + rw, row), (column + rw, row + rh)], 
                                    closed=True, color=color1)
        ax.add_patch(triangle1)
        
        # Draw lower left triangle for property2
        triangle2 = patches.Polygon([(column, row), (column, row + rh), (column + rw, row + rh)], 
                                    closed=True, color=color2)
        ax.add_patch(triangle2)

        # Element symbol
        plt.text(column + rw / 2, row + rh / 2, symbol,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=22,  # Adjusted for visibility
                 fontweight='semibold',
                 color=element_font_color)
    position1 = 3.5, 7.8
    position2 = 3.5, 9.4
    # draw_color_bar(fig, ax, norm1, cmap1, property_name1, position1, granularity=20)
    # draw_color_bar(fig, ax, norm2, cmap2, property_name2, position2, granularity=20)
    draw_color_bar(fig, ax, norm1, cmap1, property_name1, position1, show_symbols, granularity=20)
    draw_color_bar(fig, ax, norm2, cmap2, property_name2, position2, show_symbols, granularity=20)

    ax.set_ylim(-0.15, n_row + .1)
    ax.set_xlim(0.85, n_column + 1.1)
    ax.axis('off')
    
    plt.draw()
    plt.pause(0.001)
    plt.close()
    return fig, ax

def draw_color_bar(fig, ax, norm, cmap, property_name, position, show_symbols=True, granularity=20):
    colormap_array = np.linspace(norm.vmin, norm.vmax, granularity)
    
    length = 9
    width = length / granularity
    height = 0.35
    x_offset, y_offset = position

    for i, value in enumerate(colormap_array):
        color = cmap(norm(value))
        color = 'silver' if value == 0 and not norm.vmin <= 0 <= norm.vmax else color
        x_loc = i / granularity * length + x_offset
        
        rect = patches.Rectangle((x_loc, y_offset), width, height,
                                 linewidth=1.5,
                                 edgecolor='gray',
                                 facecolor=color,
                                 alpha=1)
        ax.add_patch(rect)

        if i in [0, granularity//4, granularity//2, 3*granularity//4, granularity-1]:
            label = f'{value:.1f}'
            if show_symbols:
                if i == 0:
                    label = "<" + label
                elif i == granularity - 1:
                    label = ">" + label
            
            plt.text(x_loc + width / 2, y_offset - 0.4, label,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontweight='semibold',
                     fontsize=20, color='k')

    plt.text(x_offset + length / 2, y_offset + 0.75,
             property_name,
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='semibold',
             fontsize=24, color='k')
