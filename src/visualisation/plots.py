"""
visualisation/plots.py
----------------------
Statistical plots: bar charts, violin plots, heatmaps, scatter plots,
and the SRG-pair heatmap.  All functions accept DataFrames produced by
the analysis module and save publication-quality PDFs and SVGs.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform

from src.constants import (
    CITY_MAP,
    ISP_COLOURS, ISP_LABELS, ISP_LIST,
    SRG_COLOURS,
    STATUS_COLOURS,
)


# ---------------------------------------------------------------------------
# rcParams helper
# ---------------------------------------------------------------------------

RCPARAMS = {
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}

RCPARAMS_LARGE = {**RCPARAMS, "font.size": 20, "legend.fontsize": 20}

RCPARAMS_SMALL = {
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize" :14,
    "ytick.labelsize" :14,
    "legend.fontsize" :14,
}


def _apply_rc(size: str = 'small'):
    if size == 'large':
        plt.rcParams.update(RCPARAMS_LARGE)
    elif size == 'regular': 
        plt.rcParams.update(RCPARAMS)
    else:
        plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# SRLG vs ISP bar chart (counts)
# ---------------------------------------------------------------------------

def srg_isp_bar(srg_isp_df: pd.DataFrame, filename: str, save_dir: str = "images") -> None:
    """Bar chart: number of ISPs affected per SRG. Same as inset bar graph from SRLG map."""
    _apply_rc()

    # copy and get counts
    df = srg_isp_df.copy().set_index('srg_id')
    df['total'] = df.sum(axis=1, numeric_only=True)
    unique_vals, counts = np.unique(df['total'], return_counts=True)

    # make plot
    plt.bar(unique_vals, counts,
            color=SRG_COLOURS['conservative'], edgecolor='black',
            tick_label=list(range(1, len(ISP_LIST) + 1)))
    plt.xlabel('Number of ISPs affected')
    plt.ylabel('Count of SRGs')
    plt.grid(axis='y')

    # save and show
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# ISP sharing violin plot
# ---------------------------------------------------------------------------

def srg_isp_violin(srg_isp_df: pd.DataFrame, filename: str, save_dir: str = "images") -> None:
    """
    Violin plot showing how many other ISPs share each SRLG with a given ISP.
    Overlaid with individual scatter points. That is, for each ISP, look at each of their SRLGs, 
    and count how many ISPs also use that SRLG. Then create violin plot based on how many ISPs 
    they share their SRLGs with. 
    """
    _apply_rc()

    # copy df
    df = srg_isp_df.copy().set_index('srg_id')

    # count ISPs per SRG using the indentifier matrix
    isp_per_srg = df.sum(axis=1)

    # for each ISP column, find amount shared for each of their SRLGs.
    results = []
    for isp in ISP_LIST:
        mask   = df[isp] == 1
        vals   = (isp_per_srg[mask] - 1).dropna().values
        results.append(vals)

    # create plot
    positions = np.arange(1, len(ISP_LIST) + 1) #put in set order
    fig, ax   = plt.subplots(figsize=(7, 5))

    vplot = ax.violinplot(results, positions=positions,
                          showmeans=False, showmedians=True, showextrema=False)

    # specify violin shading
    for body, isp in zip(vplot['bodies'], ISP_LIST):
        body.set_facecolor(ISP_COLOURS[isp])
        body.set_edgecolor('black')
        body.set_alpha(0.65)
        body.set_linewidth(0.8)
    # set median line shading
    vplot['cmedians'].set_color('black')
    vplot['cmedians'].set_linewidth(1.2)

    # overlay plots with scatter points
    for i, vals in enumerate(results, start=1):
        jitter = np.random.default_rng(42).normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=14, color='black', alpha=0.45, linewidths=0, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(ISP_LABELS.values()))
    ax.set_ylabel("Count of other ISPs for each SRLG")
    ax.tick_params(axis='y', which='both', right=True, labelright=False, direction='in')
    plt.tight_layout()

    # save and show
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# ISP risk heatmap
# ---------------------------------------------------------------------------

def isp_risk_heatmap(srg_isp_df: pd.DataFrame, filename: str, save_dir: str = "images") -> None:
    """
    Hamming-distance heatmap comparing pairwise ISP SRLG risk overlap (how many similar 
    SRLGs they have). Higher value indiates greater disimilarity (greater distance). 
    Hamming distance computes pairwise proportion of differences.
    """
    _apply_rc()

    # find hamming distance from matrix
    df = srg_isp_df.copy().set_index('srg_id')
    dist_array = pdist(df.T, metric='hamming') # transpose so that pdist calculates distance between rows
    
    # convert to readable square matrix (DataFrame)
    dist_matrix = squareform(dist_array)
    hamming_df  = pd.DataFrame(dist_matrix, index=df.columns, columns=df.columns)

    # CREATE PLOT
    plt.figure(figsize=(5, 4))
    sns.heatmap(hamming_df, square=True, cmap='viridis',
                xticklabels=list(ISP_LABELS.values()),
                yticklabels=list(ISP_LABELS.values()))
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # show and save
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# Disconnect count bar charts
# ---------------------------------------------------------------------------

def _bars(counts, x_label, y_label, filename, save_dir):
    """Shared logic for both bar charts."""
    vc = counts.value_counts().sort_index()
    all_idx = sorted(set(vc.index))
    vc = vc.reindex(all_idx, fill_value=0)
    x = np.arange(len(all_idx))

    # make plot
    fig, ax = plt.subplots()
    ax.bar(x, vc)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x); ax.set_xticklabels(all_idx)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    
    # save and show
    plt.savefig(f"{save_dir}/{filename}", transparent=True)
    plt.show()


def city_failures(
    city_isp_fail_counts: pd.DataFrame,
    filename: str = 'city_fail_bar.pdf',
    save_dir: str = "images",
) -> None:
    """Bar chart: number of cities affected per failure event."""
    _apply_rc()
    _bars(city_isp_fail_counts['n_cities'], 'Number of cities affected', 'Disconnect count',
            filename, save_dir)


def isp_failures(
    city_isp_fail_counts: pd.DataFrame,
    filename: str = 'isp_fail_bar.pdf',
    save_dir: str = "images",
) -> None:
    """Bar chart: number of ISPs affected per failure event."""
    _apply_rc()
    _bars(city_isp_fail_counts['n_isps'], 'Number of ISPs affected', 'Disconnect Count',
            filename, save_dir)


# ---------------------------------------------------------------------------
# SRLG local vs global plots
# ---------------------------------------------------------------------------

def srg_local_hist(
    srg_sum_df: pd.DataFrame,
    filename: str ='local_srg.pdf',
    save_dir: str = "images",
) -> None:
    """
    Histogram of local failures across the SRLGs (how many local failures caused by each SRLG).
    """
    _apply_rc()
    plt.hist(srg_sum_df['local_only_outages'], bins=50, alpha=0.5)
    plt.xlabel('Number of Local Failures')
    plt.ylabel('SRG Count')
    plt.grid(axis='y')
    plt.savefig(f'{save_dir}/{filename}', transparent=True)
    plt.show()

def srg_global_bar(
    srg_sum_df: pd.DataFrame,
    filename: str ='global_srg.pdf',
    save_dir: str = "images",
) -> None:
    """
    Bar chart of global failures across the SRLGs (how many global failures caused by each SRLG).
    """
    _apply_rc()
    _bars(srg_sum_df['global_outages'], 'Number of Global Failures', 'SRLG Count',
            filename, save_dir)

def srg_local_global_scatter(
    srg_sum_df: pd.DataFrame,
    filename: str ='srg_local_vs_global.pdf',
    save_dir: str = "images",
) -> None:
    # Scatter: local vs global
    plt.scatter(srg_sum_df['local_only_outages'], srg_sum_df['global_outages'])
    plt.xlabel('Local Failures')
    plt.ylabel('Global Failures')
    plt.yticks([0, 1, 2, 3])
    plt.grid()
    plt.savefig(f'{save_dir}/{filename}', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# SRG pair heatmap
# ---------------------------------------------------------------------------

def plot_srg_pair_heatmap(
    df_pairs: pd.DataFrame,
    srg_sum_df: pd.DataFrame,
    srg1_col: str = "srg_id_1",
    srg2_col: str = "srg_id_2",
    figsize_scale: float = 0.5,
    diagonal_colour: str = "white",
    filename: str = "fail_heatmap.pdf",
    save_dir: str = "images",
) -> None:
    """
    Upper-triangle heatmap showing the outcome class (none / local /
    global disconnect) for every SRG pair in *df_pairs*.

    Pairs with only 2 local outages are filtered.

    Parameters
    ----------
    df_pairs : DataFrame
        Full output of ``remove_double_srgs``.
    srg_sum_df : DataFrame
        Output of ``record_srg_failures``.
    """
    _apply_rc(size='small')

    # first get correct order of each status
    status_order  = list(STATUS_COLOURS.keys())

    # get all SRLG pair combinations
    df = df_pairs.copy().dropna(subset=[srg1_col, srg2_col])
    df[srg1_col] = df[srg1_col].astype(str)
    df[srg2_col] = df[srg2_col].astype(str)

    # observe if a local failure or global failure occurred.
    isp_comp_cols = [f"{isp}_n_components" for isp in ISP_LIST]
    df['local_fail'] = (df[isp_comp_cols] > 1).any(axis=1).values
    df['global_fail'] = (df[['n_components']] > 1).any(axis=1).values

    # create class col
    df['class'] = 0

    # drop unecessary cols
    df = df[[srg1_col, srg2_col, 'local_fail', 'global_fail', 'class']].copy()

    # check through each SRLG col. If only has two local failures and no global, filter out.
    for idx, row in df.iterrows():
        s1, s2 = row[srg1_col], row[srg2_col]
        lc1 = srg_sum_df.loc[srg_sum_df['srg_id'] == s1, 'local_only_outages']
        lc2 = srg_sum_df.loc[srg_sum_df['srg_id'] == s2, 'local_only_outages']
        gc1 = srg_sum_df.loc[srg_sum_df['srg_id'] == s1, 'global_outages']
        gc2 = srg_sum_df.loc[srg_sum_df['srg_id'] == s2, 'global_outages']
        lc1 = lc1.iloc[0] if len(lc1) else 0
        lc2 = lc2.iloc[0] if len(lc2) else 0
        gc1 = gc1.iloc[0] if len(gc1) else 0
        gc2 = gc2.iloc[0] if len(gc2) else 0
        # filter out low-risk pairs
        if (lc1 <= 2 and gc1 == 0) or (lc2 <= 2 and gc2 == 0):
            df.at[idx, 'class'] = np.nan
            continue
        if row['global_fail']:
            df.at[idx, 'class'] = 2
        elif row['local_fail']:
            df.at[idx, 'class'] = 1

    # get nice pair ordering so that are symmetric
    df = df.dropna(subset=['class'])
    df["srgA"] = np.minimum(df[srg1_col].values, df[srg2_col].values)
    df["srgB"] = np.maximum(df[srg1_col].values, df[srg2_col].values)

    # list of SRLGs involved in any pair
    srgs = sorted(set(df["srgA"]) | set(df["srgB"]))
    # create more readable labels for the SRLGs
    srg_labels = [f'({s[0:3]}, {s[4:7]}) {s[-5:-2]} {s[-1:]}' for s in srgs]

    # set up matrix/diagonals
    d_rev = df.rename(columns={srg1_col: srg2_col, srg2_col: srg1_col})
    d_sym = pd.concat([df, d_rev], ignore_index=True)
    mat = d_sym.pivot(index=srg1_col, columns=srg2_col, values="class")
    mat = mat.reindex(index=srgs, columns=srgs)
    np.fill_diagonal(mat.values, np.nan)

    # define colour map
    cmap = ListedColormap([STATUS_COLOURS[c] for c in status_order])
    cmap.set_bad(diagonal_colour)
    bounds = np.arange(len(status_order) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    # create figure
    n = len(srgs)
    fig_w = max(8, n * figsize_scale)
    fig_h = max(6, n * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="equal", interpolation="none")

    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(srg_labels, rotation=90)
    ax.set_yticklabels(srg_labels)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    handles = [Patch(facecolor=STATUS_COLOURS[c], edgecolor="none", label=c)
               for c in status_order]
    ax.legend(handles=handles[:3], title="Disconnect Type",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    # save and show
    plt.savefig(f'{save_dir}/{filename}', transparent=True)
    plt.show()