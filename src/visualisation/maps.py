"""
visualisation/maps.py
---------------------
Map plotting functions: SRG geographic maps, multilayer panel layout,
and the full multilayer supra-graph visualisation.
"""

import os
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import networkx as nx
import numpy as np

from src.constants import (
    EDGE_COLOURS, EDGE_STYLES, EDGE_LABELS,
    ISP_COLOURS, ISP_LIST, ISP_LABELS, ISP_STYLES,
    POS_ADJUST,
    TER_EDGE_LABELS, TER_EDGE_STYLES,
    PANEL_LABELS, PANEL_LABEL_ADJUSTMENTS_X, PANEL_LABEL_ADJUSTMENTS_Y
)
from src.network.build import is_ixp_node, node_isp, node_layer
from src.visualisation.geo import (
    load_au_state_polygons,
    safe_pos,
    geo_to_panel_xy,
    transform_au_polygons_to_panel,
    get_lon_lat,
    panel_map_dimensions,
)


# ---------------------------------------------------------------------------
# Default matplotlib settings
# ---------------------------------------------------------------------------

RCPARAMS_BASE = {
    "font.size": 16,
    "legend.fontsize": 20,
}


def apply_rc(params = RCPARAMS_BASE):
    plt.rcParams.update(params)


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def edge_iter(G: nx.Graph):
    """Grab edges to iterate through"""
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, d in G.edges(data=True):
            yield u, v, None, d

def curvature(seen: dict, pair: tuple, i: int, step: float = 0.15) -> float:
    """Add curvature to multi-edges so that do not overlap"""
    return 0.0 if i == 0 else step * ((i + 1) // 2) * (1 if i % 2 == 1 else -1)


# ---------------------------------------------------------------------------
# SRLG map: Coloured by type
# ---------------------------------------------------------------------------

def plot_srg_aus_type(G: nx.Graph, filename: str, save_dir: str = "images") -> None:
    """
    Plot SRLG map on Australia with SRLG edges coloured by
    route type (submarine, road, rail and road, etc.).

    Parameters
    ----------
    G : nx.Graph
        SRLG graph (output of ``build_srg_graph``).
    filename : str
        Output filename.
    save_dir : str
        Directory for output files.
    """
    apply_rc({"legend.fontsize": 22})

    # load map
    au_states = load_au_state_polygons()
    # get positions and edges
    pos = safe_pos(G)
    edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if u in pos and v in pos]
    # grab labels and remove "_1"
    labels = nx.get_node_attributes(G, 'city')
    abr_labels = {k: k[:-2] for k, _ in labels.items()}

    # begin creating plot
    fig, ax = plt.subplots(figsize=(12, 12))
    au_states.plot(ax=ax, color="whitesmoke", edgecolor="dimgray",
                   linewidth=0.9, zorder=0)

    # determine which type of node (MPoP or regional)
    node_types = nx.get_node_attributes(G, 'kind')
    ixp_nodes = [k for k, v in node_types.items() if str(v) == 'MPoP']
    regional_nodes = [k for k, v in node_types.items() if str(v) == 'regional']

    # draw the nodes separately (MPoPs have IXPs, and are presented as black diamonds)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ixp_nodes,
                           node_size=90, node_color="black",
                           node_shape="D", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=regional_nodes,
                           node_size=70, node_color="black", alpha=0.6)

    # change label positions based on custom values in POS_ADJUST
    label_pos = {
        n: (pos[n][0] + POS_ADJUST[n][0], pos[n][1] + POS_ADJUST[n][1])
        for n in G.nodes if n in POS_ADJUST and n in pos
    }
    nx.draw_networkx_labels(G, label_pos, ax=ax, labels=abr_labels,
                            font_size=18,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor="black",
                                      alpha=0.4))

    # draw edges and adjust so don't overlap. draw with specific type style
    seen = defaultdict(int)
    for u, v, k, d in edges:
        pair = (u, v) if str(u) <= str(v) else (v, u)
        i    = seen[pair]; seen[pair] += 1
        rad  = curvature(seen, pair, i)
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], arrows=True,
            connectionstyle=f"arc3,rad={rad}", width=2.2, alpha=0.8,
            edge_color=EDGE_COLOURS[d.get("srg_type", "road")],
            style=EDGE_STYLES[d.get("srg_type", "road")],
        )

    # add legend
    legend_lines = [
        Line2D([0], [0], color=EDGE_COLOURS[st], lw=2,
               linestyle=EDGE_STYLES[st], label=EDGE_LABELS[st])
        for st in sorted(EDGE_COLOURS) if st not in ('rail', 'mix')
    ]
    ax.legend(handles=legend_lines, loc="upper left")
    x_min, x_max = ax.get_xlim(); ax.set_xlim(x_min, x_max + 1.5)
    ax.axis("off")

    # save
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# SRLG map (coloured by ISP degree of edges)
# ---------------------------------------------------------------------------

def plot_srg_aus_deg(
    G: nx.Graph,
    srg_vs_isp_df,
    filename: str,
    save_dir: str = "images"
) -> None:
    """
    Plot SRLG map on Australia with SRLG edges coloured by
    the number of ISPs that use each SRLG.
    """
    apply_rc({"legend.fontsize": 18})

    # first determine the degrees of each SRLG (how many ISPs are a part of each SRLG)
    srg_degree_df = srg_vs_isp_df.copy()
    srg_degree_df['degree'] = srg_degree_df.sum(axis=1, numeric_only=True)
    # create dictionary with SRG ID then the corresponding degree
    srg_deg_dict = dict(zip(srg_degree_df['srg_id'], srg_degree_df['degree']))

    # load map
    au_states = load_au_state_polygons()
    # get positions and edges
    pos = safe_pos(G)
    edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if u in pos and v in pos]
    # grab labels and remove "_1"
    labels = nx.get_node_attributes(G, 'city')
    abr_labels = {k: k[:-2] for k, _ in labels.items()}

    # grab new colour map (requires matplotlib 3.10+)
    cmap = plt.get_cmap('managua')
    colours = cmap(np.linspace(0.05, 0.98, 6))

    # create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    au_states.plot(ax=ax, color="whitesmoke", edgecolor="dimgray",
                   linewidth=0.9, zorder=0)

    # determine which type of node (MPoP or regional)
    node_types = nx.get_node_attributes(G, 'kind')
    ixp_nodes = [k for k, v in node_types.items() if str(v) == 'MPoP']
    regional_nodes = [k for k, v in node_types.items() if str(v) == 'regional']

    # draw the nodes separately (MPoPs have IXPs, and are presented as black diamonds)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ixp_nodes,
                           node_size=90, node_color="black",
                           node_shape="D", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=regional_nodes,
                           node_size=70, node_color="black", alpha=0.6)

    # draw edges and adjust so don't overlap. draw with specific type style
    # colour based on degree
    seen = defaultdict(int)
    for u, v, k, d in edges:
        pair = (u, v) if str(u) <= str(v) else (v, u)
        i = seen[pair]; seen[pair] += 1
        rad = curvature(seen, pair, i)
        deg = round(srg_deg_dict.get(k, 1)) # determine isp deg
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], arrows=True,
            connectionstyle=f"arc3,rad={rad}", width=2.2, alpha=0.8,
            edge_color=colours[max(0, deg - 1)],
            style=TER_EDGE_STYLES.get(d.get("srg_type", "road"), '-'),
        )

    # change label positions based on custom values in POS_ADJUST
    label_pos = {
        n: (pos[n][0] + POS_ADJUST[n][0], pos[n][1] + POS_ADJUST[n][1])
        for n in G.nodes if n in POS_ADJUST and n in pos
    }
    nx.draw_networkx_labels(G, label_pos, ax=ax, labels=abr_labels,
                            font_size=18,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor="black",
                                      alpha=0.4))

    # add legend (edge type only)
    legend_lines = [
        Line2D([0], [0], lw=2,
               linestyle=TER_EDGE_STYLES.get(st, '-'),
               label=TER_EDGE_LABELS.get(st, st), color='black')
        for st in sorted(EDGE_COLOURS) if st not in ('rail', 'mix', 'road')
    ]
    first_legend = ax.legend(handles=legend_lines, loc="upper left")
    ax.add_artist(first_legend)

    # Inset bar chart: SRLG count by ISP degree
    axins = inset_axes(ax, width="25%", height="25%",
                        bbox_to_anchor=(0.01, 0, 1, 1),
                        bbox_transform=ax.transAxes,
                        loc="lower left", borderpad=1)
    unique_vals, counts = np.unique(srg_degree_df['degree'], return_counts=True)
    axins.bar(unique_vals, counts, color=colours[:len(unique_vals)],
              edgecolor='black', tick_label=list(range(1, len(ISP_LIST) + 1)))
    axins.tick_params(axis='both', labelsize=18)
    axins.set_facecolor("none"); axins.patch.set_alpha(0)
    axins.set_ylabel('Count of SRLGs', fontsize=18)
    axins.set_xlabel('Number of ISPs affected', fontsize=18)
    for tick in axins.yaxis.get_major_ticks():
        tick.label1.set_fontfamily('sans-serif')
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)

    x_min, x_max = ax.get_xlim(); ax.set_xlim(x_min, x_max + 1.5)
    ax.axis("off")

    # save
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# SRLG map: Failure types
# ---------------------------------------------------------------------------

def plot_srg_fail_map(
    G: nx.Graph,
    srg_sum_df,
    filename: str,
    save_dir: str = "images",
) -> None:
    """
    Plot SRLG map on Australia, with colour indicating severity of failure:
      - Red: involved in a global disconnect
      - Orange: involved in a local-only disconnect (> 2 occurrences)
      - Grey: not involved in any significant disconnect (<2 local)
    """
    from matplotlib.patches import Patch as MPatch
    apply_rc({"legend.fontsize": 20, "legend.title_fontsize": 18})
    os.makedirs(save_dir, exist_ok=True)

    # load map
    au_states = load_au_state_polygons()
    # get positions and edges
    pos = safe_pos(G)
    edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if u in pos and v in pos]
    # grab labels and remove "_1"
    labels = nx.get_node_attributes(G, 'city')
    abr_labels = {k: k[:-2] for k, _ in labels.items()}

    # begin creating plot
    fig, ax = plt.subplots(figsize=(12, 12))
    au_states.plot(ax=ax, color="whitesmoke", edgecolor="dimgray",
                   linewidth=0.9, zorder=0)

    # determine which type of node (MPoP or regional)
    node_types = nx.get_node_attributes(G, 'kind')
    ixp_nodes = [k for k, v in node_types.items() if str(v) == 'MPoP']
    regional_nodes = [k for k, v in node_types.items() if str(v) == 'regional']

    # draw the nodes separately (MPoPs have IXPs, and are presented as black diamonds)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ixp_nodes,
                           node_size=90, node_color="black",
                           node_shape="D", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=regional_nodes,
                           node_size=70, node_color="black", alpha=0.6)

    # draw edges and adjust so don't overlap. draw with specific type style
    # colour based on outages
    seen = defaultdict(int)
    for u, v, k, d in edges:
        pair = (u, v) if str(u) <= str(v) else (v, u)
        i    = seen[pair]; seen[pair] += 1
        rad  = curvature(seen, pair, i)

        # count outages for this SRLG
        local_rows  = srg_sum_df.loc[srg_sum_df['srg_id'] == k, 'local_only_outages']
        global_rows = srg_sum_df.loc[srg_sum_df['srg_id'] == k, 'global_outages']
        local_count  = local_rows.iloc[0]  if len(local_rows)  else 0
        global_count = global_rows.iloc[0] if len(global_rows) else 0

        # set colour based on counts
        if global_count > 0:
            edge_colour, alpha = 'tab:red', 0.8
        elif local_count > 2:
            edge_colour, alpha = 'tab:orange', 0.8
        else:
            edge_colour, alpha = 'black', 0.2

        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], arrows=True,
            connectionstyle=f"arc3,rad={rad}", width=2.2, alpha=alpha,
            edge_color=edge_colour,
            style=TER_EDGE_STYLES.get(d.get("srg_type", "road"), '-'),
        )

    # change label positions based on custom values in POS_ADJUST
    label_pos = {
        n: (pos[n][0] + POS_ADJUST[n][0], pos[n][1] + POS_ADJUST[n][1])
        for n in G.nodes if n in POS_ADJUST and n in pos
    }
    nx.draw_networkx_labels(G, label_pos, ax=ax, labels=abr_labels,
                            font_size=18,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor="black",
                                      alpha=0.4))

    # add legend
    status_colours = {"None": "lightgrey", "Local Only": "orange", "Global": "red"}
    handles = [MPatch(facecolor=c, edgecolor="none", label=lbl)
               for lbl, c in status_colours.items()]
    ax.legend(handles=handles, title="Disconnect Type",
              bbox_to_anchor=(1.02, 1), loc="upper right", borderaxespad=0)

    x_min, x_max = ax.get_xlim(); ax.set_xlim(x_min, x_max + 1.5)
    ax.axis("off")

    # save fig and show
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', transparent=True)
    plt.show()


# ---------------------------------------------------------------------------
# Multilayer panel plot: Australia panels
# ---------------------------------------------------------------------------

# first, a helper function
def multilayer_geo_layout(
    G: nx.MultiGraph,
    scale: float = 0.16,
    panel_dx: float = 8.5,
    panel_dy: float = 6.2,
    seed: int = 1,
) -> tuple[dict, dict]:
    """
    Create the layout of geo_panels for the multilayer plot. Each ISP is drawn on a corresponding Australia map on its own panel. 

    Parameters
    ----------
    G : nx.MultiGraph
    scale : float
        Degrees-to-plot-units scaling for the Australia map.
    panel_dx, panel_dy : float
        Half-spacing between panel centres (horizontal, vertical).

    Returns
    -------
    (pos, layer_centers): tuple
        pos: dict mapping node -> (x, y)
        layer_centers: dict mapping layer name -> (cx, cy)
    """
    # determine centres first
    layer_centres = {
        "aarnet":    (-panel_dx,  panel_dy),
        "abb":       (0.0,        panel_dy),
        "optus":     (panel_dx,   panel_dy),
        "ixp":       (0.0,        0.0),
        "vocus":     (-panel_dx, -panel_dy),
        "telstra":   (0.0,       -panel_dy),
        "superloop": (panel_dx,  -panel_dy),
    }

    # determine positions of nodes based on lon-lat and panel positions
    pos = {}
    for n, data in G.nodes(data=True):
        layer = node_layer(n, data)
        lonlat = get_lon_lat(n, data)
        lon, lat = lonlat
        pos[n] = geo_to_panel_xy(
            lon,
            lat,
            center=layer_centres[layer],
            scale=scale,
        )

    return pos, layer_centres


def plot_multilayer_panels(
    G: nx.MultiGraph,
    scale: float = 0.16,
    panel_dx: float = 8.5,
    panel_dy: float = 6.2,
    figsize: tuple = (18, 13),
    save_dir: str = "images",
    filename: str = "multilayer_geo_panels",
) -> None:
    """
    Draw the full multilayer ISP supra-graph with each ISP in its own
    geographic Australia panel and IXPs in a central ring.
    """
    apply_rc()

    # determine position of layers and nodes within each layer
    pos, layer_centres = multilayer_geo_layout(
        G, scale=scale, panel_dx=panel_dx, panel_dy=panel_dy,
    )

    # load aus map
    au_states = load_au_state_polygons()

    # create figure
    fig, ax   = plt.subplots(figsize=figsize)

    # plot Australia mini-maps and panels (border around map)
    for layer in layer_centres:
        au_panel = transform_au_polygons_to_panel(
            au_states, center=layer_centres[layer], scale=scale
        )
        au_panel.plot(ax=ax, color="whitesmoke", edgecolor="dimgray",
                      linewidth=0.9, zorder=0)

    # panel bounding boxes
    map_width, map_height = panel_map_dimensions(scale)
    for layer, (cx, cy) in layer_centres.items():
        rect = plt.Rectangle(
            (cx - map_width / 2, cy - map_height / 2),
            map_width, map_height,
            fill=False, linewidth=0.8, alpha=0.25, linestyle="-",
        )
        ax.add_patch(rect)
    
    # add panel labels
    for layer, (cx, cy) in layer_centres.items():
        l_cx = cx + PANEL_LABEL_ADJUSTMENTS_X[layer]
        l_cy = cy + map_height / 2 + PANEL_LABEL_ADJUSTMENTS_Y[layer]

        ax.text(
            l_cx,
            l_cy,
            PANEL_LABELS[layer],
            ha="center",
            va="bottom",
            fontsize=24,
        )

    # split intra vs inter layer edges
    def _get_layer(n):
        return node_layer(n, G.nodes[n])

    intra, inter = [], []
    for u, v, k, d in edge_iter(G):
        if _get_layer(u) == _get_layer(v):
            intra.append((u, v, k, d))
        else:
            inter.append((u, v, k, d))

    # draw intra-layer edges
    seen = defaultdict(int)
    for u, v, k, d in intra:
        pair = tuple(sorted((str(u), str(v))))
        i = seen[pair]; seen[pair] += 1
        rad = curvature(seen, pair, i, step=0.12)
        lu = _get_layer(u)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=ISP_COLOURS.get(lu, "gray"),
            alpha=0.45, width=1.2,
            connectionstyle=f"arc3,rad={rad}",
        )

    # draw inter-layer edges
    seen = defaultdict(int)
    for u, v, k, d in inter:
        pair = tuple(sorted((str(u), str(v))))
        i = seen[pair]; seen[pair] += 1
        rad = curvature(seen, pair, i, step=0.10)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color="black", alpha=0.22, width=0.9,
            style="--", connectionstyle=f"arc3,rad={rad}",
        )

    # draw ISP nodes
    for isp in ISP_LIST:
        nodelist = [n for n, d in G.nodes(data=True) if node_isp(n, d) == isp]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist, ax=ax,
            node_size=120, alpha=0.85,
            node_color=ISP_COLOURS.get(isp, "gray"),
            node_shape=ISP_STYLES[isp],
            label=ISP_LABELS[isp],
            linewidths=0.5, edgecolors="black",
        )

    # draw IXP nodes
    ixp_nodes = [n for n, d in G.nodes(data=True) if is_ixp_node(n, d)]
    if ixp_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=ixp_nodes, ax=ax,
            node_size=120, alpha=0.95,
            node_color="black", node_shape="D", label="IXP",
        )

    # final formatting
    ax.set_aspect("equal")
    ax.axis("off")

    # save and show fig
    plt.savefig(f"{save_dir}/{filename}", bbox_inches="tight", transparent=True)
    plt.show()