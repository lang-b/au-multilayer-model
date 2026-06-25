"""
simulation/metrics.py
---------------------
Network metric computation and SRG failure scenario runners.

Find key metrics while also running simulation.
"""

import itertools
from collections import defaultdict
import ast

import numpy as np
import pandas as pd
import networkx as nx

from src.constants import ISP_LIST
from src.network.build import is_ixp_node, node_isp


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def isp_subgraph(G: nx.MultiGraph, isp: str) -> nx.MultiGraph:
    """Create a copy of G consisting only of a specific ISP's layer network."""
    nodelist = [n for n, d in G.nodes(data=True) if node_isp(n, d) == isp]
    return G.subgraph(nodelist).copy()


def remove_ixps(G: nx.MultiGraph) -> nx.MultiGraph:
    """Create a copy of G with all IXPs and interlayer edges removed."""
    ixps = [n for n, d in G.nodes(data=True) if is_ixp_node(n, d)]
    H = G.copy()
    H.remove_nodes_from(ixps)
    return H


def components_index(G: nx.Graph) -> dict:
    """Map an integer ID for every component in the graph to all nodes."""
    comp_id = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_id[n] = i
    return comp_id


def pair_fraction_connected(nodes: list, comp_id: dict) -> float:
    """
    Determine fraction of node pairs within same component.
    Returns NaN if fewer than 2 nodes are present in component.
    """
    nodes = [n for n in nodes if n in comp_id]
    if len(nodes) < 2:
        return float("nan")
    total = ok = 0
    for a, b in itertools.combinations(nodes, 2):
        total += 1
        ok += 1 if comp_id[a] == comp_id[b] else 0
    return ok / total if total else float("nan")


# ---------------------------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------------------------

def compute_metrics_for_graph(
    G: nx.MultiGraph,
    sites: list | None = None,
) -> dict:
    """
    Compute connectivity and resilience metrics for graph G.

    Metrics include global supra-graph statistics, capital-city 
    connectivity fractions, IXP-dependency measures, and per-ISP breakdowns.

    Used for initial graph, but mainly after removing edges to see effect.
    """

    out = {}

    ### global counts ### 
    n = G.number_of_nodes()
    out["nodes_total"] = n
    out["edges_total"] = G.number_of_edges()

    # components
    comp    = components_index(G)
    n_comps = len(set(comp.values())) if n else 0
    out["n_components"] = n_comps
    # find largest CC fraction
    if n:
        counts = defaultdict(int)
        for cid in comp.values():
            counts[cid] += 1
        out["largest_cc_frac"] = max(counts.values()) / n if counts else 0.0
    else:
        out["largest_cc_frac"] = 0.0

    # Find cities in largest CC 
    largest_cc = set(max(nx.connected_components(G), key=len)) if n else set()

    def city_reps_in_graph(site_id):
        return [n for n in [(site_id, isp) for isp in ISP_LIST] if n in G]

    present_sites = []
    in_lcc = 0
    for site in sites:
        reps = city_reps_in_graph(site)
        if not reps:
            continue
        present_sites.append(site)
        # check if any of the layer versionsd of the city are present
        if any(r in largest_cc for r in reps):
            in_lcc += 1
    out["in_lcc"] = in_lcc
    out["present_sites"] = present_sites


    # diameter / average shortest path (simple graph)
    G_simple = nx.Graph(G)
    if nx.is_connected(G_simple):
        out["diameter"] = nx.diameter(G_simple)
        out["avg_shortest_path"] = nx.average_shortest_path_length(G_simple)
    else:
        out["diameter"] = np.inf
        out["avg_shortest_path"] = np.inf
    out["avg_cluster_coeff"] = nx.average_clustering(G_simple)


    ### per-ISP metrics ###
    for isp in ISP_LIST:
        H = isp_subgraph(G, isp)
        out[f"{isp}_nodes"] = H.number_of_nodes()
        out[f"{isp}_edges"] = H.number_of_edges()

        # initialise other metrics
        if H.number_of_nodes() == 0:
            out[f"{isp}_n_components"] = 0
            out[f"{isp}_largest_cc_frac"] = float("nan")
            out[f"{isp}_cap_pair_frac"] = float("nan")
            out[f"{isp}_diameter"] = float("nan")
            out[f"{isp}_avg_shortest_path"] = float("nan")
            out[f"{isp}_avg_cluster_coeff"] = float("nan")
            out[f"{isp}_in_lcc"] = 0
            out[f"{isp}_present_sites"] = []
            continue

        # component metrics
        compH  = components_index(H)
        countH = defaultdict(int)
        for cid in compH.values():
            countH[cid] += 1

        out[f"{isp}_n_components"] = len(set(compH.values()))
        out[f"{isp}_largest_cc_frac"] = max(countH.values()) / H.number_of_nodes()

        lcc_H = set(max(nx.connected_components(H), key=len))

        # find sites present in largest CC
        isp_present = []
        isp_in_lcc  = 0
        for site in sites:
            node = (site, isp)
            if node in lcc_H:
                isp_in_lcc += 1
                isp_present.append(site)
        out[f"{isp}_in_lcc"] = isp_in_lcc
        out[f"{isp}_present_sites"] = isp_present

        # find diameter/path lengths/clustering
        H_simple = nx.Graph(H)
        if nx.is_connected(H_simple):
            out[f"{isp}_diameter"] = nx.diameter(H_simple)
            out[f"{isp}_avg_shortest_path"] = nx.average_shortest_path_length(H_simple)
        else:
            out[f"{isp}_diameter"] = np.inf
            out[f"{isp}_avg_shortest_path"] = np.inf
        out[f"{isp}_avg_cluster_coeff"] = nx.average_clustering(H_simple)

    return out


# ---------------------------------------------------------------------------
# SRLG Failure Simulations
# ---------------------------------------------------------------------------

def remove_single_srgs(
    srg_df: pd.DataFrame,
    G: nx.MultiGraph,
    baseline_row: dict,
    site_list: list,
) -> pd.DataFrame:
    """
    Perform single SRLG analysis by iterating through all SRLGs, removing,
    then calculating metrics.

    Parameters
    ----------
    srg_df : DataFrame
        SRLG DataFrame
    G : nx.MultiGraph
        Full supra-graph
    baseline_row : dict
        Pre-computed baseline metrics (from ``compute_metrics_for_graph``).
    site_list : list of str
        All site IDs to pass to ``compute_metrics_for_graph``.

    Returns
    -------
    DataFrame sorted by ``largest_cc_frac`` ascending (worst failures first).
    """

    # initialise df using baseline_row to start.
    rows = [baseline_row]

    # go through every SRG
    for _, row in srg_df.iterrows():
        srg_id  = row['srg_id']
        # get triples for all edges within the given SRLG.
        triples = ast.literal_eval(row['edges'])
        # create G copy to remove edges from
        G_sub   = G.copy()

        # remove all edges associated with the SRG.
        for k, u, v, isp in triples:
            u_isp = (u, isp)
            v_isp = (v, isp)
            if G_sub.has_edge(u_isp, v_isp):
                G_sub.remove_edge(u_isp, v_isp)

        # compute the new metrics and add to output
        m = compute_metrics_for_graph(G_sub, site_list)
        m["scenario"] = "single removal" # single removal
        m["srg_id"] = srg_id
        m["edges_removed"] = len(triples)
        rows.append(m)

    result = pd.DataFrame(rows)
    return result.sort_values("largest_cc_frac", ascending=True).reset_index(drop=True)


def remove_double_srgs(
    srg_df: pd.DataFrame,
    G: nx.MultiGraph,
    baseline_row: dict,
    site_list: list,
) -> pd.DataFrame:
    """
    Perform double SRLG analysis by iterating through all pair combinations of SRLGs, removing, then calculating metrics. All unordered combinations of two SRLGs are analysed.

    Parameters
    ----------
    srg_df : DataFrame
        SRLG DataFrame
    G : nx.MultiGraph
        Full supra-graph (not modified).
    baseline_row : dict
        Pre-computed baseline metrics.
    site_list : list of str
        All site IDs to pass to ``compute_metrics_for_graph``.

    Returns
    -------
    DataFrame sorted by ``largest_cc_frac`` ascending (worst failures first).
    """

    # initialise df using baseline_row to start.
    rows = [baseline_row]
    row_list = list(srg_df.iterrows())

    # go through SRLG combinations using itertools
    for (_, row1), (_, row2) in itertools.combinations(row_list, 2):
        srg_id_1 = row1['srg_id']
        srg_id_2 = row2['srg_id']

        # get corresponding triples for the edges for both SRLGs
        triples_1 = ast.literal_eval(row1['edges'])
        triples_2 = ast.literal_eval(row2['edges'])

        # create G copy to remove edges from
        G_sub = G.copy()

        # iterate through all edges within both SRLGs
        for k, u, v, isp in triples_1 + triples_2:
            u_isp = (u, isp)
            v_isp = (v, isp)
            if G_sub.has_edge(u_isp, v_isp):
                G_sub.remove_edge(u_isp, v_isp)

        # compute new metrics and record
        m = compute_metrics_for_graph(G_sub, site_list)
        m["scenario"] = "double removal"
        m["srg_id_1"] = srg_id_1
        m["srg_id_2"] = srg_id_2
        m["edges_removed"] = len(triples_1) + len(triples_2)
        m["edges_removed_1"] = len(triples_1)
        m["edges_removed_2"] = len(triples_2)
        rows.append(m)

    result = pd.DataFrame(rows)
    return result.sort_values("largest_cc_frac", ascending=True).reset_index(drop=True)
