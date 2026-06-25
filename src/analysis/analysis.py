"""
analysis/failures.py
--------------------
Analysis of the failure simulation. Collect statistics from failures, and get 
per-city, per-ISP, and per-SRLG failure summaries.
"""

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from src.constants import ISP_LIST
from src.network.build import node_isp, is_ixp_node


# ---------------------------------------------------------------------------
# Filtering disconnect helpers
# ---------------------------------------------------------------------------

# column lists: 
#   separate splits for if looking at ISP (layer) level, whole graph level, or both
ISP_N_COMP_COLS = [f"{isp}_n_components" for isp in ISP_LIST]
SUPRA_N_COMP_COL = ["n_components"]
ALL_N_COMP_COLS = SUPRA_N_COMP_COL + ISP_N_COMP_COLS


def get_outage_mask(df: pd.DataFrame, include_global: bool = True) -> pd.Series:
    """
    Determine Boolean mask where disconnects have occurred (rows where number 
    of components > 1).

    Parameters
    ----------
    df : DataFrame
        Output of ``remove_single_srgs`` or ``remove_double_srgs``.
    include_global : bool
        If False, only ISP-level components are checked (local outages).
    """
    cols = ALL_N_COMP_COLS if include_global else ISP_N_COMP_COLS
    return (df[cols] > 1).any(axis=1)


def filter_outages(df: pd.DataFrame, local_only: bool = False) -> pd.DataFrame:
    """
    Return only the rows of df that correspond to disconnect events, using the 
    above mask.

    Parameters
    ----------
    local_only : bool
        If True, return rows with a local (ISP-level) disconnect but
        no global (supra) disconnect.
    """
    # firstly can find all outage instances, then differentiate between global
    # and look at local only outages.
    any_outage = get_outage_mask(df, include_global=True)
    global_outage = (df[SUPRA_N_COMP_COL] > 1).any(axis=1)
    
    # find local only (when outage but not a global one occurs)
    if local_only:
        return df.loc[any_outage & ~global_outage].reset_index(drop=True)
    return df.loc[any_outage].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Create specific failure DataFrame functions
# ---------------------------------------------------------------------------

def check_cities_removed(
    cities_kept: list,
    isp: str,
    node_df: pd.DataFrame,
) -> list:
    """
    Return list of sites that are present in an ISP's network but are 
    absent from a given list. Useful for calculating city-based stats
    (how often they are removed from the network).
    """
    # get list of known sites (nodes)
    isp_sites = list(node_df.loc[node_df['isp'] == isp, 'node_id'])
    # return a list if not present
    return [s for s in isp_sites if s not in cities_kept]


def create_city_fail_df(
    outages_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a city/site based DataFrame, with each row distinct as: 
    (failure, ISP, city).

    Each row records which site lost connectivity for which ISP, whether
    the failure was global, and the contributing SRLG pair. Failure IDs are
    therefore shared. 

    Parameters
    ----------
    outages_df : DataFrame
        Filtered subset of ``remove_double_srgs`` output.
    node_df : DataFrame
        Node table (columns: node_id, isp).

    Returns
    -------
    DataFrame with columns: failure_id, isp, city, global, srg_1, srg_2
    """
    outs = []
    for idx, row in outages_df.iterrows():
        fail_id = f"F{idx}"
        for isp in ISP_LIST:
            if row[f"{isp}_n_components"] <= 1:
                continue
            cities_kept = row[f"{isp}_present_sites"]
            cities_rm = check_cities_removed(cities_kept, isp, node_df)
            for city in cities_rm:
                outs.append({
                    'failure_id': fail_id,
                    'isp': isp,
                    'city': city,
                    'global': row['n_components'] > 1,
                    'srg_1': row.get('srg_id_1'),
                    'srg_2': row.get('srg_id_2'),
                })
    return pd.DataFrame(outs)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def count_fails(fails_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(failure, ISP, city) rows to one row per failure and get corresponding counts for number of ISPs and number of cities affected.

    Returns a DataFrame with columns:
        failure_id, n_cities, n_isps, is_global
    """
    return (
        fails_df
        .groupby("failure_id")
        .agg(
            n_cities =("city", "nunique"),
            n_isps =("isp", "nunique"),
            is_global =("global", "max"),
        )
        .reset_index()
    )

def fail_summary(removal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse results post failure analysis. Use outputted dataframe from the failure analysis. Counts number of each type of failure and overall trials.
    """
    fail_sum = {}
    
    # get some initial dfs with the specific outages
    all_outages_df = filter_outages(removal_df)
    local_only_outages_df = filter_outages(removal_df, local_only=True)
    global_mask = (all_outages_df[SUPRA_N_COMP_COL] > 1).any(axis=1)
    global_outages_df = all_outages_df.loc[global_mask].reset_index(drop=True)

    N_trials = len(removal_df) - 1 # minus 1 due to baseline
    N_disconnects = len(all_outages_df)
    N_no_disconnects = N_trials - N_disconnects

    # now find number of each type of disconnect
    N_local_outages = len(local_only_outages_df)
    N_global_outages = len(global_outages_df)

    # put stats in dictionary
    fail_sum['N_trials'] = N_trials
    fail_sum['N_disconnects'] = N_disconnects
    fail_sum['N_no_disconnects'] = N_no_disconnects
    fail_sum['N_local_disconnects'] = N_local_outages
    fail_sum['N_global_disconnects'] = N_global_outages

    return fail_sum


def record_srg_failures(outages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to analyse each individual SRLG, and count how many double-failure scenarios they appeared in, and counting local vs global disconnects.
    Currently just considers binary counts (whether a local disconnect occurred), rather than counting the number of ISPs disconnected, i.e. when a local disconnect occurs, may occur for multiple ISPs at the same time. At this point, only look at if local disconnect occurred, rather than how many at the same time. 
    """
    # first create unique list of SRLGs based on IDs, then initialise other cols
    unique_vals = pd.concat([outages_df['srg_id_1'], outages_df['srg_id_2']]).unique()
    local_count = {srg: 0 for srg in unique_vals}
    global_count = {srg: 0 for srg in unique_vals}
    local_only = {srg: 0 for srg in unique_vals}

    # go through each outage and increment counts for the corresponding SRLGs
    for _, row in outages_df.iterrows():
        s1, s2 = row['srg_id_1'], row['srg_id_2']
        local_count[s1] += 1
        local_count[s2] += 1
        if row['n_components'] > 1:
            global_count[s1] += 1
            global_count[s2] += 1
        else:
            local_only[s1] += 1
            local_only[s2] += 1

    # put in df
    out = pd.DataFrame(unique_vals, columns=['srg_id'])
    out['outages'] = out['srg_id'].map(local_count)
    out['global_outages'] = out['srg_id'].map(global_count)
    out['local_only_outages'] = out['srg_id'].map(local_only)
    return out


def create_srg_isp_df(single_srg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary indicator matrix, with SRLGs as rows, and ISPs as columns (SRLG x ISP). Indicates whether the ISP contributes an edge to each SRLG. Use 1 if yes, 0 if not. Use the single_srg analysis to calculate. Must contain the baseline row, since this is used as a reference.
    """

    baseline = single_srg_df.loc[single_srg_df['scenario'] == 'baseline']
    srg_rows = single_srg_df.loc[single_srg_df['scenario'] != 'baseline']
    srg_list = list(srg_rows['srg_id'].dropna())

    out_df = pd.DataFrame(srg_list, columns=['srg_id'])
    out_df = out_df.reindex(columns=['srg_id'] + ISP_LIST)

    # go through rows and check if edges decreased compared to avg. Will tell us if ISP had any edges corresponding to that SRLG.
    for _, row in srg_rows.iterrows():
        srg = row['srg_id']
        for isp in ISP_LIST:
            baseline_edges = baseline[f"{isp}_edges"].values[0]
            out_df.loc[out_df['srg_id'] == srg, isp] = (
                1 if row[f"{isp}_edges"] < baseline_edges else 0
            )

    return out_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# City network statistics
# ---------------------------------------------------------------------------

def _city_reps_in_graph(G: nx.MultiGraph, site_id: str) -> list:
    """Return all (site_id, isp) nodes present in G for a given site."""
    return [(site_id, isp) for isp in ISP_LIST if (site_id, isp) in G]


def isp_subgraph(G: nx.MultiGraph, isp: str) -> nx.MultiGraph:
    """Return version of G corresponding to an ISP layer (only that ISP's nodes)"""
    nodelist = [n for n, d in G.nodes(data=True) if node_isp(n, d) == isp]
    return G.subgraph(nodelist).copy()


def city_outage_dataframe(
    G_full: nx.MultiGraph,
    df_fails: pd.DataFrame,
    site_list: list,
) -> pd.DataFrame:
    """
    Build a per-site summary combining graph degree metrics with
    outage counts from the failure DataFrame. Supra node degree includes inter-layer edges. Local fails excludes global, but includes number of ISP layers affected in a specific instance.

    Parameters
    ----------
    G_full : nx.MultiGraph
        Full supra-graph.
    df_fails : pd.DataFrame
        Output of ``create_fail_df``.
    site_list : list of str
        All site IDs to include.

    Returns
    -------
    DataFrame with one row per site.
    """
    # create a dictionary with the ISP subgraphs
    isp_graphs = {isp: isp_subgraph(G_full, isp) for isp in ISP_LIST}
    rows = []

    # go through each site
    for site in site_list:
        row  = {"site": site}
        reps = _city_reps_in_graph(G_full, site)

        # count how many layers the node is present in
        row["layers"] = len(reps)

        # sum node degree across layers (including IXP edges)
        supra_deg_sum = 0
        for n in reps:
            supra_deg_sum += G_full.degree(n)
        row["supra_degree_sum"] = supra_deg_sum

        # go through stats for each ISP
        isp_total = 0
        for isp in ISP_LIST:
            H = isp_graphs[isp]
            n = (site, isp)
            if n in H:
                row[f"{isp}_degree"] = H.degree(n)
                isp_total += H.degree(n)
            else:
                row[f"{isp}_degree"] = 0

        # can find IXP count by minusing supra degree 
        row["ixps"] = supra_deg_sum - isp_total

        # outage counts
        # first check how many times an ISP's city failed (sum over all instances)
        city_check = df_fails['city'] == site
        row["isp_fails"] = int(city_check.sum())

        # then find number of global and local outage occurrences
        # local 
        local_fail_check = (df_fails['city'] == site) & (df_fails['global'] == False)
        row["local_fails"] = int(local_fail_check.sum())

        # global
        df_agg = (
            df_fails
            .drop(columns=['isp'])
            .drop_duplicates()
        )
        global_fail_check = (df_agg['city'] == site) & (df_agg['global'] == True)
        row["global_fails"] = int(global_fail_check.sum())

        # add rows and output
        rows.append(row)

    return pd.DataFrame(rows)
