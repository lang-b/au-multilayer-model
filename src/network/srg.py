"""
network/srg.py
------------
Functions for building Shared Risk Group (SRG) dataframes and
derived graph objects from the edge list and SRG city table.
"""

import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------------
# Reconfigure SRLG dataframe, to add list of edges part of the SRLG
# ---------------------------------------------------------------------------

def new_srg_df(og_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialise new srg_df with different column names and an edge list for each
    SRG ready to be populated.
    """
    srg_df = og_df.copy()
    srg_df.rename(columns={"start": "u", "end": "v"}, inplace=True)
    srg_df['edges'] = [[] for _ in range(len(srg_df))]
    return srg_df


def srg_link_grouping(
    srg_df: pd.DataFrame,
    srgs: str,
    u: str,
    v: str,
    k: str,
    isp: str,
) -> pd.DataFrame:
    """
    Append edge ``(k, u, v, isp)`` to every SRLG listed in `srgs`
    that identifies with that edge.

    `srgs` is a comma-separated string of SRG IDs as stored in the
    edge list (e.g. ``"SRG_ABC001, SRG_DEF002"``).
    """
    for srg in srgs.split(', '):
        srg = srg.strip()
        # find location of SRG
        mask = srg_df['srg_id'] == srg
        if not mask.any():
            continue
        # add edge
        idx = srg_df.index[mask][0]
        srg_df.at[idx, 'edges'].append((k, u, v, isp))
    return srg_df


def fill_srg_list(
    edge_df: pd.DataFrame,
    srg_og_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Populate conservative and ideal SRLG DataFrames from the edge list.

    Each row in the returned DataFrames corresponds to one SRLG and
    contains a list of ``(link_id, u, v, isp)`` tuples for the edges
    that belong to it.

    Parameters
    ----------
    edge_df : DataFrame
        Must contain columns: link_id, u, v, isp, srg_cons, srg_ideal
    srg_og_df : DataFrame
        Raw SRLG city table (columns: srg_id, start, end, srg_type, ...)

    Returns
    -------
    srg_c_df, srg_i_df : tuple of DataFrames
        Conservative and ideal SRLG DataFrames.
    """
    # create lists of all the columns; could have just used iterrows...
    link_k = list(edge_df['link_id'])
    link_u = list(edge_df['u'])
    link_v = list(edge_df['v'])
    link_isp = list(edge_df['isp'])
    srg_cons = list(edge_df['srg_cons'])
    srg_ideal = list(edge_df['srg_ideal'])

    # initialise new dfs for both conservative and ideal srlg sets
    srg_c_df = new_srg_df(srg_og_df)
    srg_i_df = new_srg_df(srg_og_df)

    # go through all edges and add them to the correpsonding srlgs
    for i in range(len(edge_df)):
        srg_c_df = srg_link_grouping(
            srg_c_df, srg_cons[i], link_u[i], link_v[i], link_k[i], link_isp[i]
        )
        srg_i_df = srg_link_grouping(
            srg_i_df, srg_ideal[i], link_u[i], link_v[i], link_k[i], link_isp[i]
        )

    # drop SRGs that have no associated edges
    srg_c_df = srg_c_df[srg_c_df['edges'].map(len) > 0].reset_index(drop=True)
    srg_i_df = srg_i_df[srg_i_df['edges'].map(len) > 0].reset_index(drop=True)

    return srg_c_df, srg_i_df


# ---------------------------------------------------------------------------
# SRLG graph objects
# ---------------------------------------------------------------------------

def build_srg_graph(
    srg_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> nx.MultiGraph:
    """
    Build a MultiGraph where each edge represents one SRLG. 

    Nodes are sites form sites_df and links are SRLGs between sites.

    Parameters
    ----------
    srg_df : DataFrame
        Output of ``fill_srg_list`` (columns: srg_id, u, v, edges, ...)
    sites_df : DataFrame
        Site table (columns: site_id, lat, lon, city, kind, population)
    """
    G = nx.MultiGraph()

    # now iterrows for each site and add
    for _, site_row in sites_df.iterrows():
        G.add_node(
            site_row['site_id'],
            layer="srg",
            site_id=site_row['site_id'],
            lat=site_row["lat"],
            lon=site_row["lon"],
            city=site_row["city"],
            kind=site_row["kind"],
            population=site_row["population"],
        )

    # add edges for SRLGs
    for _, row in srg_df.iterrows():
        u = row["u"]
        v = row["v"]
        if u not in G.nodes or v not in G.nodes:
            continue
        key = row['srg_id']
        attrs = row.drop(labels=["srg_id", "u", "v"]).to_dict()
        G.add_edge(u, v, key=key, **attrs)

    return G


def build_srg_dif_graph(
    srg_c_df: pd.DataFrame,
    srg_i_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> nx.MultiGraph:
    """
    Build a graph containing only the SRLGs present in the ideal set
    but absent from the conservative set (the difference).

    This highlights the additional shared risks captured by the more
    optimistic SRG assignment.
    """
    G = nx.MultiGraph()

    # iterrows for each site and add
    for _, site_row in sites_df.iterrows():
        G.add_node(
            site_row['site_id'],
            layer="srg",
            site_id=site_row['site_id'],
            lat=site_row["lat"],
            lon=site_row["lon"],
            city=site_row["city"],
            kind=site_row["kind"],
            population=site_row["population"],
        )

    # create sets of both then find the difference
    srg_c_ids = set(srg_c_df['srg_id'])
    srg_i_ids = set(srg_i_df['srg_id'])
    diff = srg_i_ids - srg_c_ids

    # create the difference graph
    for _, row in srg_i_df.iterrows():
        u = row["u"]
        v = row["v"]
        if u not in G.nodes or v not in G.nodes:
            continue
        key = row['srg_id']
        if key not in diff:
            continue
        attrs = row.drop(labels=["srg_id", "u", "v"]).to_dict()
        G.add_edge(u, v, key=key, **attrs)

    return G
