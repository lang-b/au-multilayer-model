"""
network/build.py
--------------
Functions for constructing the multilayer ISP supra-graph from
node/edge CSVs and attaching IXP interconnection nodes.
"""

import pandas as pd
import networkx as nx

from src.constants import ISP_LIST


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def norm(s: str) -> str:
    # lowercase and strip string
    return str(s).strip().lower()


def is_ixp_node(n, data: dict) -> bool:
    # check if belongs to ixp layer
    layer = norm(data.get("layer", ""))
    return "ixp" in layer


def node_isp(n, data: dict) -> str | None:
    # return isp name from node
    isp = data.get("isp")
    if isp:
        return norm(isp)
    return None


def node_layer(n, data: dict) -> str | None:
    # return layer from node
    if is_ixp_node(n, data):
        return "ixp"
    isp = node_isp(n, data)
    if isp in ISP_LIST:
        return isp
    return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_isp_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> nx.MultiGraph:
    """
    Build a multilayer MultiGraph from node and edge dataframes.

    Each node is keyed as (site_id, isp) and carries geographic and
    topological metadata sites_df.  Each edge uses its
    link_id as the MultiGraph key so parallel edges are preserved.

    Parameters
    ----------
    nodes_df : DataFrame
        Columns: node_id, isp  (one row per ISP node)
    edges_df : DataFrame
        Columns: isp, link_id, u, v, srg_cons, srg_ideal, ...
        Must already have a 'key' column (isp + "." + link_id).
    sites_df : DataFrame
        Columns: site_id, lat, lon, city, kind, population

    Returns
    -------
    nx.MultiGraph
    """
    # initialise multigraph
    G = nx.MultiGraph()

    # add nodes from node list, including site information
    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        isp = row['isp']
        site_row = sites_df[sites_df['site_id'] == node_id]

        G.add_node(
            (node_id, isp),
            layer=f"isp:{isp}",
            isp=isp,
            site_id=node_id,
            lat=site_row["lat"].values[0] if len(site_row) else None,
            lon=site_row["lon"].values[0] if len(site_row) else None,
            city=site_row["city"].values[0] if len(site_row) else None,
            kind=site_row["kind"].values[0] if len(site_row) else None,
            population=site_row["population"].values[0] if len(site_row) else None,
            ixp=0,
        )

    # add edges from edge list, including label and specifying which isp
    for _, row in edges_df.iterrows():
        isp = row['isp']
        u = (row["u"], isp)
        v = (row["v"], isp)
        key = row['key']
        attrs = row.drop(labels=["link_id", "u", "v", "key"]).to_dict()
        G.add_edge(u, v, key=key, **attrs)

    return G


def attach_ixps(
    G: nx.MultiGraph,
    ixp_df: pd.DataFrame,
    site_df: pd.DataFrame,
) -> nx.MultiGraph:
    """
    Add IXP nodes to G and connect each ISP to the IXPs it peers at.

    IXP nodes are keyed as (ixp_id, 'ixp').  Each ISP–IXP edge is
    keyed as <isp>.ixp.<ixp_id>.  The 'ixp' attribute on ISP
    nodes is set to 1 if they peer at any IXP; the IXP node's 'ixp'
    attribute counts how many ISPs peer there.

    Parameters
    ----------
    G : nx.MultiGraph
        Graph returned by ``build_isp_graph``.
    ixp_df : DataFrame
        Columns: ixp, <isp_name> (boolean columns, one per ISP)
    site_df : DataFrame
        Same sites table used in ``build_isp_graph``.

    Returns
    -------
    nx.MultiGraph  (modified in-place and returned)
    """
    # add ixp nodes from ixp list
    for ixp in ixp_df['ixp']:
        site_row = site_df[site_df['site_id'] == ixp]
        G.add_node(
            (ixp, 'ixp'),
            layer="ixp",
            isp='ixp',
            site_id=ixp,
            lat=site_row["lat"].values[0] if len(site_row) else None,
            lon=site_row["lon"].values[0] if len(site_row) else None,
            city=site_row["city"].values[0] if len(site_row) else None,
            kind=site_row["kind"].values[0] if len(site_row) else None,
            population=site_row["population"].values[0] if len(site_row) else None,
            ixp=0,
        )

    # connect isps to their ixps
    for isp in ISP_LIST:
        isp_col = pd.Series(ixp_df[isp]).astype(bool)
        for ixp in list(ixp_df['ixp'][isp_col]):
            u = (ixp, 'ixp')
            v = (ixp, isp)
            key = f"{isp}.ixp.{ixp}"
            G.add_edge(u, v, key=key)
            if v in G.nodes:
                G.nodes[v]['ixp'] = 1
            if u in G.nodes:
                G.nodes[u]['ixp'] = G.nodes[u].get('ixp', 0) + 1

    return G


def prepare_edge_df(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Small function to add the composite 'key' column (isp.link_id) 
    to an edge DataFrame. Call this before passing edge_df to 
    build_isp_graph.
    """
    df = edge_df.copy()
    df['key'] = df['isp'] + "." + df['link_id']
    return df
