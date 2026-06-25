"""
visualisation/geo.py
--------------------
Geographic helper functions: loading the Australia state polygon,
converting lon/lat to panel coordinates, and positioning nodes.
"""

import math
import os
import urllib.request

import geopandas as gpd
import networkx as nx
from shapely import affinity

from src.constants import (
    AUS_LAT0, AUS_LON0,
    AUS_LAT_MIN, AUS_LAT_MAX,
    AUS_LON_MIN, AUS_LON_MAX,
    LON_CORRECTION,
    NE_ADMIN1_POLY_LOCAL,
    NE_ADMIN1_POLY_ZIP,
)


# ---------------------------------------------------------------------------
# Australia state polygons
# ---------------------------------------------------------------------------

def load_au_state_polygons() -> gpd.GeoDataFrame:
    """
    Load the Natural Earth 1:50m admin-1 state/province polygons,
    filtered to Australia only.

    The shapefile is downloaded on first call and cached locally as
    ``ne_50m_admin_1_states_provinces.zip``.
    """
    if not os.path.exists(NE_ADMIN1_POLY_LOCAL):
        urllib.request.urlretrieve(NE_ADMIN1_POLY_ZIP, NE_ADMIN1_POLY_LOCAL)

    gdf = gpd.read_file(
        f"zip://{os.path.abspath(NE_ADMIN1_POLY_LOCAL)}"
    ).to_crs("EPSG:4326")

    if "adm0_a3" in gdf.columns:
        gdf = gdf[gdf["adm0_a3"].astype(str).str.upper() == "AUS"]
    elif "admin" in gdf.columns:
        gdf = gdf[gdf["admin"].astype(str).str.lower() == "australia"]
    elif "ADMIN" in gdf.columns:
        gdf = gdf[gdf["ADMIN"].astype(str).str.lower() == "australia"]

    return gdf


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def geo_to_panel_xy(
    lon: float,
    lat: float,
    center: tuple[float, float],
    scale: float = 0.16,
) -> tuple[float, float]:
    """
    Project (lon, lat) to a 2-D panel position centred on *center*.

    The longitude is corrected by ``LON_CORRECTION`` to reduce
    east-west stretching at Australia's mean latitude.
    """
    cx, cy = center
    x = (lon - AUS_LON0) * LON_CORRECTION
    y = lat - AUS_LAT0
    return cx + scale * x, cy + scale * y


def transform_au_polygons_to_panel(
    au_states: gpd.GeoDataFrame,
    center: tuple[float, float],
    scale: float = 0.16,
) -> gpd.GeoDataFrame:
    """
    Transform the Australia state polygons to fit a panel centred
    on *center* at the given *scale*.

    Parameters
    ----------
    au_states : GeoDataFrame
        Output of ``load_au_state_polygons()``.
    center : (cx, cy)
        Panel origin in plot coordinates.
    scale : float
        Must match the ``scale`` used in ``geo_to_panel_xy``.

    Returns
    -------
    GeoDataFrame with transformed geometries.
    """
    cx, cy = center
    sx = scale * LON_CORRECTION
    sy = scale

    transformed = au_states.copy()
    transformed["geometry"] = transformed["geometry"].apply(
        lambda geom: affinity.translate(
            affinity.scale(
                geom,
                xfact=sx,
                yfact=sy,
                origin=(AUS_LON0, AUS_LAT0),
            ),
            xoff=cx - AUS_LON0,
            yoff=cy - AUS_LAT0,
        )
    )
    return transformed


# ---------------------------------------------------------------------------
# Node position helpers
# ---------------------------------------------------------------------------

def safe_pos(G: nx.Graph) -> dict:
    """
    Build a position dictionary ``{node: (lon, lat)}`` for all nodes
    in *G* that have valid finite lon/lat attributes.
    """
    pos = {}
    for n, d in G.nodes(data=True):
        lon, lat = d.get("lon"), d.get("lat")
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if math.isfinite(lon) and math.isfinite(lat):
            pos[n] = (lon, lat)
    return pos


def get_lon_lat(n, d: dict) -> tuple[float, float] | None:
    """
    Extract (lon, lat) from node attribute dict *d*.

    Returns None if either value is missing or non-numeric.
    """
    lon, lat = d.get("lon"), d.get("lat")
    try:
        return float(lon), float(lat)
    except (TypeError, ValueError):
        return None


def panel_map_dimensions(scale: float = 0.16) -> tuple[float, float]:
    """
    Return the (width, height) of an Australia mini-map at *scale*.

    Useful for positioning panel bounding boxes and labels.
    """
    width  = scale * (AUS_LON_MAX - AUS_LON_MIN) * LON_CORRECTION
    height = scale * (AUS_LAT_MAX - AUS_LAT_MIN)
    return width, height
