"""
constants.py
------------
Shared constants for the Australian ISP topology analysis.
Includes ISP metadata, colour palettes, labels, and graphing styles.
"""

import math
import matplotlib as mpl

# ---------------------------------------------------------------------------
# ISP metadata
# ---------------------------------------------------------------------------

ISP_LIST = ['aarnet', 'abb', 'optus', 'superloop', 'telstra', 'vocus']

ISP_LABELS = {
    'aarnet': 'AARNet',
    'abb': 'ABB',
    'optus': 'Optus',
    'superloop': 'Superloop',
    'telstra': 'Telstra',
    'vocus': 'Vocus',
}

# Okabe-Ito colour palette (colour-blind friendly)
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

ISP_COLOURS = {
    "aarnet": OKABE_ITO[0],
    "abb": OKABE_ITO[2],
    "optus": OKABE_ITO[4],
    "telstra": OKABE_ITO[6],
    "ixp": OKABE_ITO[7],
    "superloop": OKABE_ITO[5],
    "vocus": OKABE_ITO[1],
}

ISP_STYLES = {
    "aarnet": 'o',
    "abb": '^',
    "optus": 's',
    "telstra": 'p',
    "ixp": 'd',
    "superloop": 'h',
    "vocus": 'v',
}

# ---------------------------------------------------------------------------
# Edge type metadata
# ---------------------------------------------------------------------------

_dark2 = mpl.colormaps['Dark2'].colors

EDGE_COLOURS = {
    'sub': _dark2[0],
    'road': _dark2[1],
    'rail': _dark2[4],
    'rail_road': _dark2[2],
    'mix': _dark2[3],
}

EDGE_STYLES = {
    'sub': '-',
    'road': ':',
    'rail': '--',
    'rail_road': '--',
    'mix': '-.',
}

EDGE_LABELS = {
    'sub': 'Submarine',
    'road': 'Road',
    'rail': 'Rail',
    'rail_road': 'Rail and Road',
    'mix': 'Mix',
}

# Terrestrial-grouped styles (used in some map plots)
TER_EDGE_STYLES = {
    'sub': '-',
    'road': '--',
    'rail': '--',
    'rail_road': '--',
    'mix': '-.',
}

TER_EDGE_LABELS = {
    'sub': 'Submarine (SUB)',
    'road': 'Terrestrial (TER)',
    'rail': 'Terrestrial (TER)',
    'rail_road': 'Terrestrial (TER)',
    'mix': 'Mix',
}

# ---------------------------------------------------------------------------
# SRLG metadata
# ---------------------------------------------------------------------------

_paired = mpl.colormaps['Paired'].colors

SRG_COLOURS = {
    'ideal': _paired[0],
    'conservative': _paired[1],
}

SRG_STYLES = {
    'ideal': '.',
    'conservative': 's',
}

# ---------------------------------------------------------------------------
# Australia map
# ---------------------------------------------------------------------------

# Approximate Australia bounding box (WGS-84)
AUS_LON_MIN, AUS_LON_MAX = 110, 155
AUS_LAT_MIN, AUS_LAT_MAX = -45, -10

AUS_LON0 = (AUS_LON_MIN + AUS_LON_MAX) / 2   # 132.5
AUS_LAT0 = (AUS_LAT_MIN + AUS_LAT_MAX) / 2   # -27.5

# Cosine correction so east-west distances are not over-stretched
LON_CORRECTION = math.cos(math.radians(AUS_LAT0))

# Natural Earth admin-1 state/province polygons
NE_ADMIN1_POLY_ZIP = ("https://naturalearth.s3.amazonaws.com/50m_cultural/"
                         "ne_50m_admin_1_states_provinces.zip")
NE_ADMIN1_POLY_LOCAL = "../data/external/ne_50m_admin_1_states_provinces.zip"

# ---------------------------------------------------------------------------
# City metadata
# ---------------------------------------------------------------------------

# note: CAPITAL_SITES is not currently used
CAPITAL_SITES = [
    'SYD_1', 'MEL_1', 'BNE_1', 'PER_1',
    'ADL_1', 'HBA_1', 'DRW_1', 'CBR_1',
]

CITY_MAP = {
    "SYD_1": "Sydney",
    "MEL_1": "Melbourne",
    "ADL_1": "Adelaide",
    "BNE_1": "Brisbane",
    "PER_1": "Perth",
    "CBR_1": "Canberra",
    "DRW_1": "Darwin",
    "HBA_1": "Hobart",
    "ROK_1": "Rockhampton",
    "LST_1": "Launceston",
    "TSV_1": "Townsville",
    "ISA_1": "Mount Isa",
    "PHE_1": "Port Hedland",
}

# Manual label offsets (lon, lat) for map annotation
POS_ADJUST = {
    'ADL_1': ( 2.2,  0.8),
    'BNE_1': ( 2.1,  0.0),
    'CBR_1': ( 2.4, -0.4),
    'DRW_1': ( 2.2,  0.0),
    'HBA_1': ( 2.4, -0.4),
    'MEL_1': ( 2.4,  0.0),
    'PER_1': (-2.2,  0.0),
    'SYD_1': ( 2.4,  0.0),
    'LST_1': ( 2.0,  0.9),
    'PHE_1': (-2.1,  0.9),
    'TSV_1': ( 2.2,  0.8),
    'ROK_1': ( 2.2,  0.8),
    'ISA_1': (-2.0, -0.8),
    'HGD_1': (-2.0,  1.2),
    'LRE_1': ( 0.0, -1.3),
    'PUG_1': ( 2.0,  0.9),
}

# ---------------------------------------------------------------------------
# Plot status colours (used in failure / heatmap plots)
# ---------------------------------------------------------------------------

STATUS_COLOURS = {
    "None": "0.6",
    "Local Only": "orange",
    "Global": "red",
}

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------

PANEL_LABELS = {
        "aarnet": "AARNet", 
        "abb": "Aussie Broadband", 
        "optus": "Optus",
        "ixp": "IXP",
        "vocus": "Vocus", 
        "telstra": "Telstra", 
        "superloop": "Superloop",
    }

PANEL_LABEL_ADJUSTMENTS_X = {
        "aarnet": -1.5, 
        "abb": -0.1, 
        "optus": -1.8,
        "ixp": -2.25,
        "vocus": -1.8, 
        "telstra": -1.8, 
        "superloop": -1.3,
    }

PANEL_LABEL_ADJUSTMENTS_Y = {
        "aarnet": 0.2, 
        "abb": 0.3, 
        "optus": 0.2,
        "ixp": -5.35,
        "vocus": -5.35, 
        "telstra": -5.35, 
        "superloop": -5.35,
    }