# Australian Internet Topology Multilayer Model
A robustness analysis of Australia's Internet using a multilayer network model. The analysis considers individual network topologies from Internet Service Providers (ISPs) in Australia. We evaluate the interactions of these networks, both through Shared Risk Link Groups (SRLGs) and peering via Internet Exchange Points (IXPs). We provide the original maps (topologies) and associated metadata, and the code used to develop the multilayer model and perform the failure analysis. We study six major Australian ISPs: AARNet, Aussie Broadband (ABB), Optus, Superloop, Telstra, and Vocus. 

The corresponding pre-print is provided at: https://arxiv.org/abs/2606.18737

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── maps/                    # Maps (topologies) and corresponding metadata
│   │   ├── lists/                   # Supporting lists, detailing methods and sources used
│   ├── processed/
│   │   ├── graphs/                  # Saved NetworkX graph objects (.pickle)
│   │   └── tabular/                 # Simulation output CSVs
│   ├── clean/
│   │   └── lists/                   # Clean lists (edge lists, node lists, etc.) extracted from the raw maps
│   ├── external/                    # External data (e.g. AUS map for plots)
│   └── support_docs/                # Supporting documentation for defining SRLGs
├── notebooks/
│   ├── 01_data_loading.ipynb        # Load CSVs and build the supra-graph
│   ├── 02_srg_grouping.ipynb        # Assign edges to SRLGs, build SRLG graphs
│   ├── 03_simulation.ipynb          # Run single and double SRLG removal scenarios
│   ├── 04_analysis.ipynb            # Failure summaries by city, ISP, and SRLG
│   └── 05_visualisation.ipynb       # All publication figures
├── src/
│   ├── constants.py                 # ISP metadata, colours, geographic bounds
│   ├── network/
│   │   ├── build.py                 # Build supra-graph from node/edge CSVs
│   │   └── srg.py                   # SRLG DataFrame and graph construction
│   ├── simulation/
│   │   └── failures.py              # Initial network metrics and SRLG failure scenario runners
│   ├── analysis/
│   │   └── analysis.py              # Disconnect summary tables and analysis per SRLG, city, and ISP
│   └── visualisation/
│       ├── geo.py                   # Geographic helpers (polygon loading, projections)
│       ├── maps.py                  # Geographic Australian map plots
│       └── plots.py                 # Statistical plots (bars, violins, heatmaps, scatter)
├── outputs/
│   ├── figures/                     # Generated plots (.pdf and .svg)
│   └── reports/                     # Results from analysing the failure simulation
├── config.yaml                      # Key file paths and folders
├── requirements.txt
└── README.md
```

---

## Methodology

### Multilayer Graph

Each ISP is modelled as a separate layer of a `networkx.MultiGraph`. Nodes are network sites (PoPs, exchanges, regional nodes) keyed as `(site_id, isp)` and carry geographic coordinates, city labels, and population metadata. IXP nodes are shared across layers and keyed as `(ixp_id, 'ixp')`.

### Shared Risk Link Groups

SRLGs represent geographic corridors where multiple fibre links share a common physical path (e.g. along the same road, rail corridor, or submarine cable). Two SRLG assignment strategies are used:

- **Conservative** — only links with confirmed shared routing are grouped
- **Ideal** — links that plausibly share a corridor based on geography are also grouped

We only use the conservative set for the main analysis.

### Failure Simulation

For each SRLG (or pair of SRLGs), all member edges are simultaneously removed from the supra-graph and a wide set of connectivity metrics are recomputed:

- Number of connected components (global and per-ISP)
- Largest connected component fraction
- Average shortest path length, diameter, clustering coefficient
- IXP dependency (metric delta with and without IXP nodes)

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/lang-b/au-multilayer-model.git
cd au-multilayer-model
pip install -r requirements.txt
```

Specifically, use matplotlib 3.10.0+ for the new managua cmap, which we use for one of the Australia maps in `05_visualisation.ipynb`. 

---

## Running the Analysis

Run the notebooks in order. Each notebook saves its outputs to `data/processed/features/` so that downstream notebooks can be run independently without re-running slow upstream steps.

```
01_data_loading.ipynb:  builds G_full
02_srg_grouping.ipynb:  builds G_srg_c and saves srg_c.csv
03_simulation.ipynb:    saves single.csv, double.csv
04_analysis.ipynb:      saves failures.csv, city_isp_fail_counts.csv, srg_summary.csv, ...
05_visualisation.ipynb: saves figures to outputs/figures/
```

---

## Key Output Files

| File | Description |
|---|---|
| `single.csv` | Results from the single failure analysis for all SRLGs |
| `double.csv` | Results from the double failure analysis for all SRLG pairs |
| `failures.csv` | Per-(failure, ISP, city) outage records |
| `city_isp_fail_counts.csv` | Summed values from failures.csv to count number of cities and ISPs affected |
| `srg_summary.csv` | Per-SRLG local and global outage counts |
| `srg_isp_usage.csv` | Binary SRLG × ISP membership matrix |
| `city_sum.csv` | Per-city degree metrics and outage counts |

---

## Configuration

Edit `config.yaml` to set file paths and analysis parameters without touching notebook code:

```yaml
paths:
  node_list: ../data/clean/lists/nodes.csv
  edge_list: ../data/clean/lists/edges.csv
  ixp_list: ../data/clean/lists/ixps.csv
  site_list: ../data/clean/lists/sites.csv
  srg_list: ../data/clean/lists/srgs.csv
  figures_folder: ../outputs/figures/
  tabular_folder: ../data/processed/tabular/
  graph_folder: ../data/processed/graphs/
  results_folder: ../outputs/results/
```