
# FeGB_PtableSeg_FromFirstPrinciples_Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data/actions/workflows/ci.yml/badge.svg)](https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data/actions/workflows/ci.yml)

Data and analysis code for reproducing the figures and results in:

> **A high-throughput ab initio study of elemental segregation and cohesion at ferritic-iron grain boundaries**
> Han Lin Mai, Xiang-Yuan Cui, Tilmann Hickel, Jörg Neugebauer, Simon Ringer
> [arXiv:2503.05640](https://arxiv.org/abs/2503.05640)

## Overview

This repository provides:

- **`2025_03_02_ptable_Fe_GB_df.csv.gz` / `.pkl.gz`** -- The main dataset containing DFT-computed segregation energies, work of separation, DDEC6 bond orders, Voronoi descriptors, and magnetic moments for ~90 elements across 6 CSL grain boundaries in bcc Fe.
- **`FeGB_PtableSeg/`** -- Python package with plotting functions and DDEC6/Chargemol analysis tools.
- **`QuickStart.ipynb`** -- Notebook to regenerate all manuscript figures.
- **`SupplementaryFigures.ipynb`** -- Supplementary figure generation.
- **`Data/`** -- Example VASP+Chargemol output for the S11-RA110-S3-32-He-19 configuration.

## Grain Boundaries Studied

| GB | Sigma | N_atoms | Tilt Axis |
|---|---|---|---|
| S3-RA110-S1-11 | 3 | 72 | [110] |
| S3-RA110-S1-12 | 3 | 48 | [110] |
| S5-RA001-S210 | 5 | 76 (doubled) | [001] |
| S5-RA001-S310 | 5 | 80 (doubled) | [001] |
| S9-RA110-S2-21 | 9 | 68 | [110] |
| S11-RA110-S3-32 | 11 | 42 | [110] |

## Installation

### Using pip (recommended)

```bash
pip install git+https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data
```

For development (with test dependencies):

```bash
git clone https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data.git
cd FeGB_PtableSeg_FromFirstPrinciples_Data
pip install -e ".[dev]"
```

### Using conda/mamba

```bash
git clone https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data.git
cd FeGB_PtableSeg_FromFirstPrinciples_Data
mamba env create -f environment.yaml
mamba activate FeGB_PtableSegData
```

## Quick Start

```python
import pandas as pd

# Load the dataset (CSV or pickle)
df = pd.read_csv("2025_03_02_ptable_Fe_GB_df.csv.gz")
# or: df = pd.read_pickle("2025_03_02_ptable_Fe_GB_df.pkl.gz", compression="gzip")

# Filter to a specific element and GB
ni_s3 = df[(df["element"] == "Ni") & (df["GB"] == "S3_RA110_S1_11")]
print(ni_s3[["site", "E_seg", "Wsep_RGS_min"]].head())
```

See `QuickStart.ipynb` for full plotting examples reproducing each manuscript figure.

## Package Modules

### `FeGB_PtableSeg.chargemol`

Parse DDEC6 Chargemol output files and compute area-normalised summed bond orders (ANSBO) across cleavage planes:

```python
from FeGB_PtableSeg.chargemol import ChargemolAnalysis

ca = ChargemolAnalysis("Data/S11-RA110-S3-32-He-19")
coords, ansbo_profile = ca.get_ANSBO_profile()
result = ca.analyse_ANSBO()  # layer_boundaries, cleavage_coord, ANSBO_profile
```

### `FeGB_PtableSeg.plotters`

Publication-quality plotting functions for periodic-table heatmaps, segregation energy profiles, and GB-resolved scatter plots. Includes GB symmetry mappings, LaTeX labels, and color/marker dictionaries.

## Dataset Columns

The dataset has 90 columns and 4,289 rows. Key columns are listed below; for the full set inspect `df.columns`.

### Identifiers

| Column | Type | Description |
|---|---|---|
| `job_name` | str | Job identifier in `GB_element_site` format |
| `GB` | str | Grain boundary name (e.g. `S11_RA110_S3_32`) |
| `element` | str | Segregant element symbol |
| `Z` | int | Atomic number of segregant |
| `site` | int | Atomic site index (0-indexed) |
| `GB_site` | str | Combined `GB-site` identifier |
| `GB_element_site` | str | Combined `GB_element_site` identifier |
| `solute_idx` | float | Index of the solute atom in the structure |
| `equivalent_sites` | list[int] | Symmetrically equivalent sites |
| `site_multiplicity` | int | Number of equivalent sites |

### Energetics (units: eV for energies, J/m2 for Wsep)

| Column | Type | Description |
|---|---|---|
| `E_seg` | float | Segregation energy (relaxed) |
| `E_seg_unrel` | float | Segregation energy (unrelaxed / static) |
| `Wsep_RGS_min` | float | Minimum rigid grain separation work of separation |
| `Wsep_RGS_list` | list[float] | Wsep at each cleavage plane |
| `Wsep_RGS_cleavage_planes` | list[float] | Fractional z-coords of cleavage planes |
| `Wsep_RGS_min_pure` | float | Pure Fe reference Wsep |
| `R_Wsep_RGS` | float | Wsep ratio: segregated / pure |
| `R_Wsep_RGS_lst` | list[float] | Wsep ratio at each cleavage plane |

### Geometry (units: angstrom)

| Column | Type | Description |
|---|---|---|
| `dist_GB` | float | Distance from GB plane (relaxed structure) |
| `dist_GB_unrel` | float | Distance from GB plane (unrelaxed structure) |
| `site_z` | float | Fractional z-coordinate of solute site (relaxed) |
| `site_z_unrel` | float | Fractional z-coordinate of solute site (unrelaxed) |
| `structure` | json | Relaxed pymatgen Structure (JSON-serialised) |
| `structure_unrel` | json | Unrelaxed pymatgen Structure (JSON-serialised) |

### Magnetic Properties

| Column | Type | Description |
|---|---|---|
| `magmoms` | array | Magnetic moments of all atoms (relaxed) |
| `magmom_solute` | float | Solute atom magnetic moment (relaxed) |
| `magmoms_unrel` | array | Magnetic moments of all atoms (unrelaxed) |
| `magmom_solute_unrel` | float | Solute atom magnetic moment (unrelaxed) |
| `convergence` | bool | Whether ionic relaxation converged |

### Cleavage / Pure GB Reference

| Column | Type | Description |
|---|---|---|
| `cleavage_planes` | list[float] | Cleavage plane fractional z-coordinates |
| `cp_names` | list[str] | Cleavage plane identifiers |
| `pure_cleavage_planes` | list[float] | Pure Fe cleavage plane coordinates |
| `pure_cleavage_energies` | list[float] | Pure Fe cleavage energies at each plane |
| `pure_min_wsep_rigid` | float | Pure Fe minimum rigid Wsep (J/m2) |
| `convergence_pureGB` | bool | Whether the pure GB reference converged |
| `magmoms_pureGB` | array | Magnetic moments of the pure Fe GB |

### DDEC6 Bond Order Analysis

| Column | Type | Description |
|---|---|---|
| `DDEC6_min_ANSBO` | float | Minimum ANSBO (duplicate of `DDEC6_ANSBO_min`) |
| `DDEC6_ANSBO_min` | float | Minimum area-normalised summed bond order |
| `DDEC6_ANSBO_profile` | list[float] | ANSBO at each cleavage plane |
| `DDEC6_ANSBO_cleavage_coords` | list[float] | Cleavage plane coordinates for ANSBO |
| `DDEC6_ANSBO_atomic_layers` | list[float] | Atomic layer boundaries |
| `DDEC6_ANSBO_within_range` | bool/list | Whether ANSBO is within expected range |
| `pure_ca_results` | dict | Raw Chargemol analysis results for pure GB |
| `pure_DDEC6_min_ANSBO` | float | Pure Fe reference minimum ANSBO |
| `pure_DDEC6_ANSBO_profile` | list[float] | Pure Fe ANSBO profile |
| `pure_DDEC6_ANSBO_cleavage_coords` | list[float] | Pure Fe ANSBO cleavage coordinates |
| `pure_DDEC6_ANSBO_atomic_layers` | list[float] | Pure Fe atomic layer boundaries |
| `pure_DDEC6_ANSBO_within_range` | bool/list | Pure Fe ANSBO range check |
| `R_DDEC6_ANSBO` | float | ANSBO ratio: segregated / pure |
| `R_DDEC6_ANSBO_lst` | list[float] | ANSBO ratio at each cleavage plane |
| `ANSBO_Wsep_RGS_corr_vals` | list[float] | ANSBO-Wsep correlation values |

### Voronoi Nearest-Neighbour Descriptors (units: angstrom/angstrom2/angstrom3)

16 Voronoi polyhedra descriptors are provided for both relaxed and unrelaxed (`_unrel` suffix) structures:

| Column Pattern | Description |
|---|---|
| `VorNN_CoordNo` | Coordination number |
| `VorNN_tot_vol` | Total Voronoi polyhedra volume |
| `VorNN_tot_area` | Total face area |
| `VorNN_{volumes,vertices,areas,distances}_{std,mean,min,max}` | Statistics of polyhedra faces |

## DFT Parameters

All calculations performed with VASP using the PBE functional:

- **ENCUT:** 400 eV
- **ISMEAR:** 1 (Methfessel-Paxton), SIGMA = 0.2
- **KSPACING:** 0.5
- **POTCAR:** Fe (8e) PAW_PBE
- **Relaxation:** ISIF=2, NSW=200, EDIFF=1E-5, EDIFFG=-0.01
- **Magnetic:** ISPIN=2, MAGMOM ~ 3 uB per Fe

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use this data or code, please cite:

```bibtex
@article{mai2025fegb_ptable,
  title={A high-throughput ab initio study of elemental segregation and cohesion at ferritic-iron grain boundaries},
  author={Mai, Han Lin and Cui, Xiang-Yuan and Hickel, Tilmann and Neugebauer, J{\"o}rg and Ringer, Simon},
  journal={arXiv preprint arXiv:2503.05640},
  year={2025}
}
```

## License

[MIT](LICENSE) -- Copyright (c) 2025 Han Lin Mai, The University of Sydney and the Max Planck Institute of Sustainable Materials.
