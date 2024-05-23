
# FePtableSeg_FromFirstPrinciples

Includes the data and code for reproducing analysis in the manuscript

## Installation
You can install this package either using `pip` directly or by setting up a conda environment using `mamba`. Below are the instructions for both methods:

### Using pip

To install the latest version of this package directly from GitHub, you can use the following command:
```bash
pip install git+https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data
```
### Using mamba

If you prefer to create a dedicated environment for this package using `mamba`, you can use the provided `environment.yml` file. This approach is recommended if you want to ensure that all specific binary dependencies are correctly installed. Follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data.git
cd FeGB_PtableSeg_FromFirstPrinciples_Data
```
2. Create the environment:
```bash
mamba env create -f environment.yml
```

3. Activate the new environment:
```bash
mamba activate your_env
```

## Usage

After installation, you can import the DataFrame by:

```python
import pandas as pd

df = pd.read_pickle("ptable_Fe_GB_df.pkl.gz", compression="gzip")
```

## Dataset Units

In the provided dataframe, the columns are as follows:

| Column Name                   | Type                           | Units       | Description                                                                                             |
|-------------------------------|--------------------------------|-------------|---------------------------------------------------------------------------------------------------------|
| job_name                      | str                            |             | Job name in "GB"-"element"-"site" format                                                                |
| GB                            | str                            |             | Grain boundary name, one of ['S11_RA110_S3_32', 'S3_RA110_S1_11', 'S3_RA110_S1_12', 'S5_RA001_S210', 'S5_RA001_S310', 'S9_RA110_S2_21'] |
| element                       | str                            |             | The element of the segregant, in symbol form, i.e. Ni = Nickel                                           |
| site                          | numpy.int64                    |             | The site occupied by the solute (0-indexed, i.e. first atom is site no. 0)                               |
| equivalent_sites              | list containing ints           |             | The symmetrically equivalent sites in the GB                                                            |
| site_multiplicity             | int                            |             | The multiplicity of the site at the GB, i.e. the length of the equivalent_sites list                     |
| dist_GB                       | numpy.float64                  | Å           | The distance from the center of the _pure_ Fe GB                                                        |
| structure                     | json-ised pymatgen structure representation v2023.11.10 |             | The relaxed structure                                                                                   |
| E_seg                         | numpy.float64                  | eV          | The segregation energy of the solute in this site                                                       |
| magmoms                       | numpy.ndarray                  |             | The magnetic moments of each atom in the final relaxed structure                                        |
| magmom_solute                 | numpy.float64                  |             | The magnetic moment of the segregant                                                                    |
| Wsep_RGS_min                  | numpy.float64                  | J/m²        | The minimum rigid work of separation                                                                    |
| Wsep_RGS_list                 | list containing numpy.float64  | J/m²        | The list of rigid work of separations computed                                                          |
| Wsep_RGS_cleavage_planes      | list containing numpy.float64  | J/m²        | The list of rigid work of separations computed                                                          |
| Wsep_RGS_min_pure             | list containing numpy.float64  | J/m²        | The list of rigid work of separations computed for the pure GB                                          |
| DDEC6_ANSBO_min               | numpy.float64                  |             | The minimum DDEC6 computed area-normalised summed bond orders in the structure                          |
| DDEC6_ANSBO_profile           | list containing numpy.float64  |             | The list of DDEC6 computed area-normalised summed bond orders in the structure                          |
| DDEC6_ANSBO_cleavage_coords   | list containing numpy.float64  |             | The coordinates at which the DDEC6 ANSBO values were computed                                           |
| pure_DDEC6_min_ANSBO          | numpy.float64                  |             | The minimum value of ANSBO computed in the pure Fe GB                                                   |
| R_Wsep_RGS                    | numpy.float64                  |             | The ratio of the segregated cohesion value to the R                                                     |
| R_Wsep_RGS_lst                | list containing numpy.float64  |             | The list of ratios of segregated cohesion values                                                        |
| R_DDEC6_ANSBO                 | numpy.float64                  |             | The ratio of the DDEC6 ANSBO value to the R                                                             |
| R_DDEC6_ANSBO_lst             | list containing numpy.float64  |             | The list of ratios of DDEC6 ANSBO values                                                                |
| ANSBO_Wsep_RGS_corr_vals      | list containing numpy.float64  |             | The list of correlation values between ANSBO and Wsep_RGS                                               |
| VorNN_CoordNo                 | numpy.float64                  |             | Pymatgen computed Voronoi coordination number                                                          |
| VorNN_tot_vol                 | numpy.float64                  | Å³          | Total volume of the Voronoi polyhedra around an atom                                                    |
| VorNN_tot_area                | numpy.float64                  | Å²          | Total area of the faces of the Voronoi polyhedra around an atom                                         |
| VorNN_volumes_std             | numpy.float64                  | Å³          | Standard deviation of the volumes of the Voronoi polyhedra around an atom                               |
| VorNN_volumes_mean            | numpy.float64                  | Å³          | Mean volume of the Voronoi polyhedra around an atom                                                     |
| VorNN_volumes_min             | numpy.float64                  | Å³          | Minimum volume of the Voronoi polyhedra around an atom                                                  |
| VorNN_volumes_max             | numpy.float64                  | Å³          | Maximum volume of the Voronoi polyhedra around an atom                                                  |
| VorNN_vertices_std            | numpy.float64                  |             | Standard deviation of the number of vertices of the Voronoi polyhedra around an atom                    |
| VorNN_vertices_mean           | numpy.float64                  |             | Mean number of vertices of the Voronoi polyhedra around an atom                                         |
| VorNN_vertices_min            | numpy.float64                  |             | Minimum number of vertices of the Voronoi polyhedra around an atom                                      |
| VorNN_vertices_max            | numpy.float64                  |             | Maximum number of vertices of the Voronoi polyhedra around an atom                                      |
| VorNN_areas_std               | numpy.float64                  | Å²          | Standard deviation of the areas of the faces of the Voronoi polyhedra around an atom                    |
| VorNN_areas_mean              | numpy.float64                  | Å²          | Mean area of the faces of the Voronoi polyhedra around an atom                                          |
| VorNN_areas_min               | numpy.float64                  | Å²          | Minimum area of the faces of the Voronoi polyhedra around an atom                                       |
| VorNN_areas_max               | numpy.float64                  | Å²          | Maximum area of the faces of the Voronoi polyhedra around an atom                                       |
| VorNN_distances_std           | numpy.float64                  | Å           | Standard deviation of the distances to the faces of the Voronoi polyhedra around an atom                |
| VorNN_distances_mean          | numpy.float64                  | Å           | Mean distance to the faces of the Voronoi polyhedra around an atom                                      |
| VorNN_distances_min           | numpy.float64                  | Å           | Minimum distance to the faces of the Voronoi polyhedra around an atom                                   |
| VorNN_distances_max           | numpy.float64                  | Å           | Maximum distance to the faces of the Voronoi polyhedra around an atom                                   |
| dist_GB_unrel                 | numpy.float64                  | Å           | Distance from the center of the _pure_ Fe GB before relaxation                                          |
| E_seg_unrel                   | numpy.float64                  | eV          | Segregation energy of the solute in this site before relaxation                                         |
| structure_unrel               | json-ised pymatgen structure representation v2023.11.10 |             | The unrelaxed structure                                                                                 |
| magmoms_unrel                 | numpy.ndarray                  |             | Magnetic moments of each atom in the initial unrelaxed structure                                        |
| magmom_solute_unrel           | numpy.float64                  |             | Magnetic moment of the segregant before relaxation                                                      |
| VorNN_CoordNo_unrel           | numpy.float64                  |             | Voronoi coordination number before relaxation                                                           |
| VorNN_tot_vol_unrel           | numpy.float64                  | Å³          | Total volume of the Voronoi polyhedra around an atom before relaxation                                  |
| VorNN_tot_area_unrel          | numpy.float64                  | Å²          | Total area of the faces of the Voronoi polyhedra around an atom before relaxation                       |
| VorNN_volumes_std_unrel       | numpy.float64                  | Å³          | Standard deviation of the volumes of the Voronoi polyhedra around an atom before relaxation             |
| VorNN_volumes_mean_unrel      | numpy.float64                  | Å³          | Mean volume of the Voronoi polyhedra around an atom before relaxation                                   |
| VorNN_volumes_min_unrel       | numpy.float64                  | Å³          | Minimum volume of the Voronoi polyhedra around an atom before relaxation                                |
| VorNN_volumes_max_unrel       | numpy.float64                  | Å³          | Maximum volume of the Voronoi polyhedra around an atom before relaxation                                |
| VorNN_vertices_std_unrel      | numpy.float64                  |             | Standard deviation of the number of vertices of the Voronoi polyhedra around an atom before relaxation  |
| VorNN_vertices_mean_unrel     | numpy.float64                  |             | Mean number of vertices of the Voronoi polyhedra around an atom before relaxation                       |
| VorNN_vertices_min_unrel      | numpy.float64                  |             | Minimum number of vertices of the Voronoi polyhedra around an atom before relaxation                    |
| VorNN_vertices_max_unrel      | numpy.float64                  |             | Maximum number of vertices of the Voronoi polyhedra around an atom before relaxation                    |
| VorNN_areas_std_unrel         | numpy.float64                  | Å²          | Standard deviation of the areas of the faces of the Voronoi polyhedra around an atom before relaxation  |
| VorNN_areas_mean_unrel        | numpy.float64                  | Å²          | Mean area of the faces of the Voronoi polyhedra around an atom before relaxation                        |
| VorNN_areas_min_unrel         | numpy.float64                  | Å²          | Minimum area of the faces of the Voronoi polyhedra around an atom before relaxation                     |
| VorNN_areas_max_unrel         | numpy.float64                  | Å²          | Maximum area of the faces of the Voronoi polyhedra around an atom before relaxation                     |
| VorNN_distances_std_unrel     | numpy.float64                  | Å           | Standard deviation of the distances to the faces of the Voronoi polyhedra around an atom before relaxation|
| VorNN_distances_mean_unrel    | numpy.float64                  | Å           | Mean distance to the faces of the Voronoi polyhedra around an atom before relaxation                    |
| VorNN_distances_min_unrel     | numpy.float64                  | Å           | Minimum distance to the faces of the Voronoi polyhedra around an atom before relaxation                 |
| VorNN_distances_max_unrel     | numpy.float64                  | Å           | Maximum distance to the faces of the Voronoi polyhedra around an atom before relaxation                 |


