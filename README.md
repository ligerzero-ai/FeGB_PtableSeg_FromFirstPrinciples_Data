
# FePtableSeg_FromFirstPrinciples

Includes the data and code for reproducing analysis in the manuscript

## Installation
You can install this package either using `pip` directly or by setting up a conda environment using `mamba`. Below are the instructions for both methods:

### Using pip

To install the latest version of this package directly from GitHub, you can use the following command:

pip install git+https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data

### Using mamba

If you prefer to create a dedicated environment for this package using `mamba`, you can use the provided `environment.yml` file. This approach is recommended if you want to ensure that all specific binary dependencies are correctly installed. Follow these steps:

1. Clone the repository:

git clone https://github.com/ligerzero-ai/FeGB_PtableSeg_FromFirstPrinciples_Data.git
cd FeGB_PtableSeg_FromFirstPrinciples_Data

2. Create the environment:

mamba env create -f environment.yml

3. Activate the new environment:

mamba activate your_env

## Usage

After installation, you can import the DataFrame by:

```python
import pandas as pd

df = pd.read_pickle("ptable_Fe_GB_df.pkl.gz", compression="gzip")
```

## Dataset Units

In the provided dataframe, the columns are as follows:

'job_name': "str", Job name in "GB"-"element"-"site" format
 'GB': "str", Grain boundary name, one of ['S11_RA110_S3_32', 'S3_RA110_S1_11', 'S3_RA110_S1_12', 'S5_RA001_S210', 'S5_RA001_S310', 'S9_RA110_S2_21'], 'S11_RA110_S3_32' -> $\Sigma$11[110](3-32)
 'element': "str", the element of the segregant, in symbol form, i.e. Ni = Nickel
 'site': "numpy.int64", the site occupied by the solute (0-indexed, i.e. first atom is site no. 0)
 'equivalent_sites': "list containing ints", the symmetrically equivalent sites in the GB
 'site_multiplicity': "int", the multiplicity of the site at the GB, i.e. the length of the equivalent_sites list
 'dist_GB': "numpy.float64", "\AA", the distance from the center of the _pure_ Fe GB.
 'structure': "json-ised pymatgen structure representation v2023.11.10", the relaxed structure 
 'E_seg': "numpy.float64" (eV), the segregation energy of the solute in this site
 'magmoms': "numpy.ndarray" the magnetic moments of each atom in the final relaxed structure 
 'magmom_solute': "numpy.float64" the magnetic moment of the segregant
 'Wsep_RGS_min': "numpy.float64" (J/m^2) the minimum rigid work of separation
 'Wsep_RGS_list': "list containing numpy.float64" (J/m^2) the list of rigid work of separations computed
 'Wsep_RGS_cleavage_planes': "list containing numpy.float64" (J/m^2) the list of rigid work of separations computed
 'Wsep_RGS_min_pure': "list containing numpy.float64" (J/m^2) the list of rigid work of separations computed
 'DDEC6_ANSBO_min': the minimum DDEC6 computed area-normalised summed bond orders in the structure
 'DDEC6_ANSBO_profile': the list of DDEC6 computed area-normalised summed bond orders in the structur
 'DDEC6_ANSBO_cleavage_coords': the coordinates at which the DDEC6 ANSBO values were computed
 'pure_DDEC6_min_ANSBO': the minimum value of ANSBO computed in the pure Fe GB
 'R_Wsep_RGS', the ratio of the segregated cohesion value to the R
 'R_Wsep_RGS_lst',
 'R_DDEC6_ANSBO',
 'R_DDEC6_ANSBO_lst',
 'ANSBO_Wsep_RGS_corr_vals',
 'VorNN_CoordNo', pymatgen computed Voronoi coordination number
 'VorNN_tot_vol', 
 'VorNN_tot_area',
 'VorNN_volumes_std',
 'VorNN_volumes_mean',
 'VorNN_volumes_min',
 'VorNN_volumes_max',
 'VorNN_vertices_std',
 'VorNN_vertices_mean',
 'VorNN_vertices_min',
 'VorNN_vertices_max',
 'VorNN_areas_std',
 'VorNN_areas_mean',
 'VorNN_areas_min',
 'VorNN_areas_max',
 'VorNN_distances_std',
 'VorNN_distances_mean',
 'VorNN_distances_min',
 'VorNN_distances_max',
 'dist_GB_unrel',
 'E_seg_unrel',
 'structure_unrel',
 'magmoms_unrel',
 'magmom_solute_unrel',
 'VorNN_CoordNo_unrel',
 'VorNN_tot_vol_unrel',
 'VorNN_tot_area_unrel',
 'VorNN_volumes_std_unrel',
 'VorNN_volumes_mean_unrel',
 'VorNN_volumes_min_unrel',
 'VorNN_volumes_max_unrel',
 'VorNN_vertices_std_unrel',
 'VorNN_vertices_mean_unrel',
 'VorNN_vertices_min_unrel',
 'VorNN_vertices_max_unrel',
 'VorNN_areas_std_unrel',
 'VorNN_areas_mean_unrel',
 'VorNN_areas_min_unrel',
 'VorNN_areas_max_unrel',
 'VorNN_distances_std_unrel',
 'VorNN_distances_mean_unrel',
 'VorNN_distances_min_unrel',
 'VorNN_distances_max_unrel',
 'dist_GB_unrel',
 'E_seg_unrel',
 'structure_unrel',
 'magmoms_unrel',
 'magmom_solute_unrel',
 'VorNN_CoordNo_unrel',
 'VorNN_tot_vol_unrel',
 'VorNN_tot_area_unrel',
 'VorNN_volumes_std_unrel',
 'VorNN_volumes_mean_unrel',
 'VorNN_volumes_min_unrel',
 'VorNN_volumes_max_unrel',
 'VorNN_vertices_std_unrel',
 'VorNN_vertices_mean_unrel',
 'VorNN_vertices_min_unrel',
 'VorNN_vertices_max_unrel',
 'VorNN_areas_std_unrel',
 'VorNN_areas_mean_unrel',
 'VorNN_areas_min_unrel',
 'VorNN_areas_max_unrel',
 'VorNN_distances_std_unrel',
 'VorNN_distances_mean_unrel',
 'VorNN_distances_min_unrel',
 'VorNN_distances_max_unrel',
 'dist_GB_unrel',
 'structure_unrel',
 'E_seg_unrel',
 'magmoms_unrel',
 'magmom_solute_unrel',
 'VorNN_CoordNo_unrel',
 'VorNN_tot_vol_unrel',
 'VorNN_tot_area_unrel',
 'VorNN_volumes_std_unrel',
 'VorNN_volumes_mean_unrel',
 'VorNN_volumes_min_unrel',
 'VorNN_volumes_max_unrel',
 'VorNN_vertices_std_unrel',
 'VorNN_vertices_mean_unrel',
 'VorNN_vertices_min_unrel',
 'VorNN_vertices_max_unrel',
 'VorNN_areas_std_unrel',
 'VorNN_areas_mean_unrel',
 'VorNN_areas_min_unrel',
 'VorNN_areas_max_unrel',
 'VorNN_distances_std_unrel',
 'VorNN_distances_mean_unrel',
 'VorNN_distances_min_unrel',
 'VorNN_distances_max_unrel'

