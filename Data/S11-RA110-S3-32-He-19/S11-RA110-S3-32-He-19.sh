#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --account=pawsey0380
#SBATCH --job-name=S11-RA110-S3-32-He-19.sh
#SBATCH --time=24:00:00
#SBATCH --partition=work
#SBATCH --export=NONE
#SBATCH --exclusive

module load vasp/5.4.4
cd "$PBS_O_WORKDIR"

ulimit -s unlimited
run_cmd="srun --export=ALL -N 1 -n 128"

source /software/projects/pawsey0380/hmai/mambaforge/bin/activate custodian

echo 'import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), UnconvergedErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler()]
jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                settings_override = [{"dict": "INCAR", "action": {"_set": {"NSW": 1, "LAECHG": True, "LCHARGE": True, "NELM": 300, "EDIFF": 1E-5}}}])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()' > StaticImage-DDEC6-custodian.py

python StaticImage-DDEC6-custodian.py $run_cmd vasp_std &> vasp.log

echo '<net charge>
0.0 <-- specifies the net charge of the unit cell (defaults to 0.0 if nothing specified)
</net charge>
<periodicity along A, B, and C vectors>
.true. <--- specifies whether the first direction is periodic
.true. <--- specifies whether the second direction is periodic
.true. <--- specifies whether the third direction is periodic
</periodicity along A, B, and C vectors>
<atomic densities directory complete path>
/home/hmai/chargemol_09_26_2017/atomic_densities/
</atomic densities directory complete path>
<charge type>
DDEC6 <-- specifies the charge type (DDEC3 or DDEC6)
</charge type>
<compute BOs>
.true. <-- specifies whether to compute bond orders or not
</compute BOs>' > job_control.txt

OMP_NUM_THREADS=128
export OMP_NUM_THREADS
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/atomic_densities/
export PATH=$PATH:/home/hmai/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux
$run_cmd Chargemol_09_26_2017_linux_parallel

# Cleanup the data so it doesn't flood the drive
rm CHG* CHGCAR* PROCAR* WAVECAR* EIGENVAL* REPORT* IBZKPT* REPORT* DOSCAR.* XDATCAR*

directory_name=$(basename "$PWD")
tar -czvf "${directory_name}.tar.gz" --exclude="AECCAR*" .
find . ! -name "${directory_name}.tar.gz" ! -name 'OUTCAR' ! -name 'CONTCAR' ! -name 'vasprun.xml' ! -name 'starter*' ! -name 'AECCAR*' -type f -exec rm -f {} +
