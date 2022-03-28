#! /bin/bash

module purge

# set PATH to DFTB+ / load module
module load intel/19.0.5.281

export PATH="/home/ws/gt5111/dftbplus/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_NPROCS

python run_dftb+.py

srun dftb+

exit 0