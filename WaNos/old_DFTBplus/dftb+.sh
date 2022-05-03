#! /bin/bash

module purge

# set PATH to DFTB+ / load module
export PATH="/home/ws/qs7669/dftbplus/install/bin:$PATH"
export LD_LIBRARY_PATH="/home/ws/qs7669/anaconda3/envs/dftb-nn/lib:$PATH"
export OMP_NUM_THREADS=$SLURM_NPROCS

python run_dftb+.py

exit 0
