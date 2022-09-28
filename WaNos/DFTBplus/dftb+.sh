#! /bin/bash

module purge

# set PATH to DFTB+ / load module

module load gnu7/7.3.0
module load openblas/0.2.20

export PATH="/home/ws/qs7669/gnu7/opt/dftb+/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_NPROCS

python run_dftb+.py
dftb+ > output

exit 0
# /home/celso/Wanos_2020/Leticia/WaNos/symFunc_all.param

# /home/celso/Wanos_2020/Leticia/WaNos/Model.tar
