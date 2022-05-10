#! /bin/bash

module purge
module load gnu8/8.3.0
module load openblas/0.3.7

export PATH="/home/ws/gt5111/opt/dftb+/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_NPROCS

python run_dftb+.py
dftb+ > output

exit 0
# /home/celso/Wanos_2020/Leticia/WaNos/symFunc_all.param

# /home/celso/Wanos_2020/Leticia/WaNos/Model.tar
