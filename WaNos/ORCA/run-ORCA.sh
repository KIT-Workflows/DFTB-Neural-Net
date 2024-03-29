#!/bin/bash
source ~/.bashrc

export PATH="/home/ws/qs7669/ORCA/orca_4_1_0_linux_x86-64_shared_openmpi313:$PATH"
export LD_LIBRARY_PATH="/home/ws/qs7669/ORCA/orca_4_1_0_linux_x86-64_shared_openmpi313:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=1

module purge
module load gnu8/8.3.0
#module load intel/19.0.5.281
module load mpich/3.3.1
#module load openmpi3/3.1.0
export  INTEL_LICENSE_FILE=28518@scclic1.scc.kit.edu
orca orca.inp > orca.out

