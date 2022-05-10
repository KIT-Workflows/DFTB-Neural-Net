#!/bin/bash
source ~/.bashrc

export PATH="/home/ws/qs7669/ORCA/orca_4_1_0_linux_x86-64_shared_openmpi313:$PATH"
export LD_LIBRARY_PATH="/home/ws/qs7669/ORCA/orca_4_1_0_linux_x86-64_shared_openmpi313:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=1

module purge
module spider gnu8/8.3.0
module spider intel/19.0.5.281
module spider mpich/3.3.1
export  INTEL_LICENSE_FILE=28518@scclic1.scc.kit.edu
orca *.inp > orca.out

