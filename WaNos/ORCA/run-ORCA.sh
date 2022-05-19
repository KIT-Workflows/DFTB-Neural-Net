#!/bin/bash
source ~/.bashrc

module purge
module spider gnu8/8.3.0
module spider intel/19.0.5.281
module spider mpich/3.3.1
export  INTEL_LICENSE_FILE=28518@scclic1.scc.kit.edu
/home/ws/gt5111/ORCA/orca/orca_4_1_0_linux_x86-64_openmpi313/orca *.inp > orca.out
