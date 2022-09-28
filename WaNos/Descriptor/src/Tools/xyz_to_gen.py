## Purpose of this script:
# To convert the input_geometry xyz file into the dftb+ input geometry file 
# Using Atomic Simulation Environment
 
from ase import Atoms
from ase.io import write, read

temp_Atoms = read("input_geometry.xyz")
write( "input_geometry.gen", temp_Atoms)

