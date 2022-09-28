import os
import numpy as np
from ase import Atoms
from ase.calculators.dftb import Dftb

class DftbMod(Dftb):
    """A modified version of the interface to dftb calculator, corrected the read
    charge process

    Warning:
    For Convienience, we implemented the population calculation in
    def read_charges[]. Otherwise we have to change the entire
    ase.Atoms class, which is not convinient.

    To Make Everything work, just uncomment the functions and
    then change the function get_charges to get_populations
    """

    implemented_properties = ['energy', 'forces', 'charges', 'stress']

    def read_charges(self):
        """Read the Population from the DFTB results.

        """
        infile = open(os.path.join(self.directory, 'detailed.out'), 'r')
        lines = infile.readlines()
        infile.close()

        qm_population= []
        for n, line in enumerate(lines):
            if ('Atom       Population' in line):
                chargestart = n + 1
                break
        else:
            print('Warning: did not find DFTB-Population')
            print('This is ok if flag SCC=NO')
            return None
        lines1 = lines[chargestart:(chargestart + len(self.atoms))]
        for line in lines1:
            qm_population.append(float(line.split()[-1]))

        return np.array(qm_population)

# Just for Comparison:
# This is the function that I have written.
def get_mulliken_charge(file_dir,file, nAtom):
    """
    Get the Mulliken charge for a particular configuration

    Input Argument:
    filedir: path of the directory that has the $(num).out (copied from detalied.out file)
    file: with extension


    Reference:
    Derived from ASE package (Calculator.dftb.py)
    """
    infile = open(os.path.join(file_dir, file), 'r')
    #fileIndex = os.path.splitext(file)[0] # Get rid of the extension for the filename and for future index

    # The following code comes from ASE calculator dftb
    lines = infile.readlines()
    infile.close()
    qm_charges = []
    for n, line in enumerate(lines):
        if ('Atom           Charge' in line):
            chargestart = n + 1
            break
    else:
        print('Warning: did not find DFTB-charges')
        print('This is ok if flag SCC=NO')
        return None
    lines1 = lines[chargestart:(chargestart + nAtom)]
    for line in lines1:
        #import pdb; pdb.set_trace()
        qm_charges.append(float(line.split()[-1]))

    return np.array(qm_charges)

