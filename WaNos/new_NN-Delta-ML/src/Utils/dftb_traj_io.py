from ase import Atoms
from ase import Atom

def loop_to_lines_xyz(loop_idx, natoms, lines_ok):
    """
    Conver the loop index to lines object in xyz file


        Args:
            loop_idx: index of the current loop
            natoms:   Number of atoms for the given configuration
            lines_ok: Lines of the xyz file (with blank lines removed)


    Does not return the first 2 line (NAtoms + 'MD Iter' Line)

    Assume that in xyz file, each loop will take (natoms + 2) lines
    """

    start_line_idx = loop_idx * (natoms + 2) + 2 # Avoid first 2 lines
    end_line_idx   = (loop_idx + 1) * (natoms + 2)

    return lines_ok[start_line_idx:end_line_idx]


def read_dftb_atom_from_traj(filename='geo_end.xyz', writeTrajectory=False):
    """
    Method to read atoms coordinate and velocities from DFTB+ output file geo_end.xyz
    Generated during MD simulation. If specified 'writeTrajectory=True', then generate
    a md.traj file in the folder.

    Assume that atoms match the trajectory:


            Args:
                filename: the path for the file (xyz trajectory) from the sampling method.
                          should corresponds to the geo_end.xyz or 'geom.out.xyz' file
                          of the DFTB trajectory
                writeTrajectory: bool
                          specify whether to write ase trajectory file.

    Return:
    md_sample_arr: An Array (pd.Series) of ase.Atoms generated during the trajectory
    md_Mulliken_arr: An Array (pd.Series)

    Comment:
    md_sample_arr is in units of angstrom, which is the fundamental unit in ase.
    Should work.


    Reference:
    Modified from ASE interface.
    """

    # Conversion of Units
    from ase.units import second
    # AA/ps -> ase units
    AngdivPs2ASE = 1.0/(1e-12*second)


    myfile = open(filename)
    lines  = myfile.readlines()
    # remove empty lines
    lines_ok = []
    for line in lines:
        if line.rstrip():
            lines_ok.append(line)

    # Loop: defined as
    #       Total number of iterations / time step
    natoms = int(lines_ok[0].split()[0])     # Because the first line, the first word is the nAtoms
    nloop = int(len(lines_ok)/(natoms+2))  # + 2 Including line with nAtoms and 'MD iter' line


    import numpy as np;
    import pandas as pd;

    # Initialize the position array
    md_sample_arr = {} #pd.Series([])
    md_mulliken_arr = np.empty((nloop, natoms))




    # Iterates each loop
    for loop_idx in np.arange(nloop):

        loop_lines = loop_to_lines_xyz(loop_idx, natoms, lines_ok);
        atom_arr_temp = []
        charge_arr_temp = []

        # Iterate through each line in the loop to get the atom coordinate and element
        for iline, line in enumerate(loop_lines):
            inp = line.split();

            # Read Atom elementElement
            ele = inp[0]

            # Read Coordinate
            x = float(inp[1])
            y = float(inp[2])
            z = float(inp[3])
            temp_posit = (x,y,z)

            # Create Atom
            atom_temp = Atom(symbol=ele, position=temp_posit)
            atom_arr_temp.append(atom_temp)

            # Read the charge
            atom_charge_temp = float(inp[4])
            md_mulliken_arr[loop_idx][iline] = atom_charge_temp




        atoms_temp = Atoms(atom_arr_temp)
        md_sample_arr[loop_idx] = atoms_temp

        # Write the trajectory if specified in input
    md_sample_arr = pd.Series(md_sample_arr)
    if writeTrajectory == True:
        from ase.io.trajectory import Trajectory
        from ase.io import write
        # First, initialize the trajectory file
        write('md.traj', md_sample_arr)

    return md_sample_arr, md_mulliken_arr



def read_dftb_md_energy(filename='../md_capture/md.out'):
    """
    Read the kinetic and potential energy from the md.out file for each loop.
    md.out file has the kinetic and potential energy for each 5 loops (This is the assumption.)

    Return:
    md_energy_arr: an numpy array of potential energy. with unit in Hartree
    md_kinetic_energy_arr, an numpy array of kinetic energy, with unit in eV

    Reference:
    Modified from ASE interface.
    """
    import numpy as np;

    myfile = open(filename)
    lines  = myfile.readlines()


    # md.out file does not have any empty lines to remove.
    # each loop will take 7 steps
    loop_nlines = 7
    nloops = int(len(lines)/ loop_nlines)

    print(nloops)
    md_energy_arr = np.empty(nloops)
    md_kinetic_energy_arr = np.empty(nloops)



    # For each loop, each the kinetic e and potential e.
    for loop_idx in np.arange(nloops):
        # Get Line number for the loop
        potent_line_idx  = loop_idx * loop_nlines + 1
        kinetic_line_idx = potent_line_idx + 1

        #import pdb; pdb.set_trace()

        # Read Energy
        potent_e = float(lines[potent_line_idx].split()[4])
        kinetic_e = float(lines[kinetic_line_idx].split()[5])
        md_energy_arr[loop_idx] = potent_e
        md_kinetic_energy_arr[loop_idx] = kinetic_e


    return md_energy_arr, md_kinetic_energy_arr





def read_scan_traj(filename='geo_end.xyz', writeTrajectory=False):
    """
    Method to read atoms coordinate and velocities from DFTB+ output file geo_end.xyz
    Generated during MD simulation. If specified 'writeTrajectory=True', then generate
    a md.traj file in the folder.

    Assume that atoms match the trajectory:


            Args:
                filename: the path for the file (xyz trajectory) from the sampling method.
                          should corresponds to the geo_end.xyz or 'geom.out.xyz' file
                          of the DFTB trajectory
                writeTrajectory: bool
                          specify whether to write ase trajectory file.

    Return:
    md_sample_arr: An Array (pd.Series) of ase.Atoms generated during the trajectory
    md_Mulliken_arr: An Array (pd.Series)

    Comment:
    md_sample_arr is in units of angstrom, which is the fundamental unit in ase.
    Should work.


    Reference:
    Modified from ASE interface.
    """

    # Conversion of Units
    from ase.units import second
    # AA/ps -> ase units
    AngdivPs2ASE = 1.0/(1e-12*second)


    myfile = open(filename)
    lines  = myfile.readlines()
    # remove empty lines
    lines_ok = []
    for line in lines:
        if line.rstrip():
            lines_ok.append(line)

    # Loop: defined as
    #       Total number of iterations / time step
    natoms = int(lines_ok[0].split()[0])     # Because the first line, the first word is the nAtoms
    nloop = int(len(lines_ok)/(natoms+2))  # + 2 Including line with nAtoms and 'MD iter' line


    import numpy as np;
    import pandas as pd;

    # Initialize the position array
    md_sample_arr = {} #pd.Series([])




    # Iterates each loop
    for loop_idx in np.arange(nloop):

        loop_lines = loop_to_lines_xyz(loop_idx, natoms, lines_ok);
        atom_arr_temp = []
        charge_arr_temp = []

        # Iterate through each line in the loop to get the atom coordinate and element
        for iline, line in enumerate(loop_lines):
            inp = line.split();

            # Read Atom elementElement
            ele = inp[0]

            # Read Coordinate
            x = float(inp[1])
            y = float(inp[2])
            z = float(inp[3])
            temp_posit = (x,y,z)

            # Create Atom
            atom_temp = Atom(symbol=ele, position=temp_posit)
            atom_arr_temp.append(atom_temp)



        atoms_temp = Atoms(atom_arr_temp)
        md_sample_arr[loop_idx] = atoms_temp

        # Write the trajectory if specified in input
    md_sample_arr = pd.Series(md_sample_arr)
    if writeTrajectory == True:
        from ase.io.trajectory import Trajectory
        from ase.io import write
        # First, initialize the trajectory file
        write('md.traj', md_sample_arr)

    return md_sample_arr 
