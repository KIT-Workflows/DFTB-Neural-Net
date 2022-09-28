"""
This Module is for the calculation of numerical force.
"""
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
import pandas as pd


import src.Calculator.src_nogrd as src_nogrd
import src.Calculator.SymmFuncIndPython as SymmFuncInd




def shake_atom_posit(atom, xyz_str, h):
    """Change the position of the atom for a given length.

        Args:
            atom: ase.atom object
            xyz_str: 'x', 'y', 'z' to specify which coordinate to change
            h: the small length used to change the atomic coordinate. (in Angstrom)

        Outputs:
            shaked_position: [x', y', z'] = [+h]

    Comments:
    This is a pure function. Will not modify the position of the original atom.
    """

    # Convert xyz_str to 1,2,3 index
    index = 0
    if xyz_str == 'x':
        index = 0
    elif xyz_str == 'y':
        index = 1
    elif xyz_str == 'z':
        index = 2
    else:
        import pdb; pdb.set_trace()
        raise ValueError("shake_atom_position: Please give a valid xyz_str")

    new_position = np.copy(atom.position)
    new_position[index] += h

    return new_position





def shake_mol_atom_posit(atoms, atom_idx, xyz_str, h):
    """Change the position of one of given atom for a given length for the molecule

    Will return 3 configurations atoms with shaked x, or y or z position.

        Args:
            atoms: ase.atoms object that represents the configurations
            atom_idx: index of the given atom
            xyz_str: 'x', 'y' or 'z'
            h: the small length used to change the atomic coordinate. (in Angstrom)

        Outputs:
            shaked_atoms: ase.atoms object that represents the atom with shaked position

    Comments:
    This is a pure function. Will Not Change the original molecule (atoms)
    """
    shaked_atoms = atoms.copy()

    atom = shaked_atoms[atom_idx]
    atom.position = shake_atom_posit(atom, xyz_str, h)
    return shaked_atoms





def force_calc_3p(e_minus, e_plus, h):
    """Calculate the Derivative of the force based on 3 point formula.

    Calculate the Force (f) at point x based on numerical differentiation
    For a step of h
    dy/dx = (y(x+h) - y(x-h)) / 2*h

        Args:
            e_minus: float, force at -dx away from the point   (y(x+h))
            e_plus: float, force at +dx away from the point  (y(x-h))
            h: distance for the points (in Angstrom)

        Outputs:
            dy/dx: float for the derivative


    """

    return (e_plus - e_minus) / (2*h)





def force_num_mol(atoms, calc, h):
    """Will Set the calculator for the atoms and then return
    the numerical derivative for all the forces.

        Args:


        Outputs:
            force_df: at index = derivative
                force_df[atom_idx][xyz_str] = numerical force
    Comments:
    For Testing Purpose Onlyself.
    Cannot be directly put into the calculator class since a calculator
    cannot contain an instance of itself.
    """
    n_atoms = len(atoms)
    atom_idx_arr = np.arange(n_atoms)
    xyz_arr = np.array(['x', 'y', 'z'])

    xyz_str_dict = {'x':0,  'y': 1,  'z': 2} # To Translate xyz_str to number


    force_arr = np.zeros((n_atoms, 3))

    for atom_idx in atom_idx_arr:
        force_arr_atom = []
        for xyz_str in xyz_arr:
            shaked_atoms_plus = shake_mol_atom_posit(atoms, atom_idx, xyz_str, h)
            shaked_atoms_plus.set_calculator(calc)

            e_plus = shaked_atoms_plus.get_potential_energy()

            shaked_atoms_minus = shake_mol_atom_posit(atoms, atom_idx, xyz_str, -h)
            shaked_atoms_minus.set_calculator(calc)
            e_minus = shaked_atoms_minus.get_potential_energy()

            xyz_num = xyz_str_dict[xyz_str]
            force_arr[atom_idx][xyz_num]= -force_calc_3p(e_minus, e_plus, h)


    return force_arr 



def deriv_num_mol(atoms, h, at_ref_idx, n_symm_func, xyz_str, Gparam_dict):
    """Calculate the Numerical Derivative dG/dx with G_deriv_xyz_pile for a
    given atom.


    """
    n_atoms = len(atoms)

    at_idx_map = src_nogrd.at_idx_map_generator(atoms)

    satoms_plus = shake_mol_atom_posit(atoms, at_ref_idx , xyz_str, h)
    Gfunc_data_plus = SymmFuncInd.symm_func_mol(satoms_plus, at_idx_map, Gparam_dict)
    Gfunc_pile_plus = get_Gfunc_pile_mol(Gfunc_data_plus, at_idx_map, n_atoms, n_symm_func)

    satoms_minus = shake_mol_atom_posit(atoms, at_ref_idx, xyz_str, -h)
    Gfunc_data_minus = SymmFuncInd.symm_func_mol(satoms_minus, at_idx_map, Gparam_dict)
    Gfunc_pile_minus = get_Gfunc_pile_mol(Gfunc_data_minus, at_idx_map, n_atoms, n_symm_func)

    Gfunc_deriv = (Gfunc_pile_plus - Gfunc_pile_minus) / (2 * h)

    return Gfunc_deriv



def get_Gfunc_pile_mol(Gfunc_data, at_idx_map,  n_atoms, n_symm_func):
    """Gfunc_pile is similar to the structure of nn_deriv_G, but use Gfunc

            Args:
                subnet_deriv_arr:
                at_ele_arr: atom element map that returns the element for given
                            atom index
                Gfunc_data: (New Type)
                atoms:

            Output:
                nn_deriv_G_mat: The dE/dG matrix for the given molecule
                nn_deriv_G_mat = (n_atoms, n_symm_func)
                nn_deriv_G_mat[at_idx] = [nn_deriv_G]
                nn_deriv_G_mat (matmul) G_deriv_xyz_mol = Force_xyz_mol

    It was eventually found that nn_deriv_G (or d dE/dG) is fixed
    for each atom. Thus, it is helpful to organize them into an arrayself.

    Doing so also helps to get all the operations on GPU
    doing before putting into sub process.
    """
    # Get The Number of symmetry functions
    Gfunc_pile = np.empty( (1, n_symm_func * n_atoms), dtype= np.float32)
    # for at_idx in np.arange(n_atoms):
    #     import pdb; pdb.set_trace()
    #     Gfunc_ind = Gfunc_data[at_idx, 0:1,:]
    #     idx_start = at_idx * n_symm_func
    #     idx_end = idx_start + n_symm_func
    #
    #     # temp = calculate_subnet_deriv_value(subnet_deriv_arr,
    #     #                                 at_symbol, Gfunc_ind)
    #     Gfunc_pile[0, idx_start:idx_end] = Gfunc_ind


    for element in at_idx_map.keys():

        for at_idx in at_idx_map[element]:
            Gfunc_ind = Gfunc_data[element][at_idx]
            idx_start = at_idx * n_symm_func
            idx_end = idx_start + n_symm_func
            Gfunc_pile[0, idx_start:idx_end] = Gfunc_ind[0,:]

    return Gfunc_pile

