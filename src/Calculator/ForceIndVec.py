"""
############################################################

Module: ForceInd

Purpose:
Calculate the Force Necessary for the individual molecules.
Call Functions From SymmDerivInd.

Notice:
Some Functions are specific to Keras backend.
Theano Backend has a quite different implementation of the
gradient calculation, (because it can only calculates the
gradient for one scalar at one time rather than a matrix, while
tensorflow automatically supports the derivative of a matrix)


############################################################
"""

import numpy as np
import keras.backend as K
import src.Calculator.SymmDerivIndVecCython as SymmDerivIndVecCython
import src.Calculator.SymmDerivIndCython as SymmDerivInd

def get_subnet_deriv_tf(subnet, n_symm_func):
    """Generate a dE/dG function for a given elemental sub-neural network.

        Args:
            subnet: keras.model object for the sub-neural network.
            nb_symm: int, number of symmetry functions for the vector.

        Outputs:
            subnet_deriv([G_ind]), a function that takes the symmetry vector of shape (1, nb_symm) of the given atom
            and return the dE/dG matrix for the atom.

    """
    G_tensor_ind = K.placeholder(shape = (1, n_symm_func))
    E_tensor_ind = subnet(G_tensor_ind)
    #subnet_deriv = K.function([G_tensor_ind], K.gradients(E_tensor_ind, [G_tensor_ind]))
    subnet_deriv = K.function([G_tensor_ind], K.gradients(E_tensor_ind, G_tensor_ind))

    return subnet_deriv



def get_subnet_deriv_arr_tf(subnet_list, n_symm_func):
    """Generated the array for the dE/dG function.
    !! For Elemental Neural Network only

        Args:
            subnet_list: a list that includes all the sub-neural network with its element
            nb_symm: number of symmetry functions for the vector.

        Outputs:
            a pandas.Serires() array of subnet_deriv([G_ind]), a function that takes the symmetry
            vector of shape (1, nb_symm) of the given atom
            and return the dE/dG matrix for the atom.
    """
    subnet_deriv_arr = {}
    for element, subnet in subnet_list.items():
        subnet_deriv_arr[element] = get_subnet_deriv_tf(subnet, n_symm_func)

    return subnet_deriv_arr



def calculate_subnet_deriv_value(subnet_deriv_arr, element, Gfunc_ind):
    """

            Args:
                Gfunc_ind: Gfunc for individual atoms in the shape
                            (1, symm_func)

            Outputs:
                output: dE(subnet)/dG for the given atom (determined by Gfunc_ind)
                (1,symm_func)


    Assumes that Gfunc_ind has the same No. Symmetry function as the function for dE_dG_arr
    """
    subnet_deriv = subnet_deriv_arr[element]
    #output =  np.array(subnet_deriv([Gfunc_ind])).reshape(Gfunc_ind.shape).astype(np.float32)
    output =  np.array( subnet_deriv([Gfunc_ind]),  dtype=np.float32).reshape(Gfunc_ind.shape).astype(np.float64)
    return output



def calculate_subnet_deriv_value_list(subnet_deriv_arr, element, Gfunc_ind):
    """

            Args:
                Gfunc_ind: Gfunc for individual atoms in the shape
                            (1, symm_func)

            Outputs:
                output: dE(subnet)/dG for the given atom (determined by Gfunc_ind)
                (1,symm_func)


    Assumes that Gfunc_ind has the same No. Symmetry function as the function for dE_dG_arr
    """
    subnet_deriv = subnet_deriv_arr[element]
    #output =  np.array(subnet_deriv([Gfunc_ind])).reshape(Gfunc_ind.shape).astype(np.float32)
    #output =  np.array( subnet_deriv([Gfunc_ind]),  dtype=np.float32).reshape(Gfunc_ind.shape)
    output =  subnet_deriv([Gfunc_ind])
    return output




def calculate_analytical_force_ind_pile_compiled(nn_deriv_G_pile,
                                distance_arr,
                                at_ele_arr,
                                rad_arr, ang_arr,
                                ele_count, pair_count,
                                drad_precalc,
                                dRij_precalc,
                                dRik_precalc,
                                dRjk_precalc,
                                n_atoms, n_symm_func, n_ele, at_ref_idx,
                                n_rads, n_angs,
                                rad_count_each, ang_count_each):
    """Calculate the analytical force for an atom
    (Compiled Version)

            Args:
                nn_deriv_G_pile: piled up version of the matrix
                    nn_deriv_G_pile =  (1, n_atoms * n_symm_func) of [nn_deriv_G]
                at_idx_map: atom index map
                Gparam_dict:
                atoms: the molecule for the given atom
                atom: the atom itself


    Currently does not support feature array


    Pile Version: Get the pile of the array and then simplify the computation.


    TODO:
    Support the nn_deriv_arr before all the calculation.
    """
    # For the Force Calculation
    # It needs to count for all the atoms
    # For the G contribution.
    atom_ref_idx = at_ref_idx
    #atom_xyz = np.zeros((1,3))


    G_deriv_xyz_pile = SymmDerivIndVecCython.symm_deriv_ind_pile_vec(
                                distance_arr,
                                at_ele_arr,
                                rad_arr, ang_arr,
                                ele_count, pair_count,
                                drad_precalc,
                                dRij_precalc,
                                dRik_precalc,
                                dRjk_precalc,
                                n_atoms, n_symm_func, n_ele, atom_ref_idx,
                                n_rads, n_angs,
                                rad_count_each, ang_count_each)
    atom_xyz = np.matmul(nn_deriv_G_pile, G_deriv_xyz_pile)


    return atom_xyz.reshape(3)






def calculate_analytical_force_mol_compiled(subnet_deriv_arr, Gfunc_data,
                                    distance_arr, at_ele_map, at_ele_arr,
                                    rad_arr, ang_arr,
                                    ele_count, pair_count,
                                    drad_precalc,
                                    dRij_precalc,
                                    dRik_precalc,
                                    dRjk_precalc,
                                    n_atoms, n_symm_func, n_ele,
                                    n_rads, n_angs,
                                    rad_count_each, ang_count_each):
    """Calculate the analytical force for a molecule
    (Compiled Version)

            Args:
                subnet_deriv_arr:
                Gfunc_data:
                    Should have only one sample!
                    Gfunc_data[elemdent][at_idx] = (1, n_symm_func)
                atoms: ase.atoms object
                at_idx_map: atom index map
                at_ele_map: atom element map (element str)
                at_ele_arr: atom element array (element index in number )
                Gparam_dict:
                n_atoms: (not necessary but for performance)



    """

    atoms_force_xyz = np.zeros((n_atoms, 3), dtype=np.float64) # To contain force in xyz direction
    # dG_all_dict = SymmDerivInd.pre_calc_deriv_ind_arr_stacked(distance_arr, at_ele_arr, xyzArr,
    #                         neighbourlist_arr, neighbour_pair_arr,
    #                         count_Gparam, count_dict,
    #                         ang_list,
    #                         n_atoms, n_symm_func)

    import time
    pile_start_time = time.time()
    nn_deriv_G_pile = get_nn_deriv_G_pile_mol(subnet_deriv_arr, at_ele_map, Gfunc_data,
                                                n_atoms, n_symm_func)
    pile_end_time   = time.time()
   #print("Pile Time: ", pile_end_time - pile_start_time)



    force_start_time = time.time()
    for at_idx in np.arange(n_atoms):
        #subnet_deriv = subnet_deriv_arr[atom.symbol]

        atoms_force_xyz[at_idx,0:3] = calculate_analytical_force_ind_pile_compiled(
                                          nn_deriv_G_pile, distance_arr,
                                          at_ele_arr,
                                          rad_arr, ang_arr,
                                          ele_count, pair_count,
                                          drad_precalc,
                                          dRij_precalc,
                                          dRik_precalc,
                                          dRjk_precalc,
                                          n_atoms, n_symm_func, n_ele,  at_idx,
                                          n_rads, n_angs,
                                          rad_count_each, ang_count_each)

    force_end_time = time.time()
   #print("Force Time: ", force_end_time - force_start_time)

    atoms_force_xyz = atoms_force_xyz * -1
    return atoms_force_xyz








def get_nn_deriv_G_pile_mol(subnet_deriv_arr, at_ele_arr,Gfunc_data, n_atoms, n_symm_func):
    """
    (Compiled Version)

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
    nn_deriv_G_pile = np.zeros( (1, n_symm_func * n_atoms), dtype= np.float64)
    for at_idx in np.arange(n_atoms):
        at_symbol = at_ele_arr[at_idx]
        Gfunc_ind = Gfunc_data[at_idx, 0:1,:]
        idx_start = at_idx * n_symm_func
        idx_end = idx_start + n_symm_func
        # temp = calculate_subnet_deriv_value(subnet_deriv_arr,
        #                                 at_symbol, Gfunc_ind)
        # import pdb; pdb.set_trace()
        nn_deriv_G_pile[0, idx_start:idx_end] = calculate_subnet_deriv_value(subnet_deriv_arr,
                                        at_symbol, Gfunc_ind)
    return nn_deriv_G_pile
