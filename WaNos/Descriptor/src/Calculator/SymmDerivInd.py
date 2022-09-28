"""
#########

For the calculation of the symmetry function derivative

#########
"""

import numpy as np
import math # Faster than Numpy for scaler operations

import Calculation
# C-Extension for the derivative calculation
#import SymmDerivIndInterC
import SymmDerivIndCython as SymmDerivIndInterC
#TODO: Supply distances rather than calculating them together
#TODO: Replacing the Kernel
def symm_deriv_ind(at_idx_map, Gparam_dict, atoms, atom_idx, at_ref):
    """
    calculate symmetry functions derivative with respect to the reference atom
    (at_ref)


    For this function, I just want one of the coordinates. Not an array of everything

        Args:
            distances: distance values; pandas dataframe of shape (nb_samples, nb_distances)
            at_idx_map: a mapping between atom types and atom indexes; dictionary
            Gparam_dict: symmetry function parameters;
                            dictionary with 1st layer keys  = atom types,
                                2nd layer keys = symmetry function types (radial or angular)
                                values = 2D arrays of sym. function parameters of
                                shape (nb_sym_functions, nb_filter_parameters)
            atom: ase.atoms object that represent that particular atom
            atom_idx: index of the atom for the given neural network.
            atom_ref: index of the atom for the given

        Outputs:
           Gfunc_data: symmetry function values;
                        dictionary with 1st layer keys = atom types,
                            2nd layer keys = atom indexes,
                            values = 2D arrays with shape=(nb_samples, nb_sym_functions)

    Comment:
    # TODO:
    1. Replace the atoms with the distance.
    2. Use Parallel Programming for the individual calculations.

    Explaination:
    i_equal_ref, j_equal_ref, and k_equal_ref is for checking which atom is the
    reference atom the program wants to get the derivative.
    If none of them is the referenc atom, then just skip the calculation by
    the continue statement.

    """


    # This for loop goes through elements
    # Are together
    at_type = atoms[atom_idx].symbol
    Gparam_rad = Gparam_dict[at_type]['rad'] # Problem Here
    Gparam_ang = Gparam_dict[at_type]['ang'] # Problem Here

    rad_count = sum([Gparam_rad[t].shape[0] for t in Gparam_rad.keys()])
    ang_count = sum([Gparam_ang[t].shape[0] for t in Gparam_ang.keys()])

    xyz_ref = atoms[at_ref].position
    xyz_i = atoms[atom_idx].position

    i_equal_ref = (atom_idx == at_ref)
    i_not_equal_ref = not i_equal_ref


    ## This for loop goes through all the atoms (belong to the same element)
    Gfunc_data = np.zeros(((rad_count + ang_count), 3))

    G_temp_count = 0


    # radial components
    # Loop through different elements
    for at2_type in Gparam_rad.keys():
        comp_count =  Gparam_rad[at2_type].shape[0]
        G_temp_component = np.zeros((comp_count, 3)) # Will store the array of the [dx, dy, dz]

        # One count is for one set of parameter values
        for count, values in enumerate(Gparam_rad[at2_type]):
            # Summing up the contribution from different distance values.
            for at2 in at_idx_map[at2_type][at_idx_map[at2_type]!=atom_idx]:


                # Check the Derivative calculation Before calculation
                j_equal_ref = (at2 == at_ref)
                j_not_equal_ref = not j_equal_ref


                if i_equal_ref:
                    which_is_ref = 'i'
                elif j_equal_ref:
                    which_is_ref = 'j'
                else:
                    which_is_ref = 'none'
                    continue

                # Test For Loop

                xyz_j = atoms[at2].position
                R12 = distance_xyz(xyz_i, xyz_j)
                # values[0] = Rs, values[1] = eta (integer, not array)
                # Then Calculate the radial symmetric function -> value of G.
                rad_temp = ddR_radial_filter_ind(values[0], values[1], R12, xyz_i, xyz_j, which_is_ref)
                G_temp_component[count, 0:3] += rad_temp
                if np.any(rad_temp > 3000):
                    import pdb; pdb.set_trace()


        Gfunc_data[G_temp_count:G_temp_count+comp_count, 0:3] = G_temp_component
        G_temp_count += comp_count

    # ======================
    # angular components
    for atAatB_type in Gparam_ang.keys():
        comp_count = Gparam_ang[atAatB_type].shape[0]
        G_temp_component = np.zeros((comp_count,3))

        # This for loop goes through all 'HH', 'HO' combo?
        for count, values in enumerate(Gparam_ang[atAatB_type]):
            atA_list = at_idx_map[atAatB_type[0]][at_idx_map[atAatB_type[0]]!=atom_idx]
            for atA in atA_list:
                xyz_j = atoms[atA].position

                j_equal_ref = atA == at_ref


                R1A = distance_xyz(xyz_i, xyz_j)


                if atAatB_type[0] == atAatB_type[1]:
                    atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=atom_idx) & (at_idx_map[atAatB_type[1]]>atA)]
                else:
                    atB_list = at_idx_map[atAatB_type[1]][(at_idx_map[atAatB_type[1]]!=atom_idx)]

                for atB in atB_list:
                    #R1B = 1
                    #RAB = 1

                    k_equal_ref = (atB == at_ref)
                    if i_equal_ref:
                        which_is_ref = 'i'
                    elif j_equal_ref:
                        which_is_ref = 'j'
                    elif k_equal_ref:
                        which_is_ref = 'k'
                    else:
                        continue

                    xyz_k = atoms[atB].position

                    R1B = distance_xyz(xyz_i, xyz_k)
                    RAB = distance_xyz(xyz_j, xyz_k)

                    ang_temp = ddR_angular_filter_ind_sep(R1A, R1B, RAB, values[0], values[1], values[2], xyz_i, xyz_j, xyz_k, which_is_ref)

                    #ang_temp = ddR_angular_filter_ind(R1A, R1B, RAB, values[0], values[1], values[2], xyz_i, xyz_j, xyz_k, which_is_ref)
                    if np.any(ang_temp > 3000):
                        import pdb; pdb.set_trace()
                        ddR_angular_filter_ind(R1A, R1B, RAB, values[0], values[1], values[2], xyz_i, xyz_j, xyz_k, xyz_ref)

                    G_temp_component[ count, 0:3] += ang_temp


        Gfunc_data[G_temp_count:G_temp_count+comp_count, 0:3] = G_temp_component
        G_temp_count += comp_count
    return Gfunc_data






def symm_deriv_ind_arr(distance_arr, at_ele_arr, xyzArr,
                        neighbourlist_arr, neighbourpair_arr,
                        count_Gparam, count_dict,
                        ang_list,
                        n_atoms, n_symm_func, at_idx, at_ref):

    Gfunc_deriv_data = np.zeros( shape = (n_symm_func, 3))
    xyz_i = xyzArr[at_idx]


    i_equal_ref = (at_idx == at_ref)
    i_not_equal_ref = not i_equal_ref

    for at_idx_j in neighbourlist_arr[at_idx]:
        Rij = Calculation.get_distance(at_idx, at_idx_j, n_atoms, distance_arr)
        at_j_ele = at_ele_arr[at_idx_j]
        xyz_j = xyzArr[at_idx_j]


        # Get the corresponding counts in the vector
        for symm_count in count_dict[at_j_ele]:


            j_equal_ref = (at_idx_j == at_ref)


            if i_equal_ref:
                which_is_ref = 'i'
            elif j_equal_ref:
                which_is_ref = 'j'
            else:
                which_is_ref = 'none'
                continue

            rad_params = count_Gparam[symm_count]
            Rs = rad_params[0]
            eta = rad_params[1]

            Gfunc_deriv_data[symm_count, 0:3] += SymmDerivIndInterC.ddR_radial_filter_ind(Rs, eta, Rij,  xyz_i, xyz_j, which_is_ref)


    # This for loop goes through all neighbours pairs of the atoms
    # For the angular components
    # Go through all the neighbour pairs
    for neighbour_pair in neighbourpair_arr[at_idx]:
        at_idx_j = neighbour_pair[0]
        at_idx_k = neighbour_pair[1]



        j_equal_ref = (at_idx_j == at_ref)
        k_equal_ref = (at_idx_k == at_ref)
        if i_equal_ref:
            which_is_ref = 'i'
        elif j_equal_ref:
            which_is_ref = 'j'
        elif k_equal_ref:
            which_is_ref = 'k'
        else:
            continue

        xyz_j = xyzArr[at_idx_j]
        xyz_k = xyzArr[at_idx_k]


        Rij = Calculation.get_distance(at_idx, at_idx_j, n_atoms, distance_arr)
        Rik = Calculation.get_distance(at_idx, at_idx_k, n_atoms, distance_arr)
        Rjk = Calculation.get_distance(at_idx_j, at_idx_k, n_atoms, distance_arr)

        at_i_ele = at_ele_arr[at_idx]
        at_j_ele = at_ele_arr[at_idx_j]
        at_k_ele = at_ele_arr[at_idx_k]

        pair_jk = Calculation.get_pair(at_j_ele, at_k_ele, ang_list)


        # Get the corresponds counts in the vector
        for symm_count in count_dict[pair_jk]:
            ang_params = count_Gparam[symm_count]
            eta = ang_params[0]
            zeta = ang_params[1]
            lambd = ang_params[2]

            Gfunc_deriv_data[symm_count, 0:3] += ddR_angular_filter_ind_sep(Rij, Rik, Rjk, eta, zeta, lambd, xyz_i, xyz_j, xyz_k, which_is_ref)

    return Gfunc_deriv_data







def ddR_radial_filter_ind(Rs, eta, Rij, xyz_i, xyz_j, which_is_ref):
    """(Python Implementation) derivative of radial filter for symmetry functions with respect to dR
    ! We want it to be  dGi/dR * -1/R, so that it will just work for
    dGi/dx = result * x


            Args:
                Rs, eta: radial symmetry function parameters; float
                Rij: distance values between two given atoms i and j;
                        1D numpy array of length nb_samples
                xyz_i: the xyz coordinate of the atom,
                        2D numpy array of length nb_samples.
                xyz_j: the same as xyz_i
                which_is_ref: either 'i' or 'j'

            Outputs
                dG/dref: [dG/dx, dG/dy, dG/dz] that represents the derivative of G with
                        respect to the reference atom's x, y, z coordinates.
    """


    ### Brutal Force Implementation
    if which_is_ref != 'i' and which_is_ref != 'j':
        import pdb; pdb.set_trace()
        print("which_is_ref should be i or j")


    # if distance_xyz(xyz_i, xyz_j) != Rij:
    #     print("Error in distance reading")



    xi = xyz_i[0]
    yi = xyz_i[1]
    zi = xyz_i[2]

    xj = xyz_j[0]
    yj = xyz_j[1]
    zj = xyz_j[2]

    ddRij_G = -2*math.exp(-eta * (Rij-Rs)**2)*eta*(Rij-Rs)

    ddxi_dRij = (xi-xj) / Rij
    ddxi_dG = ddxi_dRij * ddRij_G

    ddyi_dRij = (yi-yj) / Rij
    ddyi_dG = ddyi_dRij * ddRij_G

    ddzi_dRij = (zi-zj) / Rij
    ddzi_dG = ddzi_dRij * ddRij_G

    if which_is_ref == 'j':
        # In case of dG/dxj, there is an extra negative sign.
        ddxi_dG = -ddxi_dG
        ddyi_dG = -ddyi_dG
        ddzi_dG = -ddzi_dG

    return np.array([ddxi_dG, ddyi_dG, ddzi_dG])


# Angular Filter Function is the Angular Symmetry Function
# For a given set of (eta, zeta, lambd), caclaulte
# The augular component of G value for all the neighbour atoms

# To change the filter function, modify it here.
def ddR_angular_filter_ind(Rij, Rik, Rjk, eta, zeta, lambd, xyz_i, xyz_j, xyz_k, which_is_ref):
    """(Python Implementation) derivative of angular symmetry functions in C-Extension

            Args:
                Rij, Rik, Rjk: Distance Values (float)
                eta, zeta, lambd: parameters
                xyz_i, xyz_j, xyz_k: np array of size (3,) that represents 3 xyz values.
                which_is_ref: 'i', 'j' or 'k' that represents which one is the reference atom.

            Outputs:
                dG/dref: [dG/dx, dG/dy, dG/dz] that represents the derivative of G with
                        respect to the reference atom's x, y, z coordinates.

    Explaination:
    It could be possible that atom_i, atom_j, atom_z are the reference atom.
    In case none of them are, then dG/dref = [0,0,0]

    Because this is all numerical calculation, C-Extension is used for the calculation.

    Comments:
    Untested.
    """
    if PRINT_MODE == True:
        print(Rij, Rik, Rjk, eta, zeta, lambd)


    if which_is_ref != 'i' and which_is_ref != 'j' and which_is_ref != 'k':
        # If none of atom ijk is the reference atom, then report error
        import pdb; pdb.set_trace()
        print("ddR_angular_filter_ind: which_is_ref should be i,j,k")




    cos_angle = (Rij**2 + Rik**2 - Rjk**2)/(2.0 * Rij * Rik)
    rad_filter = math.exp(-eta*(Rij + Rik + Rjk)**2)
    G_ang_ijk = 2**(1.0-zeta) * (1.0 + lambd * cos_angle)**zeta * rad_filter

    xi = xyz_i[0]
    yi = xyz_i[1]
    zi = xyz_i[2]

    xj = xyz_j[0]
    yj = xyz_j[1]
    zj = xyz_j[2]

    xk = xyz_k[0]
    yk = xyz_k[1]
    zk = xyz_k[2]

    # if distance_xyz(xyz_i, xyz_j) != Rij:
    #     print("Error")
    # if distance_xyz(xyz_j, xyz_k) != Rjk:
    #     print("Error")
    # if distance_xyz(xyz_i, xyz_k) != Rik:
    #     print("Error")


    ddRij_G = -2.0**(2.0-zeta) * math.exp(-eta*(Rij+Rik+Rjk)**2) * eta * (Rij + Rik + Rjk) \
              * (1.0 + lambd * cos_angle)**zeta \
              + 2.0**(1.0-zeta) * math.exp(-eta*(Rij+Rik+Rjk)**2) * (lambd/Rik - lambd * cos_angle / Rij)\
              * (1.0 + lambd * cos_angle)**(-1.0+zeta) * zeta

    ddRik_G = -2.0**(2.0-zeta) * math.exp(-eta*(Rij+Rik+Rjk)**2) * eta * (Rij + Rik + Rjk) \
              * (1.0 + lambd * cos_angle)**zeta \
              + 2.0**(1.0-zeta) * math.exp(-eta*(Rij+Rik+Rjk)**2) * (lambd/Rij - lambd * cos_angle / Rik)\
              * (1.0 + lambd * cos_angle)**(-1.0+zeta) * zeta

    ddRjk_G = -2.0**(2.0-zeta) * math.exp(-eta*(Rij+Rik+Rjk)**2) * eta * (Rij + Rik + Rjk) \
              * (1.0 + lambd * cos_angle)**zeta \
              - (2**(1.0-zeta)* math.exp(-eta*(Rij+Rik+Rjk)**2) * lambd * Rjk \
              * (1.0 + lambd * cos_angle)**(-1.0+zeta) * zeta) / (Rij*Rik)




    # Calculate all the derivative

    ddxi_Rij = (xi - xj) / Rij
    ddxj_Rij = -ddxi_Rij

    ddxi_Rik = (xi - xk) / Rik
    ddxk_Rik = -ddxi_Rik

    ddxj_Rjk = (xj - xk) / Rjk
    ddxk_Rjk = - ddxj_Rjk


    ddyi_Rij = (yi - yj) / Rij
    ddyj_Rij = -ddyi_Rij

    ddyi_Rik = (yi - yk) / Rik
    ddyk_Rik = -ddyi_Rik

    ddyj_Rjk = (yj - yk) / Rjk
    ddyk_Rjk = -ddyj_Rjk


    ddzi_Rij = (zi - zj) / Rij
    ddzj_Rij = -ddzi_Rij

    ddzi_Rik = (zi - zk) / Rik
    ddzk_Rik = -ddzi_Rik

    ddzj_Rjk = (zj - zk) / Rjk
    ddzk_Rjk = -ddzj_Rjk



    if which_is_ref == 'i':
        ddx_G = ddRij_G * ddxi_Rij + ddRik_G * ddxi_Rik
        ddy_G = ddRij_G * ddyi_Rij + ddRik_G * ddyi_Rik
        ddz_G = ddRij_G * ddzi_Rij + ddRik_G * ddzi_Rik
    elif which_is_ref == 'j':
        # For here, ddxi actually converts to ddxj
        ddx_G = ddRij_G * ddxj_Rij + ddRjk_G * ddxj_Rjk
        ddy_G = ddRij_G * ddyj_Rij + ddRjk_G * ddyj_Rjk
        ddz_G = ddRij_G * ddzj_Rij + ddRjk_G * ddzj_Rjk
    elif which_is_ref == 'k':
        # For here, ddxi actually converts to ddxk
        ddx_G = ddRjk_G * ddxk_Rjk + ddRik_G * ddxk_Rik
        ddy_G = ddRjk_G * ddyk_Rjk + ddRik_G * ddyk_Rik
        ddz_G = ddRjk_G * ddzk_Rjk + ddRik_G * ddzk_Rik
    else:
        import pdb; pdb.set_trace()

    if PRINT_MODE == True:
        print("Python Result")
        print([ddx_G, ddy_G, ddz_G])
        #import pdb; pdb.set_trace()
    return np.array([ddx_G, ddy_G, ddz_G])





def ddR_angular_filter_ind_sep(Rij, Rik, Rjk, eta, zeta, lambd, xyz_i, xyz_j, xyz_k, which_is_ref):
    """(Wrapper) derivative of angular symmetry functions in C-Extension

            Args:
                Rij, Rik, Rjk: Distance Values (float)
                eta, zeta, lambd: parameters
                xyz_i, xyz_j, xyz_k: np array of size (3,) that represents 3 xyz values.
                which_is_ref: 'i', 'j' or 'k' that represents which one is the reference atom.

            Outputs:
                dG/dref: [dG/dx, dG/dy, dG/dz] that represents the derivative of G with
                        respect to the reference atom's x, y, z coordinates.

    Explaination:
    It could be possible that atom_i, atom_j, atom_z are the reference atom.
    In case none of them are, then dG/dref = [0,0,0]

    Because this is all numerical calculation, C-Extension is used for the calculation.

    """


    if which_is_ref == 'i':
        return SymmDerivIndInterC.ddR_angular_filter_ind_sep(Rij, Rik, Rjk, eta, zeta, lambd,
                                        xyz_i, xyz_j, xyz_k, 'i')
    elif which_is_ref == 'j':
        return SymmDerivIndInterC.ddR_angular_filter_ind_sep(Rij, Rik, Rjk, eta, zeta, lambd,
                                        xyz_i, xyz_j, xyz_k, 'j')
    elif which_is_ref == 'k':
        return SymmDerivIndInterC.ddR_angular_filter_ind_sep(Rij, Rik, Rjk, eta, zeta, lambd,
                                        xyz_i, xyz_j, xyz_k, 'k')
    else:
        print("ddR_angular_filter_ind_sep: which_is_ref should not be none")

    return np.array([0, 0, 0])











#
# def distance_ind(atom_1, atom_2):
#     """Calculate the distance between two atoms.
#         Args:
#             atom_1: ase.atom object
#             atom_2: ase.atom object
#
#         Return:
#             Rij: distance for the atoms in angstroms
#     """
#
#     r = np.sqrt(np.sum((atom_1.position - atom_2.position)**2, axis = 0))
#     return r

"""
##########

Util Functions to
replace slow numpy operations

##########
"""
def distance_xyz(xyz_i, xyz_j):
    #return np.sqrt(np.sum((xyz_i - xyz_j)**2, axis = 0))
    return SymmDerivIndInterC.distance_xyz(xyz_i, xyz_j)

"""
It is suspected that the xyz_equal for the shape (3)
xyz array is much slower

since numpy is optimized for large scale operations

"""

def xyz_equal(xyz_i, xyz_j):
    """Check 2 xyz_arr are equal


    """
    if xyz_i[0] == xyz_j[0] and xyz_i[1] == xyz_j[1] and xyz_i[2] == xyz_j[2]:
        return True
    else:
        return False
