"""

A Module designed for the vectorization performance of symmetry function
calculations (In the Pre-Calc Section)


Currently In the Prototype Stage

"""


import numpy as np


def k_delta_rev_place(arr_i, arr_j):
    """ (np.place() implementation )
    Return a new function that has the following property

    for each array element:
        if (arr_i == arr_j):
            retrun 0;
        if (arr_i != arr_j):
            return 1;

    (Once we have the reversed kronecker delta function,
    it becomes easier to do the calculation.)

            Args:
                arr_i, arr_j: array for comparison

            Outputs:
                arr_out:
    """

    arr_out = arr_i - arr_j
    arr_out = arr_out.place(arr_out!=0, [1])
    return arr_out

def k_delta_rev_piece(arr_i, arr_j):
    arr_out = np.piecewise(arr_i, [arr_i = arr_j, arr_i != arr_j], [0,1])
    return arr_out

def k_delta_rev_where(arr_i, arr_j):
    arr_out = arr_i - arr_j
    arr_out = np.where(arr_out == 0, 0, 1)
    return arr_out


def symm_func_mol_ang_pair():
    """A function designed to build prototype for other functions

    Still NO vectorized operations



    """

    Gfunc_data = np.zeros(shape=(n_atoms, 1, n_symm_func)) #

    for ang_ijk in angles:


        for symm_count in count_dict(ang_ijk):
            ang_params = count_Gparam[symm_count]
            eta = ang_params[0]
            zeta = ang_params[1]
            lambd = ang_params[2]

            Gfunc_data[at_idx_i, 0, symm_count] += angular_filter_ind(Rij, Rik, Rjk, eta, zeta, lambd)




def symm_func_mol_vec():
    """ Vectorized Version of the Symmetry Function Calculation


    Prototype Number:
    n_pairs: number of jk pairs in the tuple (i, j, k)
    n_angs:  total number of angles
             n_angs = n_pairs * n_atoms

    """
    (Rij, Rjk, Rik).shape = (n_angs, )  # 1D Tensor of size n_angs


    Gfunc_data = np.zeros(shape=(n_atoms, 1, n_symm_func))


    ang_vec_data = np.zeros( shape=(n_angs, ang_counts_tot))

    #ang_precalc =

    # ang_count: start from 0
    # Different from symm_count
    # symm_count: Corresponds to certain element
    for ang_count in range(0, ang_counts_tot+1, 1):
        ang_params = count_Gparam[symm_count]
        eta = ang_params[0]
        zeta = ang_params[1]
        lambd = ang_params[2]

        # Do the Massive Calculation at once.
        ang_vec[:, ang_count] = angular_filter_vec(Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd) # ang_vec has the same size as Rij_vec



    allocate(Gfunc_data, ang_vec, ang_count) # allocate ang_vec to the given atom .


    for at_i in range(0, n_atoms, 1):

        Gfunc_data[at_i, 0, pair_counts] = ang_vec]





# def symm_func_mol_vec( ddRij_G_vec, ddRik_G_vec, ddRjk_G_vec, cos_angle_vec, rad_filter_for_ang_vec, ang_filter_vec):
#
# # (np.ndarray[NPdouble_t, ndim=2] distance_arr, np.ndarray[NPint_t, ndim=1] at_ele_arr,
# #                         np.ndarray[NPint_t, ndim=2] neighbourlist_arr, np.ndarray[NPint_t, ndim=1] neighbourlist_count,
# #                         np.ndarray[NPint_t, ndim=3] neighbourpair_arr, np.ndarray[NPint_t, ndim=1] neighbourpair_count,
# #                         np.ndarray[NPdouble_t, ndim=2] count_Gparam,
# #                         np.ndarray[NPint_t, ndim=2] ele_count , np.ndarray[NPint_t , ndim=2] pair_count,
# #                         long n_atoms, long n_symm_func, long n_ang_count, long n_ele):
# """
# Refactor the entire symmetry function calculation into
# arr operations to speed up.
#
# (count refers to the symmetry function vector count )
#
#         Args:
#
#             neighbourlist_count:
#                         neighbourlist_count[at_idx] = No. Neighbours in the
#                                   neighbour list for given atom
#             neighbourpair_count:
#                         neighbourpair_count[at_idx] = No. Neighbour pairs
#                                   in the neighbour pair for given atom
#
#
#             count_Gparam: a np.array that has the structure
#                         count_Gparam[count]  = Gparam_list (specified for the given count)
#
#                         count_Gparam = (count, max No. Params)
#
#             n_atoms: number of atoms
#             n_symm_func: number of symmetry functions
#             n_ele:   number of elements
#
#
#         Outputs:
#
#            Gfunc_data: symmetry function values;
#                         dictionary with 1st layer keys = atom types,
#                             2nd layer keys = atom indexes,
#                             values = 2D arrays with shape=(1, n_symm_func)
#
#                         Gfunc_data[element][at_idx] = (1, n_symm_func)
#
#            (New) Gfunc_data[at_idx] = (1, n_symm_func)
#            New Data structure should be adopted when the input can be organzied.
#            It is much more organized since this is an numpy array.
#            Even though The New Data structure looks like it only supports
#            fixed number of atoms, since the data will be converted into input,
#            it is still able to work with multiple different arrays.
#
#            ang_precalc: get the dG(ang)/ dRij, dRik, dRjk for all the
#                         angular symmetry function
#
# Comments:
# Working In Progress.
#
# Attention:
# Neighbour list will change with different configurations.
# A simpler way might be to loop through every of the atom, and then
# just check for the distances .
#
#
# """
# # cdef NPint_t at_idx_i, at_idx_j, at_idx_k
# # cdef NPdouble_t Rij, Rjk, Rik
# # cdef NPdouble_t Rs, eta, zeta, lambd
# #
# # cdef long symm_count, symm_count_0
# # cdef long symm_count_start, symm_count_end
# # cdef long n_pairs = <long> n_atoms * (n_atoms - 1) / 2
# # cdef long angular_idx
# #
# #
# # cdef long ij_idx, ik_idx, jk_idx
#
#
# #cdef np.ndarray[NPdouble_t, ndim=3]
# Gfunc_data = np.zeros([n_atoms, 1, n_symm_func], NPdouble) # New Gfunc structure
#
#
#
# #dG_all_dict = {}
# cdef NPdouble_t ddRij_G, ddRik_G, ddRjk_G
#
# cdef NPdouble_t ddx_Rij, ddx_Rjk, ddx_Rik
# cdef NPdouble_t ddy_Rij, ddy_Rjk, ddy_Rik
# cdef NPdouble_t ddz_Rij, ddz_Rjk, ddz_Rik
# cdef DIST dist_output
#
# cdef NPint_t ele_idx_i, ele_idx_j, ele_idx_k
# cdef long pair_jk_idx
#
# cdef NPint_t nlist_count
# cdef long nlist_count_i
# cdef NPint_t npair_count
# cdef long npair_count_i
#
#
# cdef NPdouble_t cos_angle, rad_filter_for_ang
#
#
# # This for loop goes through all atoms
# for at_idx_i in prange(0, n_atoms, 1, nogil=True):
#
#     # This for loop goes through all neighbours of the atom
#     # For the Radial Components
#
#     # Prepare for For loop
#     nlist_count_i = neighbourlist_count[at_idx_i]
#     # Go Through Neighbours
#     for nlist_count in range(0, nlist_count_i, 1):
#     #for at_idx_j in neighbourlist_arr[at_idx_i]:
#         at_idx_j = neighbourlist_arr[at_idx_i, nlist_count]
#         ij_idx = distance_xyz_index(at_idx_i, at_idx_j, n_atoms)
#         Rij = distance_arr[ij_idx,0]
#
#         ele_idx_j = at_ele_arr[at_idx_j]
#
#
#         symm_count_start = ele_count[ele_idx_j, 0]
#         symm_count_end   = ele_count[ele_idx_j, 1]
#         for symm_count in range(symm_count_start, symm_count_end,1):
#             Rs = count_Gparam[symm_count, 0]                  #rad_params[0]
#             eta = count_Gparam[symm_count, 1]                 #rad_params[1]
#             Gfunc_data[at_idx_i,0, symm_count] += radial_filter_ind(Rs, eta, Rij)
#
#
#     # This for loop goes through all neighbours pairs of the atoms
#     # For the angular components
#     # Go through all the neighbour pairs
#
#     # Prepare for the For-loop (with indices)
#     npair_count_i = neighbourpair_count[at_idx_i]
#
#     for npair_count in range(0, npair_count_i, 1):
#     #for neighbour_pair in neighbourpair_arr[at_idx_i]:
#
#         # Get the indices
#         #at_idx_j = neighbour_pair[0]
#         #at_idx_k = neighbour_pair[1]
#         at_idx_j = neighbourpair_arr[at_idx_i, npair_count, 0]
#         at_idx_k = neighbourpair_arr[at_idx_i, npair_count, 1]
#
#
#         ij_idx = distance_xyz_index(at_idx_i, at_idx_j, n_atoms)
#         ik_idx = distance_xyz_index(at_idx_i, at_idx_k, n_atoms)
#         jk_idx = distance_xyz_index(at_idx_j, at_idx_k, n_atoms)
#
#
#         # Get the Distance and Derivative  (Store for next calculation)
#
#         Rij = distance_arr[ij_idx,0]
#         Rik = distance_arr[ik_idx,0]
#         Rjk = distance_arr[jk_idx,0]
#
#         ddx_Rij = distance_arr[ij_idx,1]
#         ddy_Rij = distance_arr[ij_idx,2]
#         ddz_Rij = distance_arr[ij_idx,3]
#
#         ddx_Rjk = distance_arr[jk_idx,1]
#         ddy_Rjk = distance_arr[jk_idx,2]
#         ddz_Rjk = distance_arr[jk_idx,3]
#
#         ddx_Rik = distance_arr[ik_idx,1]
#         ddy_Rik = distance_arr[ik_idx,2]
#         ddz_Rik = distance_arr[ik_idx,3]
#
#
#         # Get the pair, for next for loop over the Element Pair (eg. 'OH')
#         ele_idx_i = at_ele_arr[at_idx_i]
#         ele_idx_j = at_ele_arr[at_idx_j]
#         ele_idx_k = at_ele_arr[at_idx_k]
#
#         pair_jk_idx = get_pair_idx(ele_idx_j, ele_idx_k,  n_ele)
#
#
#         # Get the corresponds counts in the vector
#         symm_count_start = pair_count[pair_jk_idx,0]
#         symm_count_end   = pair_count[pair_jk_idx,1]
#         for symm_count in range(symm_count_start, symm_count_end , 1):
#             eta = count_Gparam[symm_count, 0]               #ang_params[0]
#             zeta = count_Gparam[symm_count, 1]              #ang_params[1]
#             lambd = count_Gparam[symm_count, 2]             #ang_params[2]
#
#             cos_angle = cos_angle_calc(Rij, Rik, Rjk)
#             rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta)
#
#             Gfunc_data[at_idx_i, 0, symm_count] += angular_filter_ind(Rij, Rik, Rjk, eta, zeta, lambd, cos_angle, rad_filter_for_ang)
#
#
#
#             ddRij_G = ddRij_G_calc_mod(Rij, Rik, Rjk, eta, zeta, lambd, cos_angle, rad_filter_for_ang);
#             ddRik_G = ddRik_G_calc_mod(Rij, Rik, Rjk, eta, zeta, lambd, cos_angle, rad_filter_for_ang);
#             ddRjk_G = ddRjk_G_calc_mod(Rij, Rik, Rjk, eta, zeta, lambd, cos_angle, rad_filter_for_ang);
#
#             # dG_all_dict[(at_idx_i, at_idx_j, at_idx_k, symm_count)] = (ddx_Rij, ddy_Rij, ddz_Rij,
#             #                                                             ddx_Rik, ddy_Rik, ddz_Rik,
#             #                                                             ddx_Rjk, ddy_Rjk, ddz_Rjk,
#             #                                                             ddRij_G, ddRik_G, ddRjk_G)
#
#             symm_count_0 = symm_count - symm_count_start
#             angular_idx  = get_angle_idx(at_idx_i, jk_idx, n_pairs)
#             #print("Angular Index", angular_idx, " symm_count_0, ", symm_count_0)
#             ang_precalc[angular_idx, symm_count_0, 0]  =  ddRij_G
#             ang_precalc[angular_idx, symm_count_0, 1]  =  ddRik_G
#             ang_precalc[angular_idx, symm_count_0, 2]  =  ddRjk_G
#
# return Gfunc_data, ang_precalc #dG_all_dict

def symm_func_mol_arr( rad_filter_vec, ang_filter_vec,
                        at_ele_arr,
                        neighbourlist_arr, neighbourpair_arr,
                        ele_count, pair_count,
                        n_atoms, n_symm_func, n_rad_count, n_ang_count, n_ele):
    """
    Refactor the entire symmetry function calculation into
    arr operations to speed up.

    (count refers to the symmetry function vector count )

            Args:

                at_ele_arr: return the element for a given atom index.
                            (Reverse dictionary of at_idx_map)
                neighbourlist_arr: np.array, return the structure of the
                            neighbourlist for the given atom.
                            neighbourlist_arr[at_idx] = [neightbour atom idx]
                            (For use in radial symmetry function)
                neighbourpair_arr: np.array, return the structure of the
                            all the neightbour pairs of the given atom.
                            neightbourpari_arr[at_idx]
                            = [(at1, at2)] (at1 and at2 are neighbours of the
                                            given atom with index of at_idx)


                count_Gparam: a np.array that has the structure
                            count_Gparam[count]  = Gparam_list (specified for the given count)

                            count_Gparam = (count, max No. Params)


                count_dict: Return the count for the given atom
                            count_dict['element'] = [counts for that element]
                            count_dict['elemental pair'] = [counts for that elemental pair]

                n_atoms: number of atoms
                n_symm_func: number of symmetry functions


            Outputs:

               Gfunc_data: symmetry function values;
                            dictionary with 1st layer keys = atom types,
                                2nd layer keys = atom indexes,
                                values = 2D arrays with shape=(1, n_symm_func)

                            Gfunc_data[element][at_idx] = (1, n_symm_func)

               (New) Gfunc_data[at_idx] = (1, n_symm_func)
               New Data structure should be adopted when the input can be organzied.
               It is much more organized since this is an numpy array.
               Even though The New Data structure looks like it only supports
               fixed number of atoms, since the data will be converted into input,
               it is still able to work with multiple different arrays.

    Comments:
    Working In Progress.
    Only designed the framework, not implemented yet.

    Attention:
    Neighbour list will change with different configurations.
    A simpler way might be to loop through every of the atom, and then
    just check for the distances .


    """



    Gfunc_data = np.zeros(shape=(n_atoms, 1, n_symm_func)) # New Gfunc structure

    # This for loop goes through all atoms
    for at_idx_i in np.arange(n_atoms):

        # This for loop goes through all neighbours of the atom
        # For the Radial Components
        for at_idx_j in neighbourlist_arr[at_idx_i]:
            #Rij = Calculation.get_distance(at_idx_i, at_idx_j, n_atoms, distance_arr)
            at_j_ele = at_ele_arr[at_idx_j]

            pair_idx = get_pair_idx(at_idx_i, at_idx_j)


            # Get the corresponding counts in the vector
            for symm_count in count_dict[at_j_ele]:

                #Gfunc_data[at_idx_i,0, symm_count] += radial_filter_ind(Rs, eta, Rij)
                Gfunc_data[at_idx_i, 0, symm_count] += rad_filter_vec[pair_idx, symm_count]

        # This for loop goes through all neighbours pairs of the atoms
        # For the angular components
        # Go through all the neighbour pairs
        for neighbour_pair in neighbourpair_arr[at_idx_i]:
            at_idx_j = neighbour_pair[0]
            at_idx_k = neighbour_pair[1]

            at_i_ele = at_ele_arr[at_idx_i]
            at_j_ele = at_ele_arr[at_idx_j]
            at_k_ele = at_ele_arr[at_idx_k]

            angle_idx = get_angle_idx(at_idx_i, at_idx_j, at_idx_k)



            # Get the corresponds counts in the vector
            for symm_count in count_dict[pair_jk]:
                # ang_params = count_Gparam[symm_count]
                # eta = ang_params[0]
                # zeta = ang_params[1]
                # lambd = ang_params[2]

                Gfunc_data[at_idx_i, 0, symm_count] += angular_filter_ind_vec[angle_idx, symm_count]
        return Gfunc_data
