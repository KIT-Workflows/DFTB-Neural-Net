from __future__ import print_function
import numpy as np
from libc.math cimport exp, pow, sqrt, trunc
cimport numpy as np
#from cython.parallel import prange
#from cython.parallel import prange

"""
Global Variables (To Speed up)
"""
cdef long no_str = 0
cdef long i_str = 1
cdef long j_str = 2
cdef long k_str = 3



"""
Numpy Data Types:
"""
NPdouble = np.float64
NPint    = np.int32

ctypedef np.float64_t NPdouble_t
ctypedef np.int32_t   NPint_t

cdef struct XYZ:
  NPdouble_t x
  NPdouble_t y
  NPdouble_t z


cdef struct DIST:
    NPdouble_t dist    # Distance  = R
    NPdouble_t dist_dx # dR/dx
    NPdouble_t dist_dy # dR/dy
    NPdouble_t dist_DZ # dR/dz


"""
##########

From CompileArr.py

##########
"""


cdef long get_pair_idx(NPint_t ele_idx_1, NPint_t ele_idx_2, long n_ele) nogil:
    """Take the index of the element pair, and return the indices in the
    pair_count (array)




    Explinataion:
    Using the same algorithm as the function `distance_xyz_index`

    Fundamentally a number lock for the pairs in the 1D array. So that
    (m, n) will have the unique number.


    """
    cdef long m, n
    cdef double index

    if ele_idx_1 > ele_idx_2:
        m = <long> ele_idx_2
        n = <long> ele_idx_1
    else:
        m = <long> ele_idx_1
        n = <long> ele_idx_2

    index = trunc(n_ele * m - m * (m+1) / 2 + (n - m) - 1)
    return <long>index

cdef long get_angle_idx(NPint_t at_i, NPint_t jk_idx, long n_pairs):
    """An pair to access the angle for between atom i, j, k.


          Args:
            at_i: atom index for atom i.
            jk_idx: the distance index for atomic pair (at_j, at_k)
            n_pairs: equal to the number of pairs totally available.
                      n_pairs = int((n_atoms) * (n_atoms - 1) / 2 )

    Mathematical Property:
      CosAngle(i, j, k)  = CosAngle(i, k, j)
      CosAngle(j, i, k) != CosAngle(j, i, k)

    Trick:
    The pair_idx for the distance already have the property that
    Pair_idx(j, k) = Pair_idx(k, j)
    Therefore, J. Zhu decided to use the jk_idx to directly access
    (i, j, k)  = (i, (j, k))


    """
    cdef long angle_idx
    angle_idx = at_i * n_pairs + jk_idx

    return angle_idx


"""
##########

Calculation.py Files

#########
"""



cdef Py_ssize_t distance_xyz_index(long at1, long at2, long n_atoms) nogil:
    """Take the index of two atoms and return the indices in the distance_xyz

            Args:
                at1, at2: (int) index of atom 1 and atom 2
                n_atoms: number of atoms

    Explaination (Algorithm):
    Junmian Zhu has shown mathematically that the index for the tuple (m,n) is
    index[ (m,n) ] = Stack[m] + (n-m)
    where Stack[m] is the sum of
    ((n_atom-1), (n_atom-2), ....)

    By the formular of sum for a series,
    Stack[m] = n_atoms * m - m * (m+1) / 2

    Keep in Mind, m must be the smaller index by convention.

    """
    cdef long m
    cdef long n
    cdef double index
    if at1 > at2:
        m = at2
        n = at1
    else:
        m = at1
        n = at2


    index = trunc(n_atoms * m - m * (m+1) / 2 + (n - m) - 1)
    return <Py_ssize_t>index






# WARNING: Change to the vectorized Version!
cdef XYZ ddR_radial_filter_ind(NPdouble_t ddRij_G, NPdouble_t ddx_dRij, NPdouble_t ddy_dRij, NPdouble_t ddz_dRij,
                              long at_idx_i, long at_idx_j, long which_is_ref) nogil:
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




    cdef NPdouble_t ddx_dG = ddx_dRij * ddRij_G

    cdef NPdouble_t ddy_dG = ddy_dRij * ddRij_G

    cdef NPdouble_t ddz_dG = ddz_dRij * ddRij_G

    if which_is_ref == i_str:
        # In case of dG/dxj, there is an extra negative sign.
        ddx_dG = -ddx_dG
        ddy_dG = -ddy_dG
        ddz_dG = -ddz_dG

    #cdef np.ndarray[NPdouble_t, ndim=1] output_arr = np.array([ddxi_dG, ddyi_dG, ddzi_dG])

    cdef XYZ xyz_result
    xyz_result.x = ddx_dG
    xyz_result.y = ddy_dG
    xyz_result.z = ddz_dG

    return xyz_result
    #return output_arr





cdef XYZ ddR_angular_filter_ind_stacked(
                                    NPdouble_t ddR1_G, NPdouble_t ddR2_G,
                                    NPdouble_t ddx_R1, NPdouble_t ddx_R2,
                                    NPdouble_t ddy_R1, NPdouble_t ddy_R2,
                                    NPdouble_t ddz_R1, NPdouble_t ddz_R2,
                                    ) nogil:
    """
    Calculate dG/dx Using the Pre-Calculated Version.

    dG/dx = dG/dR * dR/dx

    Both dG/dR and dR/dx has been pre-calculated.

    An generic verision that attempts to simplify all the representation
    (Rather than divide to 3 functions each for i,j,k)

    R1, R2 Rules:  i < j < k, and follows the sequence


    """

    cdef NPdouble_t ddx_G = ddR2_G * ddx_R2 + ddR1_G * ddx_R1;
    cdef NPdouble_t ddy_G = ddR2_G * ddy_R2 + ddR1_G * ddy_R1;
    cdef NPdouble_t ddz_G = ddR2_G * ddz_R2 + ddR1_G * ddz_R1;

    #cdef np.ndarray[NPdouble_t, ndim=1] xyz_result = np.empty(3, NPdouble)
    cdef XYZ xyz_result;
    xyz_result.x = ddx_G
    xyz_result.y = ddy_G
    xyz_result.z = ddz_G

    return xyz_result




cdef XYZ ddR_angular_filter_ind_sep_stacked(
                                NPdouble_t ddRij_G, NPdouble_t ddRik_G, NPdouble_t ddRjk_G,
                                NPdouble_t ddx_Rij, NPdouble_t ddy_Rij, NPdouble_t ddz_Rij,
                                NPdouble_t ddx_Rik, NPdouble_t ddy_Rik, NPdouble_t ddz_Rik,
                                NPdouble_t ddx_Rjk, NPdouble_t ddy_Rjk, NPdouble_t ddz_Rjk,

                                long at_idx_i, long at_idx_j, long at_idx_k,
                                long xyz_str) nogil:
    """
    Stacked Version To calculate dG/dx

    Formular:

    dG/dx = dG/dR * dR/dx

    """
    # TODO: May Move This Part in the for-loop to reduce the number of comparison
    # ddR1_G portion can be moved into the major for-loop
    cdef NPdouble_t ddR1_G, ddR2_G
    cdef NPdouble_t ddx_R1, ddy_R1, ddz_R1
    cdef NPdouble_t ddx_R2, ddy_R2, ddz_R2


    if xyz_str == i_str: # i -> (ij, jk)
        ddR1_G = ddRij_G
        ddR2_G = ddRik_G

        if at_idx_i < at_idx_j:
            ddx_R1 = - ddx_Rij
            ddy_R1 = - ddy_Rij
            ddz_R1 = - ddz_Rij

        else: # at_idx_i > at_idx_j
            ddx_R1 = ddx_Rij
            ddy_R1 = ddy_Rij
            ddz_R1 = ddz_Rij

        if at_idx_i < at_idx_k:
            ddx_R2 = - ddx_Rik
            ddy_R2 = - ddy_Rik
            ddz_R2 = - ddz_Rik

        else: # at_idx_i > at_idx_j
            ddx_R2 = ddx_Rik
            ddy_R2 = ddy_Rik
            ddz_R2 = ddz_Rik


    elif xyz_str == j_str:
        ddR1_G = ddRij_G
        ddR2_G = ddRjk_G


        if at_idx_j < at_idx_i:
            ddx_R1 = - ddx_Rij
            ddy_R1 = - ddy_Rij
            ddz_R1 = - ddz_Rij

        else: # at_idx_j > at_idx_i
            ddx_R1 = ddx_Rij
            ddy_R1 = ddy_Rij
            ddz_R1 = ddz_Rij

        if at_idx_j < at_idx_k:
            ddx_R2 = - ddx_Rjk
            ddy_R2 = - ddy_Rjk
            ddz_R2 = - ddz_Rjk

        else: # at_idx_i > at_idx_j
            ddx_R2 = ddx_Rjk
            ddy_R2 = ddy_Rjk
            ddz_R2 = ddz_Rjk


    elif xyz_str == k_str:
        ddR1_G = ddRik_G
        ddR2_G = ddRjk_G

        if at_idx_k < at_idx_i:
            ddx_R1 = - ddx_Rik
            ddy_R1 = - ddy_Rik
            ddz_R1 = - ddz_Rik

        else: # at_idx_k > at_idx_i
            ddx_R1 = ddx_Rik
            ddy_R1 = ddy_Rik
            ddz_R1 = ddz_Rik

        if at_idx_k < at_idx_j:
            ddx_R2 = - ddx_Rjk
            ddy_R2 = - ddy_Rjk
            ddz_R2 = - ddz_Rjk

        else: # at_idx_k > at_idx_j
            ddx_R2 = ddx_Rjk
            ddy_R2 = ddy_Rjk
            ddz_R2 = ddz_Rjk

    ## Warning:
    ## Potential xyz_str may not equal to any of those str.

    return ddR_angular_filter_ind_stacked(
                                    ddR1_G, ddR2_G,
                                    ddx_R1, ddx_R2,
                                    ddy_R1, ddy_R2,
                                    ddz_R1, ddz_R2)



"""
##########

Stacked Version For the For-Loop Functions

##########
"""






cpdef np.ndarray[NPdouble_t, ndim=2] symm_deriv_ind_pile_vec(
                        np.ndarray[NPdouble_t, ndim=2] distance_arr,
                        np.ndarray[NPint_t, ndim=1] at_ele_arr,
                        np.ndarray[NPint_t, ndim=2] rad_arr,
                        np.ndarray[NPint_t, ndim=2] ang_arr,
                        np.ndarray[NPint_t, ndim=2] ele_count, np.ndarray[NPint_t, ndim=2] pair_count,
                        np.ndarray[NPdouble_t, ndim=2] drad_precalc,
                        np.ndarray[NPdouble_t, ndim=2] dRij_precalc,
                        np.ndarray[NPdouble_t, ndim=2] dRik_precalc,
                        np.ndarray[NPdouble_t, ndim=2] dRjk_precalc,
                        long n_atoms, long n_symm_func, long n_ele, long at_ref,
                        long n_rads, long n_angs,
                        long rad_count_each, long ang_count_each):
    """Piled Up Version to calculate the derivative of the entire molecule.


          Outputs:
            Gfunc_deriv_pile = (n_atoms * n_symm_func, 3)
            which contains the derivative for all the atoms in the molecule.
            So that
            nn_deriv_G_pile (matmul) Gfunc_deriv_pile = derivative for the reference atom

    """

    cdef NPdouble_t xi, yi, zi, xj, yj, zj, #xk, yk, zk


    # Pre-Declare all c type variables
    cdef bint i_equal_ref
    cdef bint j_equal_ref
    cdef bint k_equal_ref

    cdef NPint_t at_idx_i, at_idx_j, at_idx_k
    cdef NPint_t ele_idx_i, ele_idx_j, ele_idx_k
    cdef long pair_jk_idx
    #cdef long at_idx_j
    #cdef long at_idx_k
    cdef long which_is_ref

    cdef long dist_idx
    cdef Py_ssize_t ij_idx, ik_idx, jk_idx


    # Distance Related Variable
    cdef NPdouble_t Rij
    cdef NPdouble_t Rik
    cdef NPdouble_t Rjk
    cdef NPdouble_t ddRjk_G, ddRij_G, ddRik_G
    cdef NPdouble_t ddx_Rij, ddx_Rjk, ddx_Rik
    cdef NPdouble_t ddy_Rij, ddy_Rjk, ddy_Rik
    cdef NPdouble_t ddz_Rij, ddz_Rjk, ddz_Rik


    cdef NPdouble_t dG_dx, dG_dy, dG_dz

    #cdef ddR1_G, ddR2_G
    #cdef ddx_R1, ddy_R1, ddz_R1
    #cdef ddx_R2, ddy_R2, ddz_R2
    cdef DIST dist_output

    # Parameters
    cdef NPdouble_t eta, zeta, lambd, Rs


    cdef long which_equal_ref

    # Iterator for symmetry functions
    cdef long symm_count, rad_count, ang_count
    cdef long symm_count_start
    cdef long symm_count_end
    cdef long n_pairs = <long> n_atoms * (n_atoms - 1) / 2

    cdef NPint_t nlist_count_i, npair_count_i
    cdef long  nlist_count, npair_count
    cdef long at_non_ref
    cdef long at_ref_count, at_non_ref_count


    cdef long ele_ref, ele_non
    cdef long symm_count_ref, symm_count_non
    cdef long symm_count_start_ref, symm_count_start_non
    cdef long at_i_count

    cdef long rad_idx, ang_idx


    cdef XYZ temp_result


    cdef np.ndarray[NPdouble_t, ndim=2] Gfunc_deriv_data = np.zeros([n_atoms * n_symm_func, 3], NPdouble )


    # WARNING: ERROR Spotted: once the radial part is added to nogil,
    # The Result is wrong.
    # The angular part is fine.
    for rad_idx in range(0, n_rads, 1 ):
    #for rad_idx in prange(0, n_rads, 1, nogil = True):
        at_idx_i = rad_arr[rad_idx, 0]
        at_idx_j = rad_arr[rad_idx, 1]


        if at_idx_i == at_ref:
            which_is_ref = i_str
            at_non_ref = at_idx_j
        elif at_idx_j == at_ref:
            which_is_ref = j_str
            at_non_ref = at_idx_i
        else:
            which_is_ref = no_str
            continue

        ij_idx = distance_xyz_index(at_idx_i, at_idx_j, n_atoms)

        if at_ref > at_non_ref:
          ddx_Rij = distance_arr[ij_idx,1]
          ddy_Rij = distance_arr[ij_idx,2]
          ddz_Rij = distance_arr[ij_idx,3]
        else:
          ddx_Rij = -distance_arr[ij_idx,1]
          ddy_Rij = -distance_arr[ij_idx,2]
          ddz_Rij = -distance_arr[ij_idx,3]


        ele_ref = at_ele_arr[at_ref]
        ele_non = at_ele_arr[at_non_ref]



        symm_count_start_ref = ele_count[ele_ref, 0]
        symm_count_start_non = ele_count[ele_non, 0]

        for rad_count in range(0, rad_count_each, 1):
            symm_count_ref = symm_count_start_ref + rad_count
            symm_count_non = symm_count_start_non + rad_count
            ddRij_G = drad_precalc[rad_count, rad_idx]


            # Trick: among (i,j), only 1 can be the at_ref
            # WARNING: WORK IN PROGRESS

            # WARNING: This Part is against Mathematical Derivation by Junmian
            # Junmian Zhu Admit that he does not know what is happening.
            # However, the results agrees with the Compiled Version.
            # And the compiled version agrees with the Numerical Calculation.
            # temp_result =  ddR_radial_filter_ind(ddRij_G, ddx_Rij, ddy_Rij, ddz_Rij, at_idx_i, at_idx_j, which_is_ref)
            # at_ref_count = at_ref * n_symm_func + symm_count_non
            # at_non_ref_count = at_non_ref * n_symm_func + symm_count_ref
            # Gfunc_deriv_data[at_ref_count, 0] += temp_result.x
            # Gfunc_deriv_data[at_ref_count, 1] += temp_result.y
            # Gfunc_deriv_data[at_ref_count, 2] += temp_result.z
            #
            # Gfunc_deriv_data[at_non_ref_count, 0] += temp_result.x
            # Gfunc_deriv_data[at_non_ref_count, 1] += temp_result.y
            # Gfunc_deriv_data[at_non_ref_count, 2] += temp_result.z


            dG_dx = ddx_Rij * ddRij_G
            dG_dy = ddy_Rij * ddRij_G
            dG_dz = ddz_Rij * ddRij_G

            at_ref_count = at_ref * n_symm_func + symm_count_non
            at_non_ref_count = at_non_ref * n_symm_func + symm_count_ref

            Gfunc_deriv_data[at_ref_count, 0] += dG_dx
            Gfunc_deriv_data[at_ref_count, 1] += dG_dy
            Gfunc_deriv_data[at_ref_count, 2] += dG_dz

            Gfunc_deriv_data[at_non_ref_count, 0] += dG_dx
            Gfunc_deriv_data[at_non_ref_count, 1] += dG_dy
            Gfunc_deriv_data[at_non_ref_count, 2] += dG_dz




            # if at_ref == 1 & at_idx_i == 1:
            #   import sys
            #   print("i = ", at_idx_i, "j = ", at_idx_j)
            #   print("at_ref", at_ref, "at_non", at_non_ref)
            #   print("Symm_count_ref:", at_ref_count, "Symm_count_non:", at_non_ref_count)
            #   print('ref value', -temp_result.x, 'non value', +temp_result.x)
            #   sys.stdout.flush()
            # Not at_ref -> Minus x,y,z

    for ang_idx in range(0, n_angs, 1):
        at_idx_i = ang_arr[ang_idx, 0]
        at_idx_j = ang_arr[ang_idx, 1]
        at_idx_k = ang_arr[ang_idx, 2]


        if  at_idx_i == at_ref:
            which_is_ref = i_str
        elif at_idx_j == at_ref:
            which_is_ref = j_str
        elif at_idx_k == at_ref:
            which_is_ref = k_str
        else:
            continue

        ij_idx = distance_xyz_index(at_idx_i, at_idx_j, n_atoms)

        ik_idx = distance_xyz_index(at_idx_i, at_idx_k, n_atoms)
        jk_idx = distance_xyz_index(at_idx_j, at_idx_k, n_atoms)


        # Get the Distance and Derivative  (Store for next calculation)

        ddx_Rij = distance_arr[ij_idx,1]
        ddy_Rij = distance_arr[ij_idx,2]
        ddz_Rij = distance_arr[ij_idx,3]

        ddx_Rjk = distance_arr[jk_idx,1]
        ddy_Rjk = distance_arr[jk_idx,2]
        ddz_Rjk = distance_arr[jk_idx,3]

        ddx_Rik = distance_arr[ik_idx,1]
        ddy_Rik = distance_arr[ik_idx,2]
        ddz_Rik = distance_arr[ik_idx,3]

        ele_idx_j = at_ele_arr[at_idx_j]
        ele_idx_k = at_ele_arr[at_idx_k]

        pair_jk_idx = get_pair_idx(ele_idx_j, ele_idx_k, n_ele)




        symm_count_start = pair_count[pair_jk_idx, 0]
        symm_count_end   = pair_count[pair_jk_idx, 1]
        for symm_count in range(symm_count_start, symm_count_end, 1):
            ang_count = symm_count - symm_count_start

            ddRij_G = dRij_precalc[ang_count, ang_idx]
            ddRik_G = dRik_precalc[ang_count, ang_idx]
            ddRjk_G = dRjk_precalc[ang_count, ang_idx]

            temp_result = ddR_angular_filter_ind_sep_stacked(
                                              ddRij_G, ddRik_G, ddRjk_G,
                                              ddx_Rij, ddy_Rij, ddz_Rij,
                                              ddx_Rik, ddy_Rik, ddz_Rik,
                                              ddx_Rjk, ddy_Rjk, ddz_Rjk,
                                              at_idx_i, at_idx_j, at_idx_k,
                                              which_is_ref)

            at_i_count = at_idx_i * n_symm_func + symm_count

            Gfunc_deriv_data[at_i_count, 0] += temp_result.x
            Gfunc_deriv_data[at_i_count, 1] += temp_result.y
            Gfunc_deriv_data[at_i_count, 2] += temp_result.z



    return Gfunc_deriv_data
