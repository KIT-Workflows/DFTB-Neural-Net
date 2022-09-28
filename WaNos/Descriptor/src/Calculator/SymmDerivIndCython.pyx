from __future__ import print_function
import numpy as np
from libc.math cimport exp, pow, sqrt, trunc
cimport numpy as np
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
NPdouble = np.float32
NPint    = np.int32

ctypedef np.float32_t NPdouble_t
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


cdef long get_pair_idx(NPint_t ele_idx_1, NPint_t ele_idx_2, long n_ele):
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



cdef long distance_xyz_index(long at1, long at2, long n_atoms):
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
    return <long>index



cpdef get_pair(at_str_1, at_str_2, list ang_list):
    """Get the element pair ('HH') for atom 1 and 2

            Args:
                at_str_1, at_str_2: string like 'H' that represent the element
                                    of the string
                ang_list: all the combination of the string in the symmetry
                          function vector
                        ["HH", "HO", "OO"]


            Outputs:
                pair_str: "HH" in the anglist

    Explaination:
    It is possible that "HN" is in the list, but "NH" is not.
    Thus, this function might be helpful.

    TODO:
    Replace all the string with the number (hash table),
    so it is able to run on C.
    """

    pair_str_1 = at_str_1 + at_str_2


    if pair_str_1 in ang_list:
        return pair_str_1
    else:
        pair_str_2 = at_str_2 + at_str_1
        return pair_str_2


"""
##########

SymmDerivInd.py Files

#########
"""






cdef XYZ ddR_radial_filter_ind(NPdouble_t Rs, NPdouble_t eta, NPdouble_t Rij,
                                    #np.ndarray[NPdouble_t, ndim=1] xyz_i, np.ndarray[NPdouble_t, ndim=1] xyz_j,
                                    NPdouble_t xi, NPdouble_t yi, NPdouble_t zi,
                                    NPdouble_t xj, NPdouble_t yj, NPdouble_t zj,
                                    long which_is_ref):
                                                            #np.ndarray[NPdouble_t, ndim=1] xyz_i, np.ndarray[NPdouble_t, ndim=1] xyz_j,
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
    if which_is_ref != i_str and which_is_ref != j_str:
        print("which_is_ref should be i or j")



    cdef NPdouble_t ddRij_G = -2*exp(-eta * (Rij-Rs)**2)*eta*(Rij-Rs)

    cdef NPdouble_t ddxi_dRij = (xi-xj) / Rij
    cdef NPdouble_t ddxi_dG = ddxi_dRij * ddRij_G

    cdef NPdouble_t ddyi_dRij = (yi-yj) / Rij
    cdef NPdouble_t ddyi_dG = ddyi_dRij * ddRij_G

    cdef NPdouble_t ddzi_dRij = (zi-zj) / Rij
    cdef NPdouble_t ddzi_dG = ddzi_dRij * ddRij_G

    if which_is_ref == j_str:
        # In case of dG/dxj, there is an extra negative sign.
        ddxi_dG = -ddxi_dG
        ddyi_dG = -ddyi_dG
        ddzi_dG = -ddzi_dG


    cdef XYZ xyz_result
    xyz_result.x = ddxi_dG
    xyz_result.y = ddyi_dG
    xyz_result.z = ddzi_dG

    return xyz_result





"""
##########

Numerical Calculation Functions

##########
"""







cdef NPdouble_t cos_angle_calc(NPdouble_t Rij, NPdouble_t Rik, NPdouble_t Rjk):
    cdef NPdouble_t cos_angle;
    cos_angle = (pow(Rij, 2.0) + pow(Rik, 2.0) - pow(Rjk, 2.0))/(2.0 * Rij * Rik);

    return cos_angle;

cdef NPdouble_t rad_filter_for_ang_calc(NPdouble_t Rij, NPdouble_t Rik, NPdouble_t Rjk, NPdouble_t eta):
    cdef NPdouble_t rad_filter;
    rad_filter = exp(-eta*   pow( (Rij + Rik + Rjk),2.0));
    return rad_filter;


cdef NPdouble_t ddRij_G_calc_mod(NPdouble_t Rij, NPdouble_t Rik, NPdouble_t Rjk, NPdouble_t eta, NPdouble_t zeta, NPdouble_t lambd):
    """Modified Version of the derivative by introducing the cache to reduce repeated calculation.

    """
    cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);


    cdef NPdouble_t ddRij_G_var = (-pow(2.0, (2.0-zeta)) * rad_filter_for_ang * eta * (Rij + Rik + Rjk)
              * pow(1.0 + lambd * cos_angle, zeta)
              + pow(2.0, (1.0-zeta)) * rad_filter_for_ang  * (lambd/Rik - lambd * cos_angle / Rij)
              * pow((1.0 + lambd * cos_angle), (-1.0+zeta)) * zeta);
    return ddRij_G_var;


cdef NPdouble_t ddRik_G_calc_mod(NPdouble_t Rij, NPdouble_t Rik, NPdouble_t Rjk, NPdouble_t eta, NPdouble_t zeta, NPdouble_t lambd):
    """ Modified version of derivative by introducing cache to reduce the repeated computation.


    """
    cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);



    cdef NPdouble_t ddRik_G_var = (-pow(2.0, (2.0-zeta)) * rad_filter_for_ang * eta * (Rij + Rik + Rjk)
              * pow(1.0 + lambd * cos_angle, zeta)
              + pow(2.0, (1.0-zeta)) * rad_filter_for_ang * (lambd/Rij - lambd * cos_angle / Rik)
              * pow((1.0 + lambd * cos_angle), (-1.0+zeta)) * zeta);
    return ddRik_G_var;



cdef NPdouble_t ddRjk_G_calc_mod(NPdouble_t Rij, NPdouble_t Rik, NPdouble_t Rjk,  NPdouble_t  eta, NPdouble_t zeta, NPdouble_t lambd):
    """ Modified version of the derivative by the introduction of cache and
    reduce repeated calculation.


    """
    cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);

    cdef NPdouble_t ddRjk_G_var = (-pow(2.0, (2.0-zeta) )    * rad_filter_for_ang * eta * (Rij + Rik + Rjk)
              * pow((1.0 + lambd * cos_angle), zeta)
              - (   pow(2,(1.0-zeta))            * rad_filter_for_ang * lambd * Rjk
              * pow( (1.0 + lambd * cos_angle), (-1.0+zeta)) * zeta) / (Rij*Rik));

    return ddRjk_G_var











"""
##########

New Design of Angular Derivative Function
(By pre-calculate dR/dx )

##########
"""





cdef XYZ ddR_angular_filter_ind_stacked(
                                    NPdouble_t ddR1_G, NPdouble_t ddR2_G,
                                    NPdouble_t ddx_R1, NPdouble_t ddx_R2,
                                    NPdouble_t ddy_R1, NPdouble_t ddy_R2,
                                    NPdouble_t ddz_R1, NPdouble_t ddz_R2,
                                    ):
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
                                long xyz_str):
    """
    Stacked Version To calculate dG/dx

    Formular:

    dG/dx = dG/dR * dR/dx

    """

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

    else:
        print("Error")


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





cpdef np.ndarray[NPdouble_t, ndim=2] symm_deriv_ind_pile_stacked(
                        np.ndarray[NPdouble_t, ndim=2] distance_arr, np.ndarray[NPint_t, ndim=1] at_ele_arr,
                        np.ndarray[NPdouble_t, ndim=2] xyzArr,
                        np.ndarray[NPint_t, ndim=2] neighbourlist_arr,  np.ndarray[NPint_t, ndim=1] neighbourlist_count,
                        np.ndarray[NPint_t, ndim=3] neighbourpair_arr,  np.ndarray[NPint_t, ndim=1] neighbourpair_count,
                        np.ndarray[NPdouble_t, ndim=2] count_Gparam,
                        np.ndarray[NPint_t, ndim=2] ele_count, np.ndarray[NPint_t, ndim=2] pair_count,
                        np.ndarray[NPdouble_t, ndim=3] ang_precalc,
                        long n_atoms, long n_symm_func, long n_ele, long at_ref):
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
    cdef long which_is_ref

    cdef long dist_idx


    # Distance Related Variable
    cdef NPdouble_t Rij
    cdef NPdouble_t Rik
    cdef NPdouble_t Rjk
    cdef NPdouble_t ddRjk_G, ddRij_G, ddRik_G
    cdef NPdouble_t ddx_Rij, ddx_Rjk, ddx_Rik
    cdef NPdouble_t ddy_Rij, ddy_Rjk, ddy_Rik
    cdef NPdouble_t ddz_Rij, ddz_Rjk, ddz_Rik

    cdef DIST dist_output

    # Parameters
    cdef NPdouble_t eta, zeta, lambd, Rs


    cdef long which_equal_ref

    # Iterator for symmetry functions
    cdef long symm_count, symm_count_0
    cdef long symm_count_start
    cdef long symm_count_end
    cdef long n_pairs = <long> n_atoms * (n_atoms - 1) / 2

    cdef NPint_t nlist_count_i, npair_count_i
    cdef long  nlist_count, npair_count


    cdef XYZ temp_result


    cdef np.ndarray[NPdouble_t, ndim=2] Gfunc_deriv_data = np.zeros([n_atoms * n_symm_func, 3], NPdouble )
    for at_idx_i in range(0, n_atoms, 1):
      xi = xyzArr[at_idx_i, 0]
      yi = xyzArr[at_idx_i, 1]
      zi = xyzArr[at_idx_i, 2]

      i_equal_ref = (at_idx_i == at_ref)



      nlist_count_i = neighbourlist_count[at_idx_i]
      for nlist_count in range(0, nlist_count_i, 1):
          at_idx_j = neighbourlist_arr[at_idx_i, nlist_count]
          j_equal_ref = (at_idx_j == at_ref)


          if i_equal_ref:
              which_is_ref = i_str
          elif j_equal_ref:
              which_is_ref = j_str
          else:
              which_is_ref = no_str
              continue

          ele_idx_j = at_ele_arr[at_idx_j]

          xj = xyzArr[at_idx_j, 0]
          yj = xyzArr[at_idx_j, 1]
          zj = xyzArr[at_idx_j, 2]

          # Get the corresponding counts in the vector
          symm_count_start = ele_count[ele_idx_j, 0]
          symm_count_end   = ele_count[ele_idx_j, 1]
          for symm_count in range(symm_count_start, symm_count_end,1):
              # Instead of Using pass the distance_arr into another function,
              # Access its content using the numpy array.
              dist_idx = distance_xyz_index(at_idx_i, at_idx_j, n_atoms)
              Rij = distance_arr[dist_idx,0]

              #rad_params = count_Gparam[symm_count]

              Rs = count_Gparam[symm_count, 0]
              eta = count_Gparam[symm_count, 1]

              temp_result = ddR_radial_filter_ind(Rs, eta, Rij, xi, yi, zi, xj, yj, zj, which_is_ref)
              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 0] += temp_result.x
              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 1] += temp_result.y
              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 2] += temp_result.z


      # This for loop goes through all neighbours pairs of the atoms
      # For the angular components
      # Go through all the neighbour pairs
      npair_count_i = neighbourpair_count[at_idx_i]
      for npair_count in range(0, npair_count_i, 1):

      #for neighbour_pair in neighbourpair_arr[at_idx_i]:
          at_idx_j = neighbourpair_arr[at_idx_i, npair_count,0]
          at_idx_k = neighbourpair_arr[at_idx_i, npair_count,1]



          j_equal_ref = (at_idx_j == at_ref)
          k_equal_ref = (at_idx_k == at_ref)
          if i_equal_ref:
              which_is_ref = i_str
          elif j_equal_ref:
              which_is_ref = j_str
          elif k_equal_ref:
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


          # Get the corresponds counts in the vector
          symm_count_start = pair_count[pair_jk_idx, 0]
          symm_count_end   = pair_count[pair_jk_idx, 1]
          for symm_count in range(symm_count_start, symm_count_end, 1):
              symm_count_0 = symm_count - symm_count_start
              angular_idx = get_angle_idx(at_idx_i, jk_idx, n_pairs)

              ddRij_G = ang_precalc[angular_idx, symm_count_0, 0]
              ddRik_G = ang_precalc[angular_idx, symm_count_0, 1]
              ddRjk_G = ang_precalc[angular_idx, symm_count_0, 2]



              temp_result = ddR_angular_filter_ind_sep_stacked(
                                                ddRij_G, ddRik_G, ddRjk_G,
                                                ddx_Rij, ddy_Rij, ddz_Rij,
                                                ddx_Rik, ddy_Rik, ddz_Rik,
                                                ddx_Rjk, ddy_Rjk, ddz_Rjk,
                                                at_idx_i, at_idx_j, at_idx_k,
                                                which_is_ref)



              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 0] += temp_result.x
              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 1] += temp_result.y
              Gfunc_deriv_data[at_idx_i * n_symm_func + symm_count, 2] += temp_result.z

    return Gfunc_deriv_data
