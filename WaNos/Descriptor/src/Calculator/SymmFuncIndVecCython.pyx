"""


This is the module for the processing of symmetry function
for individual molecules.


"""



import pandas as pd
import src.Calculator.src_nogrd as src_nogrd
import matplotlib.pyplot as plt
import src.Calculator.Calculation as Calculation

import numpy as np
from libc.math cimport exp, pow, sqrt, trunc
cimport numpy as np
import time

from cython.parallel import prange
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
ctypedef np.int32_t NPint_t

cdef struct XYZ:
  NPdouble_t x
  NPdouble_t y
  NPdouble_t z

cdef struct DIST:
  NPdouble_t dist    # Distance  = R
  NPdouble_t dist_dx # dR/dx
  NPdouble_t dist_dy # dR/dy
  NPdouble_t dist_dz # dR/dz



# Mode Specification:
# Turn on PRINT MODE

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






cdef long get_angle_idx(NPint_t at_i, NPint_t jk_idx, long n_pairs) nogil:
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

cpdef get_distance_tensor_rad(np.ndarray[NPdouble_t, ndim=2] distance_arr, np.ndarray[NPint_t, ndim=2] rad_arr, long n_atoms, long n_rads):

  cdef np.ndarray[NPdouble_t, ndim=1] Rij_rad = np.zeros([n_rads], NPdouble)

  cdef long rad_idx
  cdef NPint_t at_i, at_j
  cdef long ij_idx

  for rad_idx in range(0, n_rads, 1):
    at_i = rad_arr[rad_idx, 0]
    at_j = rad_arr[rad_idx, 1]
    ij_idx = distance_xyz_index(at_i, at_j, n_atoms)
    Rij_rad[rad_idx] = distance_arr[ij_idx, 0]

  return Rij_rad

cpdef get_distance_tensor_ang(np.ndarray[NPdouble_t, ndim=2] distance_arr, np.ndarray[NPint_t, ndim=2] angle_arr, long n_atoms, long n_angs):
  """For the given distance_arr of the molecule, get the Rij_vec, Rik_vec, Rjk_vec
  of all the possible angles (i,j,k) (i,j,k represents the individual atom index)

      Args:
        distance_arr: distances for the entire molecule.
        angle_arr:    all the angle (i,j,k) present in the molecule.

  """

  cdef np.ndarray[NPdouble_t, ndim=1] Rij_vec = np.zeros( [n_angs], NPdouble)
  cdef np.ndarray[NPdouble_t, ndim=1] Rik_vec = np.zeros( [n_angs], NPdouble)
  cdef np.ndarray[NPdouble_t, ndim=1] Rjk_vec = np.zeros( [n_angs], NPdouble)

  cdef long angle_idx
  cdef NPint_t at_i, at_j, at_k,
  cdef long ij_idx, ik_idx, jk_idx

  for angle_idx in range(0, n_angs, 1):
  #for i,j,k in angle_arr:
      at_i = angle_arr[angle_idx, 0]
      at_j = angle_arr[angle_idx, 1]
      at_k = angle_arr[angle_idx, 2]
      ij_idx = distance_xyz_index(at_i,at_j, n_atoms)
      ik_idx = distance_xyz_index(at_i,at_k, n_atoms)
      jk_idx = distance_xyz_index(at_j,at_k, n_atoms)
      Rij_vec[angle_idx]  = distance_arr[ij_idx, 0]
      Rik_vec[angle_idx]  = distance_arr[ik_idx, 0]
      Rjk_vec[angle_idx]  = distance_arr[jk_idx, 0]

  return Rij_vec, Rik_vec, Rjk_vec




"""
##########

From Calculation.py

##########
"""
cdef long distance_xyz_index(long at1, long at2, long n_atoms) nogil:
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




cdef DIST distance_xyz_ind(NPdouble_t x1, NPdouble_t y1, NPdouble_t z1, NPdouble_t x2, NPdouble_t y2, NPdouble_t z2):
    """Use the C-Definition to calculate the distances between two atoms.

          Args:
            x1, y1, z1: xyz of atom 1
            x2, y2, z2: xyz of atom 2

          Outputs:
            distance: distance between atom 1 and atom 2

    Explaination:
    In C, it is faster to access individual element of the numpy array (than getting
    an xyz and then access its x, y, z, component)
    Also, it is faster to calculate the distance directly for individual scalers.
    Therefore, the xyz are broken in to individual x,y,z component to speed up.
    """

    cdef NPdouble_t dx, dy, dz
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    cdef NPdouble_t dist = sqrt(dx*dx + dy*dy + dz*dz)

    cdef NPdouble_t dist_dx = dx / dist
    cdef NPdouble_t dist_dy = dy / dist
    cdef NPdouble_t dist_dz = dz / dist

    cdef DIST output
    output.dist = dist
    output.dist_dx = dist_dx
    output.dist_dy = dist_dy
    output.dist_dz = dist_dz

    return output



cpdef distance_xyz_mol(np.ndarray[NPdouble_t, ndim=2] xyz_arr, long n_atoms):
    """calculate distances from coordinates, and the derivatives

            Args:

                xyz_arr: coordinates; 2D numpy array of shape (n_atoms, 3)
                n_atoms: number of atoms


            Outputs:
                distances_xyz: distance values in the form of a list (& also
                                its derivatives )
                    distances_xyz[index] = [[distance ij, distance deriv ij ], ... ]
                    distances_xyz[index, 0] = distance
                    distances_xyz[index, 1:3] = dR/dx, dR/dy, dR/dz



    Ways to Get the Atom Index from xyz_arr
    xyz_arr[atom_index] = (x,y,z)


    Comment:
    There is a major change from the Thuong's original code.
    Instead of using the pandas frame, the indexing algorithm mentioned above
    will be used to access the distance_xyz (np.array) It might help to speed up.

    TODO: Convert this section into C code.

    """

    # Ways to Get the Atom Index from XYZ_to_arr
    #

    # Define Counts
    cdef long count = 0
    cdef long at1, at2

    # Define Temp Vars
    cdef NPdouble_t x1, y1, z1, x2, y2, z2
    cdef NPdouble_t dx, dy, dz
    cdef DIST output

    #distance_arr = [0.0] * int( (n_atoms * (n_atoms - 1)/2) )
    #distance_arr = np.zeros(shape = int( n_atoms * (n_atoms - 1) / 2) )
    cdef long n_dist = <long>(n_atoms * (n_atoms -1) / 2)





    cdef np.ndarray[NPdouble_t, ndim=2] distance_arr = np.zeros( [n_dist, 4], NPdouble )
    #cdef np.ndarray[NPdouble_t, ndim=2] Gfunc_deriv_data = np.zeros([n_symm_func, 3], NPdouble )

    #for at1 in np.arange(n_atoms):
    for at1 in range(0, n_atoms, 1):
        x1 = xyz_arr[at1, 0]
        y1 = xyz_arr[at1, 1]
        z1 = xyz_arr[at1, 2]

        for at2 in range(at1+1, n_atoms, 1):
            x2 = xyz_arr[at2, 0]
            y2 = xyz_arr[at2, 1]
            z2 = xyz_arr[at2, 2]

            # dx = x2 - x1
            # dy = y2 - y1
            # dz = z2 - z1

            output = distance_xyz_ind(x1, y1, z1, x2, y2, z2)

            distance_arr[count, 0] = output.dist
            distance_arr[count, 1] = output.dist_dx
            distance_arr[count, 2] = output.dist_dy
            distance_arr[count, 3] = output.dist_dz

            count += 1

    return distance_arr



"""
##########

From SymmFuncIndPython

##########
"""




cpdef symm_func_mol_vecidx(np.ndarray[NPint_t, ndim=1] at_ele_arr,
                        np.ndarray[NPint_t, ndim=2] rad_arr, np.ndarray[NPint_t, ndim=2] ang_arr,
                        np.ndarray[NPint_t, ndim=2] ele_count , np.ndarray[NPint_t , ndim=2] pair_count,
                        np.ndarray[NPdouble_t, ndim=2] rad_precalc, np.ndarray[NPdouble_t, ndim=2] ang_precalc,
                        long n_atoms, long n_symm_func, long n_ele, long n_rads, long n_angs, long rad_count_each, long ang_count_each):
    """
    Refactor the entire symmetry function calculation into
    arr operations to speed up.

    (count refers to the symmetry function vector count )

            Args:

                distance_arr: distances array
                            (Generated by distance_arr_generator function )
                at_ele_arr: return the element's index  for a given atom index.
                            (Reverse dictionary of at_idx_map)
                rad_arr: np.array of all the combos of the radial part (i,j)
                ang_arr: np.array of all the combos of the angular (i,j,k)
                n_atoms: number of atoms
                n_symm_func: number of symmetry functions
                n_ele:   number of elements


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
    Working In Progress on the vectorized version.

    Attention:
    Neighbour list will change with different configurations.
    A simpler way might be to loop through every of the atom, and then
    just check for the distances .


    """
    cdef NPint_t at_i, at_j, at_k
    cdef NPdouble_t Rij, Rjk, Rik
    cdef NPdouble_t Rs, eta, zeta, lambd

    cdef NPdouble_t rad_filter

    cdef long symm_count
    cdef long symm_count_start, symm_count_end
    cdef long symm_count_start_i, symm_count_start_j
    cdef long rad_count, ang_count
    cdef long n_pairs = <long> n_atoms * (n_atoms - 1) / 2
    cdef long ele_i, ele_j

    cdef long rad_idx
    cdef long angular_idx


    cdef long ij_idx, jk_idx


    cdef np.ndarray[NPdouble_t, ndim=3] Gfunc_data = np.zeros([n_atoms, 1, n_symm_func], NPdouble) # New Gfunc structure



    #dG_all_dict = {}


    cdef NPint_t ele_idx_j, ele_idx_k
    cdef long pair_jk_idx

    cdef NPint_t nlist_count
    cdef long nlist_count_i
    cdef NPint_t npair_count
    cdef long npair_count_i


    for rad_idx in range(0, n_rads, 1):
        at_i = rad_arr[rad_idx,0]
        at_j = rad_arr[rad_idx,1]

        ele_i = at_ele_arr[at_i]
        ele_j = at_ele_arr[at_j]

        symm_count_start_i = ele_count[ele_i, 0]
        symm_count_start_j   = ele_count[ele_j, 0]

        for rad_count in range(0, rad_count_each, 1):
            symm_count_i = symm_count_start_i + rad_count
            symm_count_j = symm_count_start_j + rad_count
            rad_filter = rad_precalc[rad_count, rad_idx]
            Gfunc_data[at_i, 0, symm_count_j] += rad_filter
            Gfunc_data[at_j, 0, symm_count_i] += rad_filter


    for ang_idx in range(0, n_angs, 1):
        at_i = ang_arr[ang_idx, 0]
        at_j = ang_arr[ang_idx, 1]
        at_k = ang_arr[ang_idx, 2]

        ele_j = at_ele_arr[at_j]
        ele_k = at_ele_arr[at_k]

        pair_jk_idx = get_pair_idx(ele_j, ele_k, n_ele)
        symm_count_start = pair_count[pair_jk_idx, 0]
        symm_count_end   = pair_count[pair_jk_idx, 1] # TODO: Redundant
        for symm_count in range(symm_count_start, symm_count_end, 1):
            ang_count = symm_count - symm_count_start
            Gfunc_data[at_i, 0, symm_count] += ang_precalc[ang_count, ang_idx]

    return Gfunc_data
