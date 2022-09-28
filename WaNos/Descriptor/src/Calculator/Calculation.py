"""

Module: Calculation

Purpose: This module is designed to hold some of the commonly used functions
for calculation of individual molecules or atoms , so that they can be called
by both the symmetry function calculation and the symmetry function derivative
calculation.

Including:
1. Distance Calculation.
2. Neighbour list calculation.

Potentially, it can be compiled into a c-extensoin

"""

"""
#########

Symmetry Function For Molecules:
(For Numerical Calculation)
(Optimized Version for
individual molecules )

#########
"""
import math
import numpy as np
from scipy.spatial import cKDTree as KDTree

def distance_xyz(xyz_i, xyz_j):
    """

    TODO:
    1. Rewrite in C

    Comments:
    Should be move to the common utility for derivative and symmetry function.
    """

    dx = xyz_j[0] - xyz_i[0]
    dy = xyz_j[1] - xyz_i[1]
    dz = xyz_j[2] - xyz_i[2]

    return math.sqrt(dx*dx+dy*dy+dz*dz)



def distance_xyz_index(at1, at2, n_atoms):
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
    if at1 > at2:
        m = at2
        n = at1
    else:
        m = at1
        n = at2


    index = n_atoms * m - m * (m+1) / 2 + (n - m) - 1
    return int(index)


def distance_xyz_brutal_mol(xyz_arr, n_atoms):
    """calculate distances from coordinates

            Args:

                xyz_arr: coordinates; 2D numpy array of shape (n_atoms, 3)
                n_atoms: number of atoms


            Outputs:
                distances_xyz: distance values in the form of a list



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
    count = 0
    distance_arr = [0.0] * int( (n_atoms * (n_atoms - 1)/2) )
    #distance_arr = np.zeros(shape = int( n_atoms * (n_atoms - 1) / 2) )

    for at1 in np.arange(n_atoms):
        at1_xyz = xyz_arr[at1]
        for at2 in np.arange(at1+1, n_atoms):
            at2_xyz = xyz_arr[at2]
            distance_arr[count]= distance_xyz(xyz_arr[at1], xyz_arr[at2])
            count += 1

    return np.array(distance_arr, dtype=np.float64)


def get_distance(at_idx_1, at_idx_2, n_atoms, distance_arr):
    """Return the distance for the given pair of atom index
    for the distance_xyz_brutal_mol function's distance_arr

            Args:
                at_idx_1, at_idx_2: indexes of the atom
                distance_arr:
                            distance_arr[indexes for (at1, at2)] = distance

            Outputs:
                distance: the distance between atom 1 and atom 2



    """
    index = distance_xyz_index(at_idx_1, at_idx_2, n_atoms)
    if index >= 45:
        import pdb; pdb.set_trace()
    return distance_arr[index]



def get_pair(at_str_1, at_str_2, ang_list):
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
    pair_str_2 = at_str_2 + at_str_1

    if pair_str_1 in ang_list:
        return pair_str_1
    elif pair_str_2 in ang_list:
        return pair_str_2
    else:
        raise ValueError("get_pair: the pair is not in ang_list")


def get_neighbour_cutoff(xyz_arr, rad_cutoff, ang_cutoff):
    """ Get the neighbour list for each atom

    """

    ntree = KDTree(xyz_arr)



    # WARNING: The output is a set, not a ndarray. For loop in Cython, use ndarray.
    rad_arr = ntree.query_pairs(r = rad_cutoff)
    rad_arr = get_rads(rad_arr)

    nlist = get_nest_nlist(xyz_arr, ntree, ang_cutoff)
    ang_rads = ntree.query_pairs(r = ang_cutoff)
    ang_arr = get_angles(ang_rads, nlist)




    n_rads = rad_arr.shape[0]
    n_angs = ang_arr.shape[0]

    return rad_arr, ang_arr, n_rads, n_angs



def get_nest_nlist(xyz_arr, ntree, cutoff_r = 2):
    """Get the nested neighbour listself, return nlist[atom idx] = [neighbour atom idx of i]
    """

    n_atom = xyz_arr.shape[0]
    nlist = []

    for i in range(n_atom):
        nlist_i = ntree.query_ball_point(xyz_arr[i], r = cutoff_r)
        nlist_i.remove(i)
        nlist.append(nlist_i)

    return nlist

def get_rads(pairs):

    rad_arr = []


    for pair in pairs:
        rad_arr.append([ pair[0], pair[1] ])

    return np.array(rad_arr, dtype = np.int32)


def get_angles(pairs, nlist):
    """ Find all the angles (3 atom combinations i,j,k) within a cutoff radius
        Args:
            pairs: the atom pairs within the radius.
            nlist: nested neighbour list, nlist[atom idx] = [neighbour of i]

        Outputs:
            angles: all the 3 atom combos

    TODO: Implement in Cython for the for-loop
    """

    ang_arr = []

    for pair in pairs:
        j = pair[0]
        k = pair[1]

        nlist_j = nlist[j]
        nlist_k = nlist[k]

        nlist_jk = np.intersect1d(nlist_j, nlist_k)

        for i in nlist_jk:
            ang_arr.append( [i,j,k] )

    return np.array(ang_arr, dtype = np.int32)



def get_neighbour(distance_arr, n_atoms):
    """For the molecule, get the neighbour list and neighbour pairs for all of its
    atoms.

            Args:
                distance_arr:
                n_atoms:

            Outputs:
                neighbourlist_arr:
                    neighbourlist_arr[at_idx] = [list of neighbour atom indexes]
                neighbourpair_arr:
                    neighbourpair_arr[at_idx_1] = [(at_idx_2, at_idx_3) that are
                                                    neighbour pairs of at_idx_1]
                neighbourlist_count: number of neighrours for the given atom
                neighbourpair_count: number of neighbour pairs for the given
                                        atom

    Comments:

    Development Stages:
    1. Return all atoms. (Current stage)
    1.5 Then Inside the function, add an if statement to check the distance.
    2. Linear Search the Neighbour and neighbour pairs
    3. Implement the KD Tree structure for neighbour list search.



    """

    # Only Valid for returning all other atoms.
    neighbourlist_arr = np.zeros((n_atoms, n_atoms-1), dtype=np.int32)
    neighbourpair_arr = np.zeros((n_atoms, int( (n_atoms - 1) * (n_atoms - 2 ) / 2),2), dtype=np.int32)
    all_atom_idx_arr = np.arange(n_atoms)
    for at_idx_1 in all_atom_idx_arr:
        neighbourlist_arr[at_idx_1] = np.delete(all_atom_idx_arr, at_idx_1)


        count = 0
        for at_idx_2 in neighbourlist_arr[at_idx_1]:
            # Notice, the indices in the neighbour list will change.
            at_idx_2_idx = np.where(neighbourlist_arr[at_idx_1]==at_idx_2)[0]
            #import pdb; pdb.set_trace()
            neighbour_2nd_arr_temp = np.delete(neighbourlist_arr[at_idx_1], at_idx_2_idx)
            #neighbour_2nd_arr_temp = np.arange(at_idx_2_idx+1, n_atoms-1)

            for at_idx_3 in neighbour_2nd_arr_temp:
                # Only put (smaller index, larger index)
                if at_idx_2 < at_idx_3:
                    neighbourpair_arr[at_idx_1, count, 0] = at_idx_2
                    neighbourpair_arr[at_idx_1, count, 1] = at_idx_3
                    #neighbourpair_arr_spec.append( (at_idx_2, at_idx_3) )
                    count += 1
    neighbourlist_count = np.full( n_atoms, n_atoms - 1, np.int32)
    neighbourpair_count = np.full( n_atoms, int( (n_atoms - 1) * (n_atoms - 2 ) / 2), np.int32)

    return neighbourlist_arr, neighbourpair_arr, neighbourlist_count, neighbourpair_count
