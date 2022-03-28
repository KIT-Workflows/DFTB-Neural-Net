"""
##################################################


Module: CompileVec

Brief:
The Vectorized Version of the CompileArr.

Purpose:
Since MD simulation is only for one strucuture, many things are repeated.
Therefore, it might be helpful to "compile" the individual molecules into
vectors to store some information that are constant throughout the MD process.




#WARNING:
Don't Change the position of the hyperparameters in the sequence.

#################################################
"""
import numpy as np




def compile_Gparam_vec(at_idx_map, ele_dict , Gparam_dict):
    """Independently Generate the iterator array directly from the Gparam_dict
    Assume that the symmetry function vector (numbers) does not vary for different
    elements. Thus, can use np.array instead of a nested list for the
    outputs to speed up the process.


                Args:
                    at_idx_map: atom index map
                    Gparam_dict:


                Outputs:
                    (Original Design: Dictionary)
                    count_dict['element'] = [counts for that element]
                    count_dict['elemental pair'] = [counts for that elemental pair]
                    count_Gparam[symm_count] = Gparam for the given count in
                                                the symmetry function vector.

                    (New Version - Array)
                    ele_arr[element index] = [count.start, count.end ] (for the element)
                    pair_arr[pair lock] = [count.start, count.end] (for the given count)
                        where pair_lock is calculated from  the element idx
                    count_Gparam[symm_count] = Gparam for the given count in the
                                             symmetry function vector


    Automatically loop through the

    See Also:
    symm_func_process_benchmark, generate the arrays from the old symmetry function.

    Comments:
    Maybe the constant mode is not necessary, since it still needs to loop the
    entire at_idx_map.

    # TODO:
    Automatically support multiple symmetry functions by
    looping through the Gparam_dict
    """

    # Randomly select the first element
    at_type_rand = next(iter(at_idx_map))

    # Randonly generate the element pair
    pair_rand = at_type_rand + at_type_rand

    Gparam_rad = Gparam_dict[at_type_rand]['rad']
    Gparam_ang = Gparam_dict[at_type_rand]['ang']

    # Get the number of symmetry functions
    # Below is the total number of symm_func in radial or angular component
    rad_count = sum([Gparam_rad[t].shape[0] for t in Gparam_rad.keys()])
    ang_count = sum([Gparam_ang[t].shape[0] for t in Gparam_ang.keys()])

    rad_cutoff = Gparam_rad[at_type_rand][0, 2]
    ang_cutoff = Gparam_ang[pair_rand][0, 3]

    total_count = rad_count + ang_count
    n_symm_func = total_count

    # Below is the number of symm_func for each element type or
    # atomic pairs
    rad_count_each = Gparam_rad[at_type_rand].shape[0]
    ang_count_each = Gparam_ang[pair_rand].shape[0]


    # Get the number of parameters in the Gparam_dict
    n_rad_params = len(Gparam_rad[at_type_rand][0])
    n_ang_params = len(Gparam_ang[pair_rand][0])


    n_ele = len(ele_dict)     # Number of elements

    ele_count = np.zeros( shape = (n_ele, 2), dtype=np.int32)
    pair_count = np.zeros( shape =   (int( (n_ele * (n_ele - 1)/2)), 2) , dtype=np.int32)


    # Prepare both in one loop
    # Notice: loop with respect to Gparam_rad rather than at_idx_map
    # Just to be consistent with the original symmetry function


    count = 0

    for at_type in Gparam_rad.keys():
        #count_dict[at_type] = np.arange(count, count+rad_count_each)
        ele_idx = ele_dict[at_type]
        ele_count[ele_idx][0:2] = [count, count + rad_count_each]
        #ele_count[ele_idx][1] = count + rad_count_each

        count += rad_count_each




    for atAatB_type in Gparam_ang.keys():
        #count_dict[atAatB_type] = np.arange(count, count+ang_count_each)
        ele_1 = atAatB_type[0]
        ele_2 = atAatB_type[1]

        ele_idx_1 = ele_dict[ele_1]
        ele_idx_2 = ele_dict[ele_2]

        pair_idx = get_pair_idx(ele_idx_1, ele_idx_2, n_ele)
        pair_count[pair_idx, 0:2] = [count, count+ ang_count_each]
        count += ang_count_each


    # Prepare  count_Gparam
    #count_Gparam = np.zeros(shape=(total_count, max_params), dtype=np.float32)
    rad_Gparam = np.zeros(shape=(rad_count_each, n_rad_params), dtype=np.float32)
    ang_Gparam = np.zeros(shape=(ang_count_each, n_ang_params), dtype=np.float32)

    for count, values in enumerate(Gparam_rad[at_type_rand]):
        for idx_n, value_n in enumerate(values):
            rad_Gparam[count, idx_n] = value_n

    for count, values in enumerate(Gparam_ang[pair_rand]):
        for idx_n, value_n in enumerate(values):
            ang_Gparam[count, idx_n] = value_n


    # Find eta array
    eta_arr = get_eta(Gparam_ang)


    return ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, rad_cutoff, ang_cutoff, n_ele, eta_arr






def get_eta(Gparam_ang):
    """Get all values of eta hyperparameter in the ang_Gparam for future
    simplified calculations.

        Args:
            ang_Gparam: from the Gparam_dict, follow sequence
                        ang_Gparam[idx] = [eta, zeta, lambd]

        Outputs:
            eta_arr:   All values of eta in the array that appears in ang_Gparam
                       [eta_1, eta_2, eta_3]

    """

    eta_arr = []

    eta_temp = 0

    ang_params = next(iter(Gparam_ang.values()))

    for params in ang_params:
        eta = params[0]
        if eta == eta_temp:
            continue
        elif eta in eta_arr:
            continue
        else:
            eta_arr.append(eta)

    return np.array(eta_arr, dtype=np.float32)








def get_pair_idx(ele_idx_1, ele_idx_2, n_ele):
    """Take the index of the element pair, and return the indices in the
    pair_count (array)




    Explinataion:
    Using the same algorithm as the function `distance_xyz_index`

    Fundamentally a number lock for the pairs in the 1D array. So that
    (m, n) will have the unique number.


    """

    if ele_idx_1 > ele_idx_2:
        m = ele_idx_2
        n = ele_idx_1
    else:
        m = ele_idx_1
        n = ele_idx_2

    index = n_ele * m - m * (m+1) / 2 + (n - m) - 1
    return int(index)

def get_rads(n_atoms, n_rads):
    """Fundamentally, get all possible (i,j) combos for the radial pairs. In the
    vectorized operations, will replace all the neighbour lists.

    Maths Property: (Symmetry Function)
    G(i,j) = G(j,i)
    j != i


            Args:
                n_atoms: (int) number of atoms in the molecule

            Outputs:
                rad_arr: np.array of shape(n_rads, 2)
                         rad_arr[rad_idx] = (i,j)

    """

    rad_arr = np.zeros(shape=(n_rads ,2), dtype=np.int32)

    rad_idx = 0
    for at_i in range(0, n_atoms, 1):
        for at_j in range(at_i+1, n_atoms, 1):
            rad_arr[rad_idx, 0] = at_i
            rad_arr[rad_idx, 1] = at_j
            #rad_arr.append([at_i, at_j])
            rad_idx += 1
    return rad_arr




def get_angles(n_atoms, n_angs):
    """Fundamentally, get all the possible (i,j,k) for the angles pairs for the
    entire molecule. Output angle_arr should be able to replace the original
    indexing system. (which involves too much maths)

    Maths Property:
    G(i,j,k) = G(i,k,j)
    i != j, j!= k


            Args:
                n_atoms: (int) number of atoms in the molecules.

            Outputs:
                angle_arr: np.array of shape (n_angles, 3)
                            angle_arr[angle_idx] = (i,j,k)
    """

    #np.ndarray[NPint_t, ndim=2] angle_arr = np.zeros([n_angs,3],NPint)
    angle_arr = np.zeros(shape=(n_angs, 3), dtype=np.int32)
    #angle_arr = []

    angle_idx = 0
    for at_i in range(0, n_atoms, 1):
        for at_j in range(0, n_atoms, 1):
            if at_j == at_i:
                continue
            for at_k in range(at_j+1, n_atoms, 1):
                if at_k == at_i:
                    continue
                #angle_arr.append([at_i, at_j, at_k])
                angle_arr[angle_idx,0] = at_i
                angle_arr[angle_idx,1] = at_j
                angle_arr[angle_idx,2] = at_k
                angle_idx += 1

    return angle_arr

