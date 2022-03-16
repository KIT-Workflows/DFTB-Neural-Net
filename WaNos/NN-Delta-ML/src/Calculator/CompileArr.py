"""This module is designed for "compile" the individual molecules into a

Independent Compilation Progress:

1. Compile the molecule into atom index map (at_idx_map) and n_atoms and other
properties that would remain constant through out the MD.

2. Compile the Gparam_dict to generate the necessary iterator array that will
remain constant during the MD simulation.

The idea is if the iteration and the neighbour list searching process can be
separated, and since the iteration should be constant for the given symmetry
function vector, separate those 2 process would accerlerate the process.

Notice:
neighbour list generator should not be included here, since neighbour list will
constantly change over the time.
"""
import numpy as np





def compile_Gparam(at_idx_map, ele_dict, Gparam_dict, mode='const'):
    """Independently Generate the iterator array directly from the Gparam_dict.
    Allow different mode.

                Args:
                    mode:
                        'const': the symmetry function vector does not vary
                                for different elements.
                        'vary': the symmetry function vector varies for different
                                elements

    """

    if mode == 'const':
        count_dict = compile_Gparam_const(at_idx_map, ele_dict, Gparam_dict)
    elif mode == 'vary':
        raise NotImplementedError("Vary Mode Not implemented yet")

    return count_dict







def compile_Gparam_const(at_idx_map, ele_dict , Gparam_dict):
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

    total_count = rad_count + ang_count

    # Below is the number of symm_func for each element type or
    # atomic pairs
    rad_count_each = Gparam_rad[at_type_rand].shape[0]
    ang_count_each = Gparam_ang[pair_rand].shape[0]


    # Get the number of parameters in the Gparam_dict
    n_rad_params = len(Gparam_rad[at_type_rand][0])
    n_ang_params = len(Gparam_ang[pair_rand][0])

    max_params = max(n_rad_params, n_ang_params)

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



    #count_pair = 0

    for atAatB_type in Gparam_ang.keys():
        #count_dict[atAatB_type] = np.arange(count, count+ang_count_each)
        ele_1 = atAatB_type[0]
        ele_2 = atAatB_type[1]

        ele_idx_1 = ele_dict[ele_1]
        ele_idx_2 = ele_dict[ele_2]

        pair_idx = get_pair_idx(ele_idx_1, ele_idx_2, n_ele)
        pair_count[pair_idx, 0:2] = [count, count+ ang_count_each]
        count += ang_count_each



    count = 0

    # Prepare  count_Gparam
    count_Gparam = np.zeros(shape=(total_count, max_params), dtype=np.float32)
    temp_count = 0
    for at_type in at_idx_map:

        for count, values in enumerate( Gparam_rad[at_type]):
            for idx_n, value_n in enumerate(values):
                count_Gparam[temp_count + count, idx_n] = value_n
        temp_count += rad_count_each



    for pair in Gparam_ang.keys():
        for count, values in enumerate(Gparam_ang[pair]):
            for idx_n, value_n in enumerate(values):
                count_Gparam[temp_count + count, idx_n] = value_n
        temp_count += ang_count_each

    n_symm_func = total_count



    return ele_count, pair_count, count_Gparam, n_symm_func, ang_count_each, n_ele


def compile_Gparam_dict(at_idx_map, ele_dict , Gparam_dict):
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

    total_count = rad_count + ang_count

    # Below is the number of symm_func for each element type or
    # atomic pairs
    rad_count_each = Gparam_rad[at_type_rand].shape[0]
    ang_count_each = Gparam_ang[pair_rand].shape[0]


    # Get the number of parameters in the Gparam_dict
    n_rad_params = len(Gparam_rad[at_type_rand][0])
    n_ang_params = len(Gparam_ang[pair_rand][0])

    max_params = max(n_rad_params, n_ang_params)


    # ele_count = np.zeros( shape = (n_ele, 2), dtype=np.int32)
    # pair_count = np.zeros( shape =   (int( (n_ele * (n_ele - 1)/2)), 2) , dtype=np.int32)


    # Prepare both in one loop
    # Notice: loop with respect to Gparam_rad rather than at_idx_map
    # Just to be consistent with the original symmetry function
    count_dict = {}

    count = 0

    for at_type in Gparam_rad.keys():
        count_dict[at_type] = np.arange(count, count+rad_count_each)
        #ele_idx = ele_dict[at_type]
        #ele_count[ele_idx][0:2] = [count, count + rad_count_each]
        #ele_count[ele_idx][1] = count + rad_count_each

        count += rad_count_each



    #count_pair = 0

    for atAatB_type in Gparam_ang.keys():
        count_dict[atAatB_type] = np.arange(count, count+ang_count_each)
        #ele_1 = atAatB_type[0]
        #ele_2 = atAatB_type[1]

        #ele_idx_1 = ele_dict[ele_1]
        #ele_idx_2 = ele_dict[ele_2]

        #pair_idx = get_pair_idx(ele_idx_1, ele_idx_2, n_ele)
        #pair_count[pair_idx, 0:2] = [count, count+ ang_count_each]
        count += ang_count_each


    #
    # count = 0

    # # Prepare  count_Gparam
    # count_Gparam = np.zeros(shape=(total_count, max_params), dtype=np.float64)
    # temp_count = 0
    # for at_type in at_idx_map:
    #
    #     for count, values in enumerate( Gparam_rad[at_type]):
    #         for idx_n, value_n in enumerate(values):
    #             count_Gparam[temp_count + count, idx_n] = value_n
    #     temp_count += rad_count_each
    #
    #
    #
    # for pair in Gparam_ang.keys():
    #     for count, values in enumerate(Gparam_ang[pair]):
    #         for idx_n, value_n in enumerate(values):
    #             count_Gparam[temp_count + count, idx_n] = value_n
    #     temp_count += ang_count_each
    #
    # n_symm_func = total_count



    return count_dict


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

