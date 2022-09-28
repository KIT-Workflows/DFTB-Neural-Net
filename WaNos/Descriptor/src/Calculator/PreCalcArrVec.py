"""
##################################################

Module:  PreCalcVec

Brief:
Vectorized Calculation of PreCalc Using Using Numpy
To speed up for large scale system simulations.

In Progress !


##################################################
"""

import numpy as np
import math





"""
##############################

Vectorized Computation

##############################
"""


def cos_angle_vec_calc(Rij_vec, Rik_vec, Rjk_vec) :
    cos_angle_vec = (np.square(Rij_vec) + np.square(Rik_vec) - np.square(Rjk_vec))/(2.0 * Rij_vec * Rik_vec);

    return cos_angle_vec;

def rad_filter_for_ang_calc(Rij_vec, Rik_vec, Rjk_vec, eta) :
    rad_filter = np.exp(-eta*   np.square( (Rij_vec + Rik_vec + Rjk_vec)));
    return rad_filter;


def ddRij_G_vec_calc(ddRij_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang):
    """Modified Version of the derivative by introducing the cache to reduce repeated calculation.

    """
    #cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    #cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);

    ddRij_precalc[ang_count, :, :] = (-2.0**(2.0-zeta) * rad_filter_for_ang * eta * (Rij_vec + Rik_vec + Rjk_vec)
              * np.power(1.0 + lambd * cos_angle_vec, zeta)
              + (2.0**(1.0-zeta)) * rad_filter_for_ang  * (lambd/Rik_vec - lambd * cos_angle_vec / Rij_vec)
              * np.power((1.0 + lambd * cos_angle_vec), (-1.0+zeta)) * zeta)
    return


def ddRik_G_vec_calc(ddRik_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang):
    """ Modified version of derivative by introducing cache to reduce the repeated computation.


    """
    #cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    #cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);



    ddRik_precalc[ang_count, :, :] = (-2.0**(2.0-zeta) * rad_filter_for_ang * eta * (Rij_vec + Rik_vec + Rjk_vec)
              * np.power(1.0 + lambd * cos_angle_vec, zeta)
              + 2.0**(1.0-zeta) * rad_filter_for_ang * (lambd/Rij_vec - lambd * cos_angle_vec / Rik_vec)
              * np.power((1.0 + lambd * cos_angle_vec), (-1.0+zeta)) * zeta);
    return



def ddRjk_G_vec_calc(ddRjk_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec,  eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang):
    """ Modified version of the derivative by the introduction of cache and
    reduce repeated calculation.


    """
    #cdef NPdouble_t cos_angle = cos_angle_calc(Rij, Rik, Rjk);
    #cdef NPdouble_t rad_filter_for_ang = rad_filter_for_ang_calc(Rij, Rik, Rjk, eta);

    ddRjk_precalc[ang_count, :, :] = (- 2.0**(2.0-zeta)   * rad_filter_for_ang * eta * (Rij_vec + Rik_vec + Rjk_vec)
              * np.power((1.0 + lambd * cos_angle_vec), zeta)
              - (   (2.0**(1.0-zeta))         * rad_filter_for_ang * lambd * Rjk_vec
              * np.power( (1.0 + lambd * cos_angle_vec), (-1.0+zeta)) * zeta) / (Rij_vec*Rik_vec));

    return



def rad_filter_ind_vec(Rs, eta, Rij_vec) :
    """radial filter for symmetry functions

            Args:
                Rs, eta: radial symmetry function parameters; float
                Rij_vec: distance values between two given atoms i and j;
                1D numpy array of length Nsamples

            Outputs:
                G_rad_ij_vec: radial filter values; 1D numpy array of length nb_samples

    """
    #G_rad_ij = math.exp(-eta * (Rij-Rs)**2)
    G_rad_ij_vec = np.exp(-eta * np.square(Rij_vec - Rs))
    ddRij_G_vec  = -2*G_rad_ij_vec*eta*(Rij_vec-Rs)
    return G_rad_ij_vec, ddRij_G_vec


def angular_filter_ind_vec(ang_precalc, ang_count, zeta ,lambd, cos_angle_vec, rad_filter_vec):
    ang_precalc[ang_count, :, :]= 2.0 ** (1.0-zeta) * np.power( (1.0+lambd*cos_angle_vec), zeta) * rad_filter_vec
    return






"""
##############################

For-Loop Calculations

##############################
"""
def symm_rad_arr_precalc(Rij_vec, rad_Gparam, n_rads, rad_count_each, n_samples):
    """To generate the dG(rad)/dRij and other intensive calculations by applying
    the 1D vector on numpy to speed up, save in memory and then used for other
    calculations of symmetry function and derivatives.

    For Purposes of performance, we avoid the 3D numpy array but uses 2 2D array
    to reduce the number of reading on np.strides.

            Args:
                Rij_vec: (np.array) vectors for all the Rij values for all the
                         pairs wants to be calculated. Rij_vec has shape (n_rads,)
                rad_Gparam: (np.array) returns the parameters for the
                        symmetry function (in one pair)
                n_rads: (int) number of total radial pairs (ij pairs)
                rad_count_each: (int) the numer of symmetry functions for each
                        element.

            Outputs:
                rad_precalc_ind_vec: (np.array) of all the precalculated values
                        of G(rad)
                        Warning: rad_precalc_ind_vec has a different shape than
                        other pre-calculate functions (Because of numpy stride
                        operations, Junmian swtiched the dimention to make it
                        easier for calculation )
                drad_precalc_ind_vec: (np.array) of all the precalculated values
                        of dG(rad)/dRij





    """
    rad_precalc= np.zeros(shape=(rad_count_each, n_rads, n_samples), dtype=np.float32)
    drad_precalc= np.zeros(shape=(rad_count_each, n_rads, n_samples), dtype=np.float32)

    for rad_count in range(rad_count_each):
        # Load the Parameters for the given symmetry function
        eta = rad_Gparam[rad_count, 1]
        Rs  = rad_Gparam[rad_count, 0]
        #import pdb; pdb.set_trace()
        rad_temp, drad_temp = rad_filter_ind_vec(Rs, eta, Rij_vec)
        rad_precalc[rad_count, :, :] = rad_temp
        drad_precalc[rad_count, :, :] = drad_temp

        # if rad_count == 1:
        #     print(eta, Rs)

    return rad_precalc, drad_precalc




def symm_ang_arr_precalc_buf(Rij_vec, Rik_vec, Rjk_vec, ang_Gparam, n_angs, ang_count_each, ang_Gparam_rev):
    """Just for Demo. The parameter buffering alogrithm.
    Central Idea: Same Parameter, Don't Calculate Twice.

            Args:
                    ang_Gparam_rev: reversed dictionary for angular G symmetry
                                    parameter list (ang_Gparam). Take parameter tuple
                                    return count.
                                    ang_Gparam_rev[(eta, zeta, lambd)] = ang_count
                    WARNING: python dictionary may make it impossible to run on
                            Cython or do multi-thread processing.
    """


    for eta in eta_array:
        # Calculate Short Cuts that only depends on eta
        for lambd in lambd_array:
            # Calculate short cuts that only depends on lambd
            for zeta in zeta_array:
                # Calculate All Properties.
                ang_count = ang_Gparam_rev[(eta, zeta, lambd)]


    return
#def symm_ang_precalc():

#    pass

def symm_ang_arr_precalc(Rij_vec, Rik_vec, Rjk_vec, ang_Gparam, n_angs, ang_count_each, n_samples):
    """To generate the dG(angular)/dRij and other massive calculations by applying
    the 1D vector on numpy to speed up, save in memory and then used for other
    calculations of symmetry function and derivatives. .

            Args:
                Rij_vec, Rik_vec, Rjk_vec:
                        vectors for the Rij, Rik and Rjk for all the angles.
                        have shape (n_angs,)
                        each of those are unique to the angle

                n_angs: number of angles in the system
                        n_angs = n_atoms * n_pairs
                        n_pairs = truncate( n_atoms * (n_atoms - 1) / 2))

                ang_count_each: number of symmetry functions for the angular component
                            of the symmetry function OF 1 pair.

                            e.g. If the angular components has a total of 60 symm funcs.
                                 by 3 pairs 'HH', 'OH' and 'OO'. Each pair takes
                                 only 60 symmetry function vectors. Then
                                 n_ang_count = 60.


            Outputs:
                ang_precalc_vec: dG/dRij, dRik, dRjk
                    ang_precalc_vec[n_angs, n_ang_count(each), 3]
                    (The Same with ang_precalc in structure)

                ang_precalc_vec[ang_ijk, ang_count] = (dG/dRij, dG/dRik, dG/dRjk, cos_angle, rad_filter)

    Comment:
    This might be a for loop that are able to replace all the for loops in other
    strcuture. However, in the symmetry function, the data structure looks like
        `Gfunc_data[at index, symm func idx].`
    It is difficult to directly loop the entire symmetry function without indexing.

    Better ways are needed for the replacing of entire for loop.

    Also, to generate the Rij_vec, Rik_vec and Rjk is still time-consuming.
    """
    ang_precalc = np.zeros( shape=(ang_count_each, n_angs, n_samples), dtype = np.float32)
    ddRij_G_precalc = np.zeros( shape=(ang_count_each, n_angs, n_samples), dtype=np.float32)
    ddRik_G_precalc = np.zeros( shape=(ang_count_each, n_angs, n_samples), dtype=np.float32)
    ddRjk_G_precalc = np.zeros( shape=(ang_count_each, n_angs, n_samples), dtype=np.float32)


    # Loop over all angles
    #for ang_ijk in range(0, n_angs, 1):


    # Note:
    # ang_count is only the angular portion of the symm_count for each individual
    # angle
    # TODO: cos_angle_vec and rad_filter for ang is only dependent on eta
    # Add the eta buffer.
    for ang_count in range(0, ang_count_each, 1):
        eta = ang_Gparam[ang_count, 0]               #ang_params[0] ! Change symm_count
        zeta = ang_Gparam[ang_count, 1]              #ang_params[1]
        lambd = ang_Gparam[ang_count, 2]             #ang_params[2]

        cos_angle_vec = cos_angle_vec_calc(Rij_vec, Rik_vec, Rjk_vec)
        rad_filter_for_ang = rad_filter_for_ang_calc(Rij_vec, Rik_vec, Rjk_vec, eta)

        # Put the precalc array into the func to reduce copy
        angular_filter_ind_vec(ang_precalc, ang_count, zeta, lambd, cos_angle_vec, rad_filter_for_ang)

        ddRij_G_vec_calc(ddRij_G_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang)
        ddRik_G_vec_calc(ddRik_G_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang)
        ddRjk_G_vec_calc(ddRjk_G_precalc, ang_count, Rij_vec, Rik_vec, Rjk_vec, eta, zeta, lambd, cos_angle_vec, rad_filter_for_ang)

        #ang_precalc[ang_count, :] = ang_filter_vec
        #ddRij_G_precalc[ang_count, :] = ddRij_G_vec
        #ddRik_G_precalc[ang_count, :] = ddRik_G_vec
        #ddRjk_G_precalc[ang_count, :] = ddRjk_G_vec


        # Release the temporary vectors to free memory
        cos_angle_vec = None
        rad_filter_for_ang_vec = None
    return ang_precalc, ddRij_G_precalc, ddRik_G_precalc, ddRjk_G_precalc
