"""
##################################################

Test Module:  test_utils

Description:
Generate necessary shortcuts for the testing


TODO:
0. Separate the import glycine molecule into another
module.
##################################################
"""
import os
import numpy as np
import pandas as pd
import Utils.DirNav  import get_project_dir
from ase.io import read, write
import Testing.src_nogrd as src_nogrd
import Calculator.CompileArr as CompileArr
import Calculator.SymmFuncIndPython as SymmFuncIndPython
import Calculator.SymmFuncIndCython
from md_capture.dftb_traj_io import read_dftb_atom_from_traj
import matplotlib.pyplot as plt


def import_glycine_mol(filename='glycine.xyz'):
    project_dir = get_project_dir()
    glycine_test = read(os.path.join(project_dir, 'md_sampling', 'example_molecules', filename))
    print(glycine_test)

    at_idx_map = src_nogrd.at_idx_map_generator(glycine_test)
    model_dir  = os.path.join(project_dir, 'model', 'glycine')

    return glycine_test, at_idx_map



def get_glycine_Gfunc():
    """By the Compiled Vector
    """

    glycine_test, at_idx_map, Gparam_dict = import_glycine_mol()

    n_atoms = len(glycine_test)

    at_ele_map, at_ele_arr, ele_dict = src_nogrd.at_ele_arr_generator(glycine_test)
    ele_count, pair_count, count_Gparam, n_symm_func, n_ang_count, n_ele = CompileArr.compile_Gparam_const(
                                 at_idx_map, ele_dict, Gparam_dict)
    Gfunc_Cython, xyzArr, distance_arr, neighbourlist_arr, neighbourlist_count, neighbourpair_arr, neighbourpair_count, ang_precalc= SymmFuncIndCython.symm_func_mol_compiled(
                                glycine_test, at_ele_arr,
                                count_Gparam, ele_count, pair_count,
                                n_atoms, n_symm_func, n_ang_count, n_ele)
    return Gfunc_Cython



"""
####################

To Get the Dihedral Angle From the MD Trajectory

####################
"""

def get_glycine_OCCN(glycine):
    dih_OCCN = glycine.get_dihedral(a1=0, a2=8, a3=2, a4=5)
    return dih_OCCN

def get_glycine_HNCC(glycine):
    dih_HNCC = glycine.get_dihedral(a1=0, a2=2, a3=5, a4=6)
    return dih_HNCC


def get_glycine_HOCO(glycine):
    dih_HOCO = glycine.get_dihedral(a1=1, a2=0, a3=8, a4=9)
    return dih_HOCO


def get_dih(md_samples_arr):
    """Get the Dihedral for samples of glycine molecule configurations.

    """
    arr_len = len(md_samples_arr)
    # OCCN_arr =  np.zeros(arr_len)
    # HNCC_arr =  np.zeros(arr_len)
    # HOCO_arr =  np.zeros(arr_len)

    OCCN_list = [0.0] * arr_len
    HNCC_list = [0.0] * arr_len
    HOCO_list = [0.0] * arr_len

    for idx, mol in md_samples_arr.iteritems():
        OCCN_temp = get_glycine_OCCN(mol)
        HNCC_temp = get_glycine_HNCC(mol)
        HOCO_temp = get_glycine_HOCO(mol)


        #
        # OCCN_list.append(OCCN_temp)
        # HNCC_list.append(HNCC_temp)
        # HOCO_list.append(HOCO_temp)
        #

        OCCN_list[idx] = OCCN_temp
        HNCC_list[idx] = HNCC_temp
        HOCO_list[idx] = HOCO_temp

    OCCN_arr = np.array(OCCN_list)
    HNCC_arr = np.array(HNCC_list)
    HOCO_arr = np.array(HOCO_list)


    return OCCN_arr, HNCC_arr, HOCO_arr


def plot_dih(sample_name, save_dir, show=False):
    """Generate the Dihedral Angle Distribution for the DFTB Trajectory file
    for OCCN, HNCC, HOCO, and save them to the figures.


    """


    sample_file_name = '../../md_sampling/samples/' + sample_name  + '/geom.out.xyz'
    md_samplesArr, md_Mulliken = read_dftb_atom_from_traj(sample_file_name, writeTrajectory=False)
    print(md_Mulliken.shape)
    print(md_samplesArr.shape)


    OCCN_arr, HNCC_arr, HOCO_arr = get_dih(md_samplesArr)

    plt.figure()
    plt.hist(OCCN_arr, bins=35, range=(0, 360), label='O-C-C-N', alpha=0.5)
    plt.hist(HNCC_arr, bins=35, range=(0, 360), label="H-N-C-C", alpha=0.5)
    plt.hist(HOCO_arr, bins=35, range=(0, 360), label='H-O-C-O', alpha=0.5)
    plt.xlabel("Dihedral Angle (Degree)")
    plt.legend(loc='upper right')
    plt.title(sample_name + " Dihedral Angle Distribution")
    if show == True:
        plt.show()

    save_name = os.path.join(save_dir, sample_name) + '-hist.png'
    plt.savefig(save_name)
    plt.close()

    return



if __name__ == '__main__':
    Gfunc_data = get_glycine_Gfunc()
    print(Gfunc_data)
