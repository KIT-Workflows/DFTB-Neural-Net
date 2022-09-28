"""

Module: Test_SymmDerivInd

Brief:
To test the calculation of the symmetry function derivative


"""

import unittest
import Calculator.CompileArr as CompileArr
import Calculator.CompileVec as CompileVec
import Calculator.SymmDerivIndCython  as SymmDerivIndCython
import Calculator.SymmDerivIndVecCython as SymmDerivIndVecCython
import Calculator.SymmFuncIndCython as SymmFuncIndCython
import Calculator.SymmFuncIndVecCython as SymmFuncIndVecCython
import Calculator.PreCalcVec as PreCalcVec
import Training.src_nogrd as src_nogrd
import Calculator.ModelIO  as ModelIO


from Utils.DirNav import get_project_dir

import os
import numpy as np
from ase.io import read


class TestDeriv(unittest.TestCase):

    def setUp(self):
        pass



    def test_SymmDerivVec(self):

        project_dir = get_project_dir()
        glycine_test = read(os.path.join(project_dir, 'md_sampling', 'example_molecules', 'glycine_opt.xyz'))
        save_dir = '/home/junmian/Applications/Archive/model/glycine_non_opt_b3lyp631G_400K_20000sNoFirstDrop'
        Gparam_dict = ModelIO.read_data_from_pickle(save_dir, "Gparam_dict.pkl")

        n_atoms = len(glycine_test)  #

        at_idx_map = src_nogrd.at_idx_map_generator(glycine_test)


        at_ele_map, at_ele_arr, ele_dict = src_nogrd.at_ele_arr_generator(glycine_test)
        ele_count, pair_count, count_Gparam, n_symm_func, n_ang_count, n_ele = CompileArr.compile_Gparam_const(
                                     at_idx_map, ele_dict, Gparam_dict)
        Gfunc_Cython, xyzArr, distance_arr, neighbourlist_arr, neighbourlist_count, neighbourpair_arr, neighbourpair_count, ang_precalc= SymmFuncIndCython.symm_func_mol_compiled(
                                    glycine_test, at_ele_arr,
                                    count_Gparam, ele_count, pair_count,
                                    n_atoms, n_symm_func, n_ang_count, n_ele)

        G_deriv_xyz_pile_compiled = SymmDerivIndCython.symm_deriv_ind_pile_stacked(distance_arr, at_ele_arr, xyzArr,
                                    neighbourlist_arr, neighbourlist_count,
                                    neighbourpair_arr, neighbourpair_count,
                                    count_Gparam,
                                    ele_count, pair_count,
                                    ang_precalc,
                                    n_atoms, n_symm_func, n_ele, 1)
        #print(G_deriv_xyz_pile_compiled)


        n_rads =  int(n_atoms * (n_atoms-1) /2)
        n_angs  = int((n_atoms * (n_atoms-1) * (n_atoms - 2) / 2))


        ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, n_ele, eta_arr = CompileVec.compile_Gparam_vec(
                                                at_idx_map, ele_dict, Gparam_dict)

        rad_arr = CompileVec.get_rads(n_atoms, n_rads)
        ang_arr = CompileVec.get_angles(n_atoms, n_angs)
        Rij_rad = SymmFuncIndVecCython.get_distance_tensor_rad(distance_arr, rad_arr, n_atoms, n_rads)
        Rij_vec, Rik_vec, Rjk_vec = SymmFuncIndVecCython.get_distance_tensor(distance_arr, ang_arr, n_atoms, n_angs)
        rad_precalc, drad_precalc = PreCalcVec.symm_rad_ind_precalc(Rij_rad, rad_Gparam, n_rads, rad_count_each)
        ang_precalc, ddRij_G_precalc, ddRik_G_precalc, ddRjk_G_precalc= PreCalcVec.symm_ang_ind_precalc(Rij_vec, Rik_vec, Rjk_vec, ang_Gparam, eta_arr, n_angs, ang_count_each)

        # Get the reference atom 0
        G_deriv_xyz_pile_vec = SymmDerivIndVecCython.symm_deriv_ind_pile_vec(
                                    distance_arr,
                                    at_ele_arr,
                                    rad_arr, ang_arr,
                                    ele_count, pair_count,
                                    drad_precalc,
                                    ddRij_G_precalc,
                                    ddRik_G_precalc,
                                    ddRjk_G_precalc,
                                    n_atoms, n_symm_func, n_ele, 1,
                                    n_rads, n_angs,
                                    rad_count_each, ang_count_each)

        #print(G_deriv_xyz_pile_vec)

        temp_diff = G_deriv_xyz_pile_compiled - G_deriv_xyz_pile_vec
        print(temp_diff.shape)
        print(temp_diff[276:276+276+1, 0])
        print(np.max(temp_diff))

if __name__ == '__main__':
    unittest.main()
