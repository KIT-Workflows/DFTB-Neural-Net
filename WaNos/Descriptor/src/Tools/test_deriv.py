
import os
import numpy as np
import pandas as pd
import unittest

from ase.io import read
import Calculator.CompileArr as CompileArr
import Calculator.CompileVec as CompileVec
import Calculator.Calculation as Calculation
import Calculator.SymmDerivIndCython  as SymmDerivIndCython
import Calculator.SymmDerivIndVecCython as SymmDerivIndVecCython
import Calculator.SymmFuncIndCython as SymmFuncIndCython
import Calculator.SymmFuncIndVecCython as SymmFuncIndVecCython
import Calculator.PreCalcVec as PreCalcVec
import Training.src_nogrd as src_nogrd
import Calculator.ModelIO as ModelIO
import Calculator.SymmFuncIndPython as SymmFuncInd
import Calculator.ForceNum as ForceNum
from Utils.DirNav import get_project_dir





class test_deriv_dGdx(unittest.TestCase):
    """ A test used to compare the calculation of dG/dx by numerical calculation
    and analytical calculation.
    """

    def setUP(self):
        """Initialize all the calculators, molecules,
        import
        """



    #
    # def test_dftb(self):
    #     """Test the force calculation of DFTB and the numerical derivative calculation
    #     of DFTB.
    #     """
    #     from Calculator.ASEDFTBMod import DftbMod
    #
    #     os.environ['DFTB_PREFIX'] = '/home/junmian/Applications/dftbplus/mio-1-1/'
    #     os.environ['DFTB_COMMAND'] = '/home/junmian/Applications/dftbplus/_install/bin/dftb+'
    #
    #     dftb_calc = DftbMod(
    #                          Hamiltonian_MaxAngularMomentum_='',
    #                          Hamiltonian_MaxAngularMomentum_O='"p"',
    #                          Hamiltonian_MaxAngularMomentum_H='"s"',
    #                          Hamiltonian_MaxAngularMomentum_C='"p"',
    #                          Hamiltonian_MaxAngularMomentum_N='"p"',
    #                          )
    #
    #
    #     project_dir = get_project_dir()
    #     glycine_test = read(os.path.join(project_dir, 'md_sampling', 'example_molecules', 'glycine_opt.xyz'))
    #
    #     glycine_test.set_calculator(dftb_calc)
    #     force_ana = glycine_test.get_forces()
    #
    #     force_num = ForceNum.force_num_mol(glycine_test, dftb_calc, 0.001)
    #
    #     print(force_ana - force_num)
    #




    def test_lib(self):
        project_dir = get_project_dir()
        glycine_test = read(os.path.join(project_dir, 'md_sampling', 'example_molecules', '20_water_self.xyz'))
        save_dir = '/home/junmian/Applications/models/glycine_non_opt_b3lyp631G_400K_100000s_cutoff'
        Gparam_dict = ModelIO.read_data_from_pickle(save_dir, "Gparam_dict.pkl")

        n_atoms = len(glycine_test)  #

        at_idx_map = src_nogrd.at_idx_map_generator(glycine_test)








        n_rads =  int(n_atoms * (n_atoms-1) /2)
        n_angs  = int((n_atoms * (n_atoms-1) * (n_atoms - 2) / 2))

        at_ele_map, at_ele_arr, ele_dict = src_nogrd.at_ele_arr_generator(glycine_test)

        # ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, n_ele, eta_arr = CompileVec.compile_Gparam_vec(
        #                                         at_idx_map, ele_dict, Gparam_dict)
        md_samplesArr = pd.Series([glycine_test])
        n_atoms, xyz_arr = src_nogrd.xyzArr_generator(md_samplesArr)
        #
        # distance_arr = SymmFuncIndCython.distance_xyz_mol(xyzArr, n_atoms)
        #
        # rad_arr = CompileVec.get_rads(n_atoms, n_rads)
        # ang_arr = CompileVec.get_angles(n_atoms, n_angs)
        # Rij_rad = SymmFuncIndVecCython.get_distance_tensor_rad(distance_arr, rad_arr, n_atoms, n_rads)
        # Rij_vec, Rik_vec, Rjk_vec = SymmFuncIndVecCython.get_distance_tensor(distance_arr, ang_arr, n_atoms, n_angs)
        # rad_precalc, drad_precalc = PreCalcVec.symm_rad_ind_precalc(Rij_rad, rad_Gparam, n_rads, rad_count_each)
        # ang_precalc, ddRij_G_precalc, ddRik_G_precalc, ddRjk_G_precalc= PreCalcVec.symm_ang_ind_precalc(Rij_vec, Rik_vec, Rjk_vec, ang_Gparam, eta_arr, n_angs, ang_count_each)
        #
        # Get the reference atom 0



        distance_arr = SymmFuncIndVecCython.distance_xyz_mol(xyz_arr, n_atoms).astype(np.float64)

        ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, rad_cutoff, ang_cutoff, n_ele, eta_arr = CompileVec.compile_Gparam_vec(
                                                at_idx_map, ele_dict, Gparam_dict)

        rad_arr, ang_arr, n_rads, n_angs = Calculation.get_neighbour_cutoff(xyz_arr, rad_cutoff, ang_cutoff)

        Rij_rad = SymmFuncIndVecCython.get_distance_tensor_rad(distance_arr, rad_arr, n_atoms, n_rads)
        Rij_vec, Rik_vec, Rjk_vec = SymmFuncIndVecCython.get_distance_tensor_ang(distance_arr, ang_arr, n_atoms, n_angs)
        rad_precalc, drad_precalc = PreCalcVec.symm_rad_ind_precalc(Rij_rad, rad_Gparam, n_rads, rad_count_each)
        ang_precalc, ddRij_G_precalc, ddRik_G_precalc, ddRjk_G_precalc= PreCalcVec.symm_ang_ind_precalc(Rij_vec, Rik_vec, Rjk_vec, ang_Gparam, eta_arr, n_angs, ang_count_each)




        # ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, rad_cutoff, ang_cutoff, n_ele, eta_arr = CompileVec.compile_Gparam_vec(
        #                                         at_idx_map, ele_dict, Gparam_dict)
        #




        at_ref_idx = 3
        h = 0.001
        G_deriv_xyz_pile_vec = SymmDerivIndVecCython.symm_deriv_ind_pile_vec(
                                    distance_arr,
                                    at_ele_arr,
                                    rad_arr, ang_arr,
                                    ele_count, pair_count,
                                    drad_precalc,
                                    ddRij_G_precalc,
                                    ddRik_G_precalc,
                                    ddRjk_G_precalc,
                                    n_atoms, n_symm_func, n_ele, at_ref_idx,
                                    n_rads, n_angs,
                                    rad_count_each, ang_count_each)

        #print(G_deriv_xyz_pile_vec)


        Gfunc_deriv_pile = ForceNum.deriv_num_mol(glycine_test,  h, at_ref_idx, n_symm_func, 'x', Gparam_dict)



        temp_diff = G_deriv_xyz_pile_vec[:, 0] - Gfunc_deriv_pile
        print(temp_diff.shape)
        print(Gfunc_deriv_pile[0, 60*0:60*3+1])

        fraction = np.nan_to_num(temp_diff / Gfunc_deriv_pile)
        #print(fraction[0, 0:276+1])
        print(temp_diff[0, 60*0:60*3+1])
        print(np.max(temp_diff))
        print(np.max(fraction))







if __name__ == '__main__':
    unittest.main()
