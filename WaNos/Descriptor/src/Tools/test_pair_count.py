"""
##################################################

Module: test_pair_count

Description:
Test how the pair_count gives the correct results



##################################################
"""



import unittest
import Utils.DirNav
import os
from ase.io import read, write
import src_nogrd
import test_utils
import CompileArr
import Calculator.Calculation as Calculation

class TestPairCount(unittest.TestCase):

    def setUp(self):
        # Get the Glycine Molecule
        glycine_test, at_idx_map, Gparam_dict = test_utils.import_glycine_mol()
        self.test_mol = glycine_test
        self.at_idx_map = at_idx_map
        self.Gparam_dict = Gparam_dict

    def test_pair_count(self):
        # Print the Corresponding counts for the given pair
        # Criteria:
        # Pair Count will generate the same symm count for every pair in the list
        # Same Results as in Calculation.get_pair


        ang_list = []
        at_type_rand = next(iter(self.at_idx_map))

        pair_rand = at_type_rand + at_type_rand
        Gparam_ang = self.Gparam_dict[at_type_rand]['ang']


        for pair in Gparam_ang.keys():
            ang_list.append(pair)

        n_atoms = len(self.test_mol)


        at_ele_map, at_ele_arr, ele_dict = src_nogrd.at_ele_arr_generator(self.test_mol)
        ele_count, pair_count, count_Gparam, n_symm_func, n_ang_count, n_ele = CompileArr.compile_Gparam_const(
                                     self.at_idx_map, ele_dict, self.Gparam_dict)

        count_dict = CompileArr.compile_Gparam_dict(self.at_idx_map, ele_dict, self.Gparam_dict)

        # Generate the Pairs
        for at1 in range(n_atoms):
            at_ele_str_1 = at_ele_map[at1]
            at_ele_idx_1 = at_ele_arr[at1]

            for at2 in range(n_atoms):
                at_ele_str_2 = at_ele_map[at2]
                at_ele_idx_2 = at_ele_arr[at2]

                pair_str = Calculation.get_pair(at_ele_str_1, at_ele_str_2, ang_list)
                # Not Excatly Cython
                pair_idx_cython = CompileArr.get_pair_idx(at_ele_idx_1, at_ele_idx_2, n_ele)
                pair_start_cython = pair_count[pair_idx_cython][0]
                pair_end_cython   = pair_count[pair_idx_cython][1] -1

                #print(pair_start_cython, pair_end_cython)

                pair_iter_arr = count_dict[pair_str]
                pair_start_dict = pair_iter_arr[0]
                pair_end_dict   = pair_iter_arr[-1]

                #print(pair_start_dict, pair_end_dict)

                self.assertEqual(pair_start_cython, pair_start_dict)
                self.assertEqual(pair_end_cython, pair_end_dict)












if __name__ == "__main__":
    unittest.main()
