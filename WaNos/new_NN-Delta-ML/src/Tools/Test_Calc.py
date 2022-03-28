import os
import numpy as np
import pandas as pd
import keras
import Utils.traj
import Utils.DirNav
import matplotlib.pyplot as plt
from Utils.traj import get_pe_arr
from Utils.DirNav import get_project_dir
import unittest

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ASEDFTBMod import DftbMod
from Calculator.ModelCalc import ModelCalc
from Calculator.CompileVecCalc import CompileVecCalc

def get_calc_result(calc, lib_name, mol_name):
    #project_dir = Utils.DirNav.get_project_dir()
    project_dir ='/home/qv0/Dropbox/projects/codes/dftb-nn'
    model_path = os.path.join(project_dir, 'models', lib_name)
    #glycine_test, at_idx_map, Gparam_dict = test_utils.import_glycine_mol(mol_name)
    molecule = read(mol_name)

    if isinstance(calc, ModelCalc):
        calc.import_model(model_path, 'model.h5', 'at_idx_map.pkl', 'Gparam_dict.pkl')
    else:
        calc.import_model(model_path,['H', 'O', 'N', 'C'], 'Gparam_dict.pkl')
    molecule.set_calculator(calc)
    if calc.need_compile:
        calc.compile(molecule)
    molecule.get_potential_energy()

    denergy = molecule._calc.get_denergy()
    dforce_arr = molecule._calc.get_dforces()

    return denergy, dforce_arr


def criteria(energy_1, force_1, energy_2, force_2):
    energy_diff = abs(energy_1- energy_2)
    energy_equ = abs(energy_1 - energy_2)  < 0.001

    force_diff = force_1 - force_2
    force_equ  = np.amax(force_diff) < 0.001
    #np.all(force_diff < 0.0001)
    print("Energy Diff: ", energy_diff, 'eV')

    print("Force Diff: ", force_diff, 'eV/angstrom')

    return energy_equ, force_equ

class TestCalc(unittest.TestCase):

    def setUP(self):
        """Initialize all the calculators, molecules,
        import
        """

    def test_lib(self):
        os.environ['DFTB_PREFIX'] = '/home/qv0/Apps/slako/mio/mio-1-1/'
        os.environ['DFTB_COMMAND'] = '/home/qv0/Apps/dftbplus/dftbplus-18.1.x86_64-linux/bin/dftb+'

        dftb_calc = DftbMod(
                             Hamiltonian_MaxAngularMomentum_='',
                             Hamiltonian_MaxAngularMomentum_O='"p"',
                             Hamiltonian_MaxAngularMomentum_H='"s"',
                             Hamiltonian_MaxAngularMomentum_C='"p"',
                             Hamiltonian_MaxAngularMomentum_N='"p"',
                             )


        model_calc  = ModelCalc(dftb_calc)

        compile_calc = CompileCalc(dftb_calc)

        compilevec_calc = CompileVecCalc(dftb_calc)

        calc_list = [compile_calc, compilevec_calc]

        lib_list = ['glycine_50samples_300K_models']
        mol_list = ['glycine_origin.xyz']

        for lib in lib_list:
            for mol in mol_list:
                compile_e, compile_force = get_calc_result(model_calc, lib, mol)

        for lib in lib_list:
            for mol in mol_list:
                vec_e, vec_force   = get_calc_result(compile_calc, lib, mol)
                equal_energy, equal_force = criteria(compile_e, compile_force, vec_e, vec_force )
                self.assertTrue(equal_energy)
                self.assertTrue(equal_force)

        for lib in lib_list:
            for mol in mol_list:
                compile_e, compile_force = get_calc_result(model_calc, lib, mol)
                vec_e, vec_force   = get_calc_result(compilevec_calc, lib, mol)
                equal_energy, equal_force = criteria(compile_e, compile_force, vec_e, vec_force )
                self.assertTrue(equal_energy)
                self.assertTrue(equal_force)

        return

if __name__ == '__main__':
    unittest.main()
