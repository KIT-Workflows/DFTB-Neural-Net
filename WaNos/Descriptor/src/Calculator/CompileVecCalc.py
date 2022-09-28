"""
##################################################

Module: CompileVecCalc

Brief:
Vectorized Version of the CompileCalc (CompileCalc.py)

Explaination:
Because the Gfunc_data supported here is very different from the old Gfunc_data
(with elements), J. Zhu decided to change the entire module rather than just
create a child class.

##################################################
"""


import os
import sys
import time

import numpy as np
import pandas as pd
import keras


import src.Calculator.SymmFuncIndCython as SymmFuncInd
import src.Calculator.CompileVec as CompileVec
import src.Calculator.ModelIO  as ModelIO
import src.Calculator.ForceIndVec as ForceIndVec
from src.Calculator.NeuralCalc import NeuralCalc
from src.Calculator.src_nogrd import at_ele_arr_generator
import src.Calculator.src_nogrd as src_nogrd
from ase.calculators.calculator import Calculator, all_changes
import src.Calculator.PreCalcVec as PreCalcVec

import src.Calculator.Calculation as Calculation
from src.Utils.DirNav import get_project_dir
from src.Utils.DirNav import get_model_dir

import src.Calculator.SymmFuncIndVecCython as SymmFuncIndVecCython

class CompileVecCalc(NeuralCalc):
    """
    CompileVecCalc is a child class of NeuralCalc. SubNetCalc uses the elemental
    sub-nerual network for the training instead of the entire model.
    CompileCalc is fundamentally the same with SubNetCalc for the calculation,
    but pre-compile the iterator so to speed up the for loop for the MD simulation
    iteration on the same molecule.

    SubNetCalc Read and Write the sub-neural network from its weights and
    the model .json file.

    As discussed, for modularity, it is recommended to implement the training
    process for the model in another module than writing everything in this
    calculator. This is only designed as the ase calculator rather than
    then integrated model trainer.


    Add support for the CPU-Multi Processing Ability.

    Compile Mode: See the docs for compile function.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'denergy', 'dforces']

    def __init__(self, calc, **kwargs):
        Calculator.__init__(self, **kwargs)
        """
            Args:
                calc: the ase.calculator used for the calculation.
                        Default: a DFTB calculator. Should be consistent
                        with the calculator used for training.

        """

        self.subnet_list = None # To store the sub-neural network
        self.Gparam_dict = None
        self.at_idx_map = None
        self.calc = calc
        self.calc_properties = {} # To store properties by the calculator
        self.e_ref_calc = 0     # Reference Point for energy of given calculator
        self.e_ref_dft = 0     # Reference Point for DFT
        self.n_symm_func = 0
        self.subnet_deriv_arr = None # To store the derivative of subnet

        self.calc_time = 0 # To record the time used for the calculator to get the energy.
        self.need_compile = True

    def compile(self, atoms):
        """For the compiled mode, it will compile the iterators for the given
        molecule before running Molecular dynamics simulation.


        Will Generate:

                rad_Gparam: return the paramters (eta) for each individual vector
                            on the radial symmetry function
                ang_Gparam: return the parameters (eta, zeta, labmd) for each
                            individual vector on the angular vector.

                            rad_Gparam[rad_count] = (eta)
                            ang_Gparam[ang_count] = (eta, zeta, lambd)

                n_atoms: number of atoms
                n_symm_func: number of symmetry functions

        Other calculations during the compile mode (not in compile function)
                neighbourlist_arr, neighbourpair_arr:
                            For the neighbour list
                distance_arr: distances for the calculation.



        Explaination:
        (Please Refer to the CompileCalc Docs.)

        General Idea:
        0. Everything is numpy array, to speed up the calculation. (Using np.math)
           for linear algebra computation.
        1. Store repeated computation in memory to reduce the number of repeated
           computation.

        TODO:
        Solve the sequence of at_idx_map and Gparam_dict


        """

        at_ele_map, at_ele_arr, ele_dict = at_ele_arr_generator(atoms)
        self.at_idx_map = src_nogrd.at_idx_map_generator(atoms)

        print(self.at_idx_map)

        n_atoms = len(atoms)


        ele_count, pair_count, rad_Gparam, ang_Gparam, n_symm_func, rad_count_each,  ang_count_each, rad_cutoff, ang_cutoff, n_ele, eta_arr = CompileVec.compile_Gparam_vec(
                                                self.at_idx_map, ele_dict, self.Gparam_dict)

        self.n_atoms = n_atoms
        self.at_ele_map     = at_ele_map
        self.at_ele_arr     = at_ele_arr
        #self.count_dict = count_dict
        self.ele_count      = ele_count
        self.pair_count     = pair_count
        self.rad_Gparam     = rad_Gparam
        self.ang_Gparam     = ang_Gparam
        self.rad_count_each = rad_count_each
        self.ang_count_each = ang_count_each
        self.n_ele          = n_ele
        self.eta_arr        = eta_arr

        self.rad_cutoff     =  rad_cutoff
        self.ang_cutoff     =  ang_cutoff

        # self.n_rads =  int(n_atoms * (n_atoms-1) /2)
        # self.n_angs  = int((n_atoms * (n_atoms-1) * (n_atoms - 2) / 2))
        #

        ##import pdb; pdb.set_trace()


    def initialize(self, atoms):
        """
        Calculate the energy of the original calculator

        Must compile before initialize!
        """

        self.n_atoms = len(atoms)

        no_at_idx_map = self.at_idx_map == None
        no_Gparam_dict = self.Gparam_dict == None
        not_imported = no_at_idx_map or no_Gparam_dict

        if not_imported:
            raise ImportError("NeualCalc::initialize: the model has not been imported")

        if self.calc == None:
            raise ImportError("NeuralCalc:initiaalize: No Calculator is found ")

        #self.compile(self.at_idx_map, self.Gparam_dict)

        # Calculate the energy (lower-level calculator)

        atoms_cp = atoms.copy()
        atoms_cp.set_calculator(self.calc)
        self.dE_ref = self.e_ref_dft - self.e_ref_calc

        dftb_start = time.time()

        calc_e = atoms_cp.get_potential_energy()
        calc_charge = atoms_cp.get_charges()
        calc_force = atoms_cp.get_forces()
        self.calc_properties['energy'] = calc_e
        self.calc_properties['charge'] = calc_charge
        self.calc_properties['force'] = calc_force

        dftb_end = time.time()

       #print("DFTB Time:", dftb_end - dftb_start)

        md_samplesArr = pd.Series([atoms])
        n_atoms, xyz_arr = src_nogrd.xyzArr_generator(md_samplesArr)

        self.distance_arr = SymmFuncInd.distance_xyz_mol(xyz_arr, n_atoms).astype(np.float64)


        self.rad_arr, self.ang_arr, self.n_rads, self.n_angs = Calculation.get_neighbour_cutoff(xyz_arr, self.rad_cutoff, self.ang_cutoff)

        Rij_rad = SymmFuncIndVecCython.get_distance_tensor_rad(self.distance_arr, self.rad_arr, self.n_atoms, self.n_rads)
        Rij_vec, Rik_vec, Rjk_vec = SymmFuncIndVecCython.get_distance_tensor_ang(self.distance_arr, self.ang_arr, self.n_atoms, self.n_angs)
        self.rad_precalc, self.drad_precalc = PreCalcVec.symm_rad_ind_precalc(Rij_rad, self.rad_Gparam, self.n_rads, self.rad_count_each)
        self.ang_precalc, self.ddRij_G_precalc, self.ddRik_G_precalc, self.ddRjk_G_precalc= PreCalcVec.symm_ang_ind_precalc(Rij_vec, Rik_vec, Rjk_vec, self.ang_Gparam, self.eta_arr, self.n_angs, self.ang_count_each)

        # Release the Memory of Temp Variables
        Rij_rad = None
        Rij_vec = None
        Rik_vec = None
        Rjk_vec = None


        #print(self.rad_precalc.shape)

        self.Gfunc_data = SymmFuncIndVecCython.symm_func_mol_vecidx(self.at_ele_arr,
                                    self.rad_arr, self.ang_arr,
                                    self.ele_count, self.pair_count,
                                    self.rad_precalc, self.ang_precalc,
                                    self.n_atoms, self.n_symm_func, self.n_ele,
                                    self.n_rads, self.n_angs,
                                    self.rad_count_each, self.ang_count_each)


    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)


        #TODO: Implement the symm_func_ind function and compile it
        # by numba or cython
        self.initialize(self.atoms)
        #TODO: Implement the energy calculation in a pure function
        #TODO: Replace Pandas series with numpy array to speed up.

        denergy = calculate_subnet_energy_mol_compiled(self.subnet_list, self.Gfunc_data,
                                            self.at_ele_map, self.n_atoms)

        energy = self.calc_properties['energy'] + denergy  + self.dE_ref # DFTB + NN Correction

        # dforce = ForceThread.calculate_analytical_force_mol_multicore(self.subnet_deriv_arr, self.Gfunc_data,
        #                                         self.atoms, self.at_idx_map,
        #                                         self.Gparam_dict, self.n_atoms)
        dforce = ForceIndVec.calculate_analytical_force_mol_compiled(self.subnet_deriv_arr,
                                    self.Gfunc_data, self.distance_arr,
                                    self.at_ele_map, self.at_ele_arr,
                                    self.rad_arr, self.ang_arr,
                                    self.ele_count, self.pair_count,
                                    self.drad_precalc,
                                    self.ddRij_G_precalc,
                                    self.ddRik_G_precalc,
                                    self.ddRjk_G_precalc,
                                    self.n_atoms, self.n_symm_func, self.n_ele,
                                    self.n_rads, self.n_angs,
                                    self.rad_count_each, self.ang_count_each)
        print("force delta correction --- start")
        print(dforce)
        print("force delta correction --- end")
        print("dftb forces --- start")
        print(self.calc_properties['force'])
        print("dftb forces --- end")

        force = self.calc_properties['force'] + dforce

        #force = self.calc_properties['force'] += dforce
        stress = np.zeros(6)

        self.results['energy'] = energy
        self.results['forces']  = force
        self.results['stress'] = stress
        self.results['dforces'] = dforce
        self.results['denergy'] = denergy





    def import_model(self, save_dir, subnet_str_list):
        """Assumes all models are saved in the "src.models/" directory,
        import all the elemental sub-neural network.

                Args:
                    save_dir: the folder name that contains the model
                    subnet_str_list: list of subnet str in the form
                            ['H', 'O', 'C', 'N']

                Effect:
                    Load the entire Keras model.

        Comments:
        Junmian Zhu's experiments on the iPython notebook indicates that if directly
        load the keras.model.h5, and then get the sub-neural network, it would
        lose the name of each subnet. Therefore, it is recommended to use the
        load the weights and the model.json file for each sub neural network.
        """

        try:
            self.subnet_list = pack_subnet_list(save_dir, subnet_str_list)
        except:
            ImportError("SubNetCalc::import_model failed")

        Gparam_file = 'Gparam_dict.pkl'

        try:
            self.Gparam_dict = ModelIO.read_data_from_pickle(save_dir, Gparam_file)
        except:
            raise ImportError("ModelCalc::import_model unable to read Gparam_dict and at_idx_map")

        try:
            rel_e_arr = pd.read_pickle(os.path.join(save_dir, "rel_e_ref.pkl"))
        except:
            raise ImportError("ModelCalc::import_model unable to read reference energy")
        # Test whether the model has been successfully imported
        if self.subnet_list != None:
            print("Successfully Imported the training model")
        else:
            print("Failed to import the training model")
        self.e_ref_dft = rel_e_arr['dft']
        self.e_ref_calc = rel_e_arr['calc']

        # Get the number of symmetry function by the input shape
        # Input_shape = [n_samples, n_symm_func]
        #TODO: Assumes that each subnet has the same No. symm function
        any_ele = subnet_str_list[0]
        self.n_symm_func = self.subnet_list[any_ele].layers[0].input_shape[1]

        # Also get the neural network derivative

        # WARNING: Currently Using Theano Backend
        self.subnet_deriv_arr = ForceIndVec.get_subnet_deriv_arr_tf(self.subnet_list, self.n_symm_func)
    #



    """
    ##########

    Methods for Calculation.

    ##########
    """
    def get_dforces(self, atoms=None):
        "Get the Delta Force"
        return self.get_property('dforces', atoms)

    def get_denergy(self, atoms=None):
        return self.get_property('denergy', atoms)

def pack_subnet_list(save_dir, subnet_str_list):
    """Read an list of sub-neural networks based on the list from the directory

            Args:
                save_dir: directory with the subnet weights and .json model
                subnet_str_list: list of subnet str in the form
                        ['H', 'O', 'C', 'N']

    Comments:
    Will Load from subnet weights and the .json file
    """
    subnet_list = {}
    for subnet_str in subnet_str_list:
        try:
            subnet_name = subnet_str + '-subnet'
            subnet_list[subnet_str] = ModelIO.read_keras_subnet_ind(save_dir, subnet_name)
        except:
            raise ImportError("pack_subnet_list: error while import subnet " + subnet_name)
    return subnet_list








"""
##########

Methods for Evaluating Symmetry Function Using the Subnet

##########
"""


def calculate_subnet_energy_mol_compiled(subnet_list, Gfunc_data, at_ele_map, n_atoms):
    """Calculate the total energy using the list of sub-nerual network.

            Args:
                subnet_list: the list of elemental subnet
                    subnet_list['h'] = H_subnet
                Gfunc_data: (New) symmetry function values for one molecule
                at_ele_map: atom element map
                            return the element for the given atom index
                n_atom: (Just for performance, not necessary for processing)
                        number of atoms

            Outputs:
                total_energy: total energy for a given molecule calculated
                              by the neural network


    Currently, This function is only designed for the symmetry function
    without feature. For Feature suppot, please refer to :
    calculate_subnet_energy_total function
    """
    atom_e = np.zeros(n_atoms)

    for at_idx in np.arange(n_atoms):
        Gfunc_ind = Gfunc_data[at_idx]
        element = at_ele_map[at_idx]
        atom_e[at_idx] = subnet_list[element].predict(Gfunc_ind)

    return np.sum(atom_e)



def calculate_subnet_energy_total(subnet_arr, inp):
    """Calculating the total energy based on the input. (Thus support
    Inputting the feature function)

            Args:
                subnet_arr: similar structure with the defined neural
                network model.  .
                    subnet_arr['element']['h'] = H_subnet
                    subnet_arr['rc'] = Feat_subnet

                inp: input for the neural network

            Outputs:
                total_energy: total energy for the entire neural network.

    Not Implemented Yet
    """
    pass
