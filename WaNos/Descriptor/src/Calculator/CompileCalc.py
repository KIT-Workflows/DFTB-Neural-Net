"""
This Module is designed for the ase calculator based on the subnet
(The Compiled Version)
Also have other functions relavant to the packing of the input for
calculation in the sub-neural network

Able to support multi-CPU accerlation.

Multi-Processing Guide:
For MD, every next calculation must be based on the previous calculations.

Therefore, it is recommended to support the Multi-Processing.

Warning:
Compiled Force Calculation Not Implemented yet.

### Explaination
Because the Gfunc_data supported here is very different from the old Gfunc_data
(with elements), J. Zhu decided to change the entire module rather than just
create a child class.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import keras
import src.Calculator.SymmFuncIndCython as SymmFuncInd
import src.Calculator.CompileArr  as CompileArr
import src.Calculator.ModelIO as ModelIO
import src.Calculator.ForceInd  as ForceInd
from src.Calculator.NeuralCalc import NeuralCalc
from src.Calculator.src_nogrd import at_ele_arr_generator
import src.Calculator.src_nogrd as src_nogrd
from ase.calculators.calculator import Calculator, all_changes
from src.Utils.DirNav import get_project_dir
from src.Utils.DirNav import get_model_dir

class CompileCalc(NeuralCalc):
    """
    CompileCalc is a child class of NeuralCalc. SubNetCalc uses the elemental
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
                count_Gparam: Return the parameters for the given count in
                              symmetry function vectors.

                            a np.array that has the structure
                            count_Gparam[count]  = Gparam_list (specified for the given count)

                            count_Gparam = (count, max No. Params)


                count_dict: Return the count for the given atom
                            count_dict['element'] = [counts for that element]
                            count_dict['elemental pair'] = [counts for that elemental pair]

                n_atoms: number of atoms
                n_symm_func: number of symmetry functions

        Other calculations during the compile mode (not in compile function)
                neighbourlist_arr, neighbourpair_arr:
                            For the neighbour list
                distance_arr: distances for the calculation.



        Explaination:
        (Since symmetry function has a lot of for loops, which slows down the
        numerical calculation. For the molecule dynamics, only the neighbour list
        will be updated. The symmetry function vector's indices should not change.
        Therefore, it is necessary to get all the iterators before running
        individual calculations. It is found that it reduces the number of
        for loops from 7 to 3.

        TODO:
        Solve the sequence of at_idx_map and Gparam_dict


        """

        at_idx_map = src_nogrd.at_idx_map_generator(atoms)

        at_ele_map, at_ele_arr, ele_dict = at_ele_arr_generator(atoms)
        ele_count, pair_count, count_Gparam, n_symm_func, n_ang_count, n_ele = CompileArr.compile_Gparam_const(
                                                at_idx_map, ele_dict, self.Gparam_dict)

        self.at_ele_map     = at_ele_map
        self.at_ele_arr     = at_ele_arr
        #self.count_dict = count_dict
        self.ele_count      = ele_count
        self.pair_count     = pair_count
        self.count_Gparam   = count_Gparam
        self.n_ang_count    = n_ang_count
        self.n_ele          = n_ele




        ##import pdb; pdb.set_trace()


    def initialize(self, atoms):
        """
        Calculate the energy of the original calculator

        Must compile before initialize!
        """

        self.n_atoms = len(atoms)

        no_Gparam_dict = self.Gparam_dict == None
        not_imported = no_Gparam_dict

        if not_imported:
            raise ImportError("NeualCalc::initialize: the model has not been imported")

        if self.calc == None:
            raise ImportError("NeuralCalc:initiaalize: No Calculator is found ")

        #self.compile(self.at_idx_map, self.Gparam_dict)

        # Calculate the energy (lower-level calculator)

        atoms_cp = atoms.copy()
        atoms_cp.set_calculator(self.calc)
        self.dE_ref = self.e_ref_dft - self.e_ref_calc

        #print("Calculation Files Saved in folder ", os.getcwd())
        #start = time.time()
        calc_e = atoms_cp.get_potential_energy()
        calc_charge = atoms_cp.get_charges()
        calc_force = atoms_cp.get_forces()
        self.calc_properties['energy'] = calc_e
        self.calc_properties['charge'] = calc_charge
        self.calc_properties['force'] = calc_force

        #end = time.time()
        #self.calc_time += end - start

        #import pdb; pdb.set_trace()
        # Get the Symmetry Function
        Gfunc_data, xyzArr, distance_arr, neighbourlist_arr, neighbourlist_count, neighbourpair_arr, neighbourpair_count, ang_precalc= SymmFuncInd.symm_func_mol_compiled(
                                    self.atoms, self.at_ele_arr,
                                    self.count_Gparam, self.ele_count, self.pair_count,
                                    self.n_atoms, self.n_symm_func, self.n_ang_count, self.n_ele)
        self.Gfunc_data = Gfunc_data
        self.distance_arr = distance_arr
        self.neighbourlist_arr = neighbourlist_arr
        self.neighbourlist_count = neighbourlist_count
        self.neighbourpair_arr = neighbourpair_arr
        self.neighbourpair_count = neighbourpair_count
        self.xyzArr = xyzArr
        #self.dG_all_dict = dG_all_dict
        self.ang_precalc = ang_precalc


    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)


        self.initialize(self.atoms)
        #TODO: Implement the energy calculation in a pure function
        #TODO: Replace Pandas series with numpy array to speed up.


        denergy = calculate_subnet_energy_mol_compiled(self.subnet_list, self.Gfunc_data,
                                            self.at_ele_map, self.n_atoms)

        energy = self.calc_properties['energy'] + denergy  + self.dE_ref # DFTB + NN Correction

        # dforce = ForceThread.calculate_analytical_force_mol_multicore(self.subnet_deriv_arr, self.Gfunc_data,
        #                                         self.atoms, self.at_idx_map,
        #                                         self.Gparam_dict, self.n_atoms)
        dforce = ForceInd.calculate_analytical_force_mol_compiled(self.subnet_deriv_arr,
                                    self.Gfunc_data, self.distance_arr,
                                    self.at_ele_map, self.at_ele_arr,
                                    self.xyzArr,
                                    self.neighbourlist_arr,  self.neighbourlist_count,
                                    self.neighbourpair_arr,  self.neighbourpair_count,
                                    self.count_Gparam, self.ele_count, self.pair_count,
                                    self.ang_precalc,
                                    self.n_atoms, self.n_symm_func, self.n_ele)
        force = self.calc_properties['force'] + dforce

        #force = self.calc_properties['force'] += dforce
        stress = np.zeros(6)

        self.results['energy'] = energy
        self.results['forces']  = force
        self.results['stress'] = stress
        self.results['dforces'] = dforce
        self.results['denergy'] = denergy


    def import_model(self, model_name, subnet_str_list):
        """Assumes all models are saved in the "src.models/" directory,
        import all the elemental sub-neural network.

                Args:
                    model_name: the folder name that contains the model
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

        save_dir = get_model_dir(model_name)
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
        self.subnet_deriv_arr = ForceInd.get_subnet_deriv_arr_tf(self.subnet_list, self.n_symm_func)
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

