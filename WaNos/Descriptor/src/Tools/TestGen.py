"""This module is designed for generating the test glycine molecule.


"""
from Utils.DirNav import get_project_dir
from ase.io import read, write
import os
import numpy as np


def get_test_glycine_mol():

    project_dir = get_project_dir()

    glycine_test = read(os.path.join(project_dir, 'Calculator', 'glycine.xyz'))
    print(glycine_test.get_chemical_symbols())
    return glycine_test

    


def get_test_Gparam_dict(at_idx_map):
    Rs_array = np.linspace(0.8,8,num= 24)   # based on max and min of the distances
    eta_array = 1./(2.0*np.square(0.2*Rs_array))
    rad_params = np.array([(Rs_array[i],eta_array[i]) for i in range(len(Rs_array)) ])
    # angular symmetry function parameters
    lambd_array = np.array([-1, 1])
    zeta_array = np.array([1, 4, 16])
    eta_ang_array = np.array([0.001, 0.01, 0.05])



    # Each of the element need to be parametrized for all of the list.
    # Automation_Point: Generate the angList and ang_comp dictionary quickly.
    angList = np.array(['HH', 'HC', 'HN', 'HO','CC','CO','CN','OO','ON','NN'])
    ang_comp = {'H':angList, 'O':angList, 'C':angList, 'N':angList}
    ang_params = np.array([[eta, zeta, lambd] for eta in eta_ang_array for zeta in zeta_array for lambd in lambd_array])
    # Parameter dictionary
    # Maintain: This part only works for the individual atoms
    #           Assumes that the configuration does not change over the time.
    Gparam_dict = {}
    for at_type in at_idx_map.keys():
        Gparam_dict[at_type] = {}
        Gparam_dict[at_type]['rad'] = {}
        for at2_rad in sorted(at_idx_map.keys()):
                Gparam_dict[at_type]['rad'][at2_rad] = rad_params

        # This Section is already designed to be general
        Gparam_dict[at_type]['ang'] = {}
        for at23_ang in ang_comp[at_type]:
            Gparam_dict[at_type]['ang'][at23_ang] = ang_params
    return Gparam_dict


if __name__ == '__main__':
    get_test_glycine_mol()
