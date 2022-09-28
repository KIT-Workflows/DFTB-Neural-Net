from cmath import tanh
from scipy.sparse import data
import yaml, tarfile

import os                                                             
import time   
import pickle
import math
import sys
import itertools
import numpy as np                                                     
import pandas as pd   
import seaborn as sns                                                  
import matplotlib.pyplot as plt   
import tarfile                                             

                                                                       
import ase                                                             
from ase import io
import ase.build                                                       
from ase import Atoms                                                  
from ase.atoms import Atoms                                            
from ase.io import read, write                                         
from ase.calculators.dftb import Dftb                                  
from ase.units import Hartree, mol, kcal, Bohr                         
                                                                       


my_tar = tarfile.open('src.tar')
my_tar.extractall() # specify which folder to extract to
my_tar.close()
print(os.getcwd())

from src.Calculator import src_nogrd                                                                                         
from src.Calculator.src_nogrd import xyzArr_generator       
from src.Calculator.src_nogrd import dictonary_elements
from src.Calculator.src_nogrd import at_idx_map_generator                                                        
                                                                                                                        
import pickle
from itertools import combinations_with_replacement as comb_replace
                                                                                                                 
from src.Utils.dftb_traj_io import read_scan_traj

def get_elements(struct):
    num=list(dict.fromkeys(struct.get_atomic_numbers()))
    num.sort()
    atomic_num=[f'{str(i)}' for i in num]
    ELEMENTS=[]
    for element in atomic_num:
        ELEMENTS.append(dictonary_elements[element])
    return ELEMENTS

if __name__ == '__main__':

    with open("rendered_wano.yml") as file:
        wano_file = yaml.full_load(file)

    geom_filename = wano_file['Geometries']
    CO_distance=wano_file['Cut-off distance']

    md_train_arr_origin = read_scan_traj(filename=sys.path[0]+'/'+ geom_filename)
    md_train_arr = md_train_arr_origin.copy(deep=False).reset_index(drop=True)


    nAtoms, xyzArr = xyzArr_generator(md_train_arr)# Calculate distance dataframe from xyz coordinates
    distances = src_nogrd.distances_from_xyz(xyzArr, nAtoms)

    SUPPORTED_ELEMENTS=get_elements(md_train_arr[0])
    
    at_idx_map = at_idx_map_generator(md_train_arr[0])
    

    # radial symmetry function parameters
    cutoff_rad = CO_distance
    Rs_array = np.linspace(min(distances.min()), CO_distance, num=int(max(distances.max())-min(distances.min())))   # based on max and min of the distances
    eta_array = (int(max(distances.max())-min(distances.min())))/(2*np.square(min(distances.min())*Rs_array))
    rad_params = np.array([(Rs_array[i], eta_array[i], cutoff_rad) for i in range(len(Rs_array)) ],dtype=object)


    # angular symmetry function parameters
    cutoff_ang = 5
    lambd_array = np.array([-1, 1])
    zeta_array = np.array([1,4,16])
    eta_ang_array = np.array([0.001, 0.01, 0.05])
        
    # Each of the element need to be parametrized for all of the list. 
    angList = np.array([e1+e2 for e1, e2 in comb_replace(SUPPORTED_ELEMENTS, 2)])
    ang_comp = {el : angList for el in SUPPORTED_ELEMENTS}
    ang_params = np.array([[eta, zeta, lambd, cutoff_ang] for eta in eta_ang_array for zeta in zeta_array for lambd in lambd_array])

    Gparam_dict = {}
    for at_type in at_idx_map.keys():
        Gparam_dict[at_type] = {}
        Gparam_dict[at_type]['rad'] = {}
        for at2_rad in SUPPORTED_ELEMENTS:
                Gparam_dict[at_type]['rad'][at2_rad] = rad_params

        Gparam_dict[at_type]['ang'] = {}
        for at23_ang in ang_comp[at_type]:
            Gparam_dict[at_type]['ang'][at23_ang] = ang_params     

    path = os.getcwd()
    rad_name = "symFunc_rad.param"
    ang_name = "symFunc_ang.param"
    sym_name = "symFunc_all.param"
    with open(os.path.join(path, rad_name), "w") as rad_f:
            rad_f.write(f"RadialCutoff = {str(cutoff_rad)} \n")
            rad_f.write("RadialParameters {\n") 
            for dist, eta in zip(Rs_array, eta_array):
                rad_f.write(f"{dist} {eta}\n")
            rad_f.write("}\n") 

    with open(os.path.join(path, ang_name), "w") as ang_f:
            ang_f.write(f"AngularCutoff = {str(cutoff_ang)} \n")
            ang_f.write("AngularParameters {\n") 
            for row in ang_params:
                ang_f.write(f"{row[0]} {row[1]} {row[2]}\n")
            ang_f.write("}\n") 

    with open(os.path.join(path, sym_name), "w") as all_f:
            all_f.write(f"RadialCutoff = {str(cutoff_rad)} \n")
            all_f.write("RadialParameters {\n") 
            for dist, eta in zip(Rs_array, eta_array):
                all_f.write(f"{dist} {eta}\n")
            all_f.write("}\n") 
            all_f.write(f"AngularCutoff = {str(cutoff_ang)} \n")
            all_f.write("AngularParameters {\n") 
            for row in ang_params:
                all_f.write(f"{row[0]} {row[1]} {row[2]}\n")
            all_f.write("}\n")

    with open(sym_name, "w") as all_f:
            all_f.write(f"RadialCutoff = {str(cutoff_rad)} \n")
            all_f.write("RadialParameters {\n") 
            for dist, eta in zip(Rs_array, eta_array):
                all_f.write(f"{dist} {eta}\n")
            all_f.write("}\n") 
            all_f.write(f"AngularCutoff = {str(cutoff_ang)} \n")
            all_f.write("AngularParameters {\n") 
            for row in ang_params:
                all_f.write(f"{row[0]} {row[1]} {row[2]}\n")
            all_f.write("}\n") 

    Gfunc_data = src_nogrd.symmetry_function(distances, at_idx_map, Gparam_dict)

    if wano_file['Include charges']:
        file_count=len(md_train_arr)
        Q=np.loadtxt(wano_file['Charges'])
        QQ=np.reshape(Q, (file_count,nAtoms))

        Gfunc_dict = Gfunc_data.to_dict()
        for at_type in Gfunc_dict:
            Gfunc_dict[at_type] = Gfunc_dict[at_type].to_dict()
        Dfunc_data = Gfunc_dict

        for at_type in Dfunc_data.keys():
            for at_idx in Dfunc_data[at_type]:
                Dfunc_data[at_type][at_idx] = Dfunc_data[at_type][at_idx].tolist()
                for z in range(file_count):
                    Dfunc_data[at_type][at_idx][z] = np.append(Dfunc_data[at_type][at_idx][z],QQ[z][at_idx])
                Dfunc_data[at_type][at_idx] = np.array(Dfunc_data[at_type][at_idx])
        with open('Gfunc.pkl', 'wb') as file:
            pickle.dump(Dfunc_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        Gfunc_data.to_pickle('Gfunc.pkl')
