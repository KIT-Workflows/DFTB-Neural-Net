from scipy.sparse import data
import yaml, tarfile

import os                                                             
import time   
import pickle
import math
import sys
import numpy as np                                                     
import pandas as pd   
import seaborn as sns                                                  
import matplotlib.pyplot as plt                                        
                                                                       
from sklearn.metrics import mean_absolute_error                        
from sklearn.metrics import mean_squared_error                         
from sklearn.model_selection import train_test_split                   
from sklearn.preprocessing import StandardScaler                       
from sklearn.model_selection import train_test_split                   
                                                                       
import tensorflow.keras                                                           
from tensorflow.keras.optimizers import Adam                                      
from tensorflow.keras.models import Model, Sequential                             
from tensorflow.keras.layers import Input, Dense, Dropout         
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping   
                                                                       
import ase                                                             
from ase import io
import ase.build                                                       
from ase import Atoms                                                  
from ase.atoms import Atoms                                            
from ase.io import read, write                                         
from ase.calculators.dftb import Dftb                                  
from ase.units import Hartree, mol, kcal, Bohr                         
                                                                       


my_tar = tarfile.open('src.tar.xz')
my_tar.extractall() # specify which folder to extract to
my_tar.close()
os.mkdir('NN-parameter') 
print(os.getcwd())

from src.Calculator import src_nogrd                                                       
from src.Calculator.src_nogrd import sym_func_show                                    
from src.Calculator.src_nogrd import xyzArr_generator                                 
from src.Calculator.src_nogrd import feat_scaling_func                                
from src.Calculator.src_nogrd import at_idx_map_generator
from src.Calculator.src_nogrd import at_idx_map_generator_old
from src.Calculator.store_models import write_subnet_text
                                                                                                                        
import pickle
from itertools import combinations_with_replacement as comb_replace
                                                                    
import src.Utils.DirNav                                                
from src.Utils.dftb_traj_io import read_scan_traj
import src.Utils.netBuilder
from src.Utils.netBuilder import netBuilder
from src.Utils.netTrainer import netTrainer

with open("rendered_wano.yml") as file:
    wano_file = yaml.full_load(file)

geom_filename = wano_file['Geometries']

struct = io.read(geom_filename, format = "xyz")
elem = struct.get_chemical_symbols()
SUPPORTED_ELEMENTS = list(dict.fromkeys(elem))


data_energy = wano_file['Energies'] 
data_set = pd.read_csv(data_energy)


with open(wano_file['Ref-Energy-DFTB']) as file:
    dftb_file = yaml.full_load(file)
ref_dftb = dftb_file['dftb_energy']


with open(wano_file['Ref-Energy-DFT']) as file:
    dft_file = yaml.full_load(file)
ref_dft = dft_file['orca_energy']

data_set['Delta'] = data_set['dft'] - data_set['dftb'] + ref_dftb - ref_dft

    
md_train_arr_origin = read_scan_traj(filename=geom_filename)


md_train_arr = md_train_arr_origin.copy(deep=False).reset_index(drop=True)
md_rel_energy_arr = data_set['Delta']
# Get rid of error values.
nan_index = np.where(np.isnan(md_rel_energy_arr))

for idx in nan_index:
    md_train_arr.drop(idx)
    md_rel_energy_arr = md_rel_energy_arr[~np.isnan(md_rel_energy_arr)]

# Maintainence: Natom is a global variable 
# Assumes that the configuration does not change the Number of atoms. 
nAtoms, xyzArr = xyzArr_generator(md_train_arr)

# Calculate distance dataframe from xyz coordinates
distances = src_nogrd.distances_from_xyz(xyzArr, nAtoms)

at_idx_map_naive = at_idx_map_generator_old(md_train_arr[0])
## Hotfix for atom ordering without touching at_idx_map_generator
at_idx_map_old = {el : at_idx_map_naive[el] for el in SUPPORTED_ELEMENTS}

at_idx_map = at_idx_map_generator(md_train_arr[0])

lst=list(range(0, nAtoms))
def find_tuples(lst, key, num=3):
    return [i for i in itertools.permutations(lst, num) if sum(i)==key]
angles=find_tuples(lst,nAtoms,3)

all_ang=[]
for j in range(0,len(md_train_arr)):
    for i in range (0,nAtoms):
        all_ang.append(md_train_arr[j].get_angle(*angles[i]))

# radial symmetry function parameters
cutoff_rad = max(distances.max(1))
Rs_array = np.linspace(distances.min(1), max(distances.max(1)), num=int(max(distances.max(1))-min(distances.min(1))))   # based on max and min of the distances
eta_array = (int(max(distances.max(1))-min(distances.min(1))))/(2*np.square(min(distances.min(1))*Rs_array))
rad_params = np.array([(Rs_array[i], eta_array[i], cutoff_rad) for i in range(len(Rs_array)) ])


# angular symmetry function parameters
cutoff_ang = math.radians(max(all_ang))
lambd_array = np.array([-1, 1])
#zeta_array = np.array([1, 4, 16])
zeta_array = np.array([math.radians(sum(all_ang)/len(all_ang))])
#eta_ang_array = np.array([0.001, 0.01, 0.05])
eta_ang_array = np.array([math.radians(min(all_ang))])
    
# Each of the element need to be parametrized for all of the list. 
angList = np.array([e1+e2 for e1, e2 in comb_replace(SUPPORTED_ELEMENTS, 2)])
print(angList)
ang_comp = {el : angList for el in SUPPORTED_ELEMENTS}
ang_params = np.array([[eta, zeta, lambd, cutoff_ang] for eta in eta_ang_array for zeta in zeta_array for lambd in lambd_array])

Gparam_dict = {}
for at_type in at_idx_map.keys():
    Gparam_dict[at_type] = {}
    Gparam_dict[at_type]['rad'] = {}
    for at2_rad in at_idx_map.keys():
        Gparam_dict[at_type]['rad'][at2_rad] = rad_params

# This Section is already designed to be general
    Gparam_dict[at_type]['ang'] = {}
    for at23_ang in ang_comp[at_type]:
        Gparam_dict[at_type]['ang'][at23_ang] = ang_params
for at_type in Gparam_dict.keys():
    print(Gparam_dict[at_type]['rad'].keys())
        

print(rad_params)
print(ang_params)

path = "NN-parameter"
rad_name = "symFunc_rad.param"
ang_name = "symFunc_ang.param"
with open(os.path.join(path, rad_name), "w") as rad_f:
    rad_f.write(str(cutoff_rad)+"\n")
    for dist, eta in zip(Rs_array, eta_array):
        rad_f.write(f"{dist} {eta}\n")

with open(os.path.join(path, ang_name), "w") as ang_f:
    ang_f.write(str(cutoff_ang)+"\n")
    for row in ang_params:
        ang_f.write(f"{row[0]} {row[1]} {row[2]}\n")

Gfunc_data = src_nogrd.symmetry_function(distances, at_idx_map, Gparam_dict)

n_symm_func =Gfunc_data[SUPPORTED_ELEMENTS[0]][0][0].shape[0]
builder = netBuilder(SUPPORTED_ELEMENTS, n_symm_func)
subnets = builder.build_subnets(n_dense_layers=2, n_units=15, 
                    hidden_activation='tanh',
                    dropout_type="NoFirstDrop", dropout_ratio=0.015)
model = builder.build_molecular_net(at_idx_map, subnets)
print(model.summary())

def idx_generator(n_samples, val_ratio, test_ratio):
    """
    Function:
    Randomly shuffle the indexes and to generate indexes for the training, validation and test set.
    
        Args:
            n_samples: number of samples, an interger
            val_ratio: ratio of the validation set (compared with all data set)
            test_ratio: 
    
        Warning: 0 < val_ratio + test_ratio < 1.
    
        Output:
            train_idx: indexes for training set
            val_idx: indexes for the validation set
            test_idx: indexes for the test set    
    """
    if val_ratio + test_ratio >= 1 or val_ratio + test_ratio <= 0:
        raise  ValueError("idx_generator: the val_ratio and test_ratio must be in between 0 and 1")
    
    shuffled_indices = np.random.permutation(n_samples)
    
    
    val_set_size = int(n_samples * val_ratio)
    val_idx  = shuffled_indices[:val_set_size]
    
    test_set_size= int(n_samples * val_ratio)
    test_idx = shuffled_indices[val_set_size:val_set_size+test_set_size]
    
    train_idx = shuffled_indices[val_set_size + test_set_size:]
    
    return train_idx, val_idx, test_idx
    
## Split the Training, Validation & Test Data 
n_samples = len(md_train_arr)
train_idx, val_idx, test_idx = idx_generator(n_samples, 0.1,0.1)
print(train_idx.shape)
print(val_idx.shape)
# Check whether it is totally splitted
if train_idx.shape[0] + test_idx.shape[0] + val_idx.shape[0] != n_samples:
    raise ValueError("Splitting Test does not equal to the entire set!")
    


# rescale target values 
# All training values in kcal/mol unit

y_train = md_rel_energy_arr[train_idx] 
y_val   = md_rel_energy_arr[val_idx]
y_test  = md_rel_energy_arr[test_idx]

print('y_train min, max = ', '%.5f  %.5f' %(y_train.min(), y_train.max() ))
print('y_test min, max = ', '%.5f  %.5f' %(y_test.min(), y_test.max()) )

def split_training_data(Feat_data, at_idx_map, train_idx, val_idx, test_idx):
    """
    Function:
    Split the training set, 
        
    Input:
    Feat_data_train: Strucutre for the feat data Feat_data_train['element'][atom]
    at_idx_map: Atom Index Map
    train_idx: the indices used for the training 
    
    
    Output:
    Return the Feat_train, Feat_val and Feat_test set in the shape
    Feat_scaler['element'][atom][Feature Number]    
    """
    
    
    Feat_train_scaled = {}
    Feat_val_scaled = {}
    Feat_test_scaled = {}

    
    for at_type in at_idx_map.keys():
        Feat_train_scaled[at_type] = {}
        Feat_val_scaled[at_type] = {}
        Feat_test_scaled[at_type] = {}
        
        for at in at_idx_map[at_type]:
            Feat_train_scaled[at_type][at] = Feat_data[at_type][at][train_idx,]
            #import pdb; pdb.set_trace()
            Feat_val_scaled[at_type][at]   = Feat_data[at_type][at][val_idx,]
            Feat_test_scaled[at_type][at]  = Feat_data[at_type][at][test_idx,]
            

            
    
    return Feat_train_scaled, Feat_val_scaled, Feat_test_scaled
    


train_scaled, val_scaled, test_scaled = split_training_data(Gfunc_data, at_idx_map, train_idx, val_idx, test_idx)
print(test_scaled[SUPPORTED_ELEMENTS[0]][0].shape)
#Feat_train_scaled, Feat_val_scaled, Feat_test_scaled = split_training_data(Feat_data, at_idx_map, train_idx, val_idx, test_idx)
#print(Feat_train_scaled['H'][4].shape)

inp_train = []
inp_val   = []
inp_test  = []
for at_type in at_idx_map.keys():

    for atA in at_idx_map[at_type]:
        inp_train.append(train_scaled[at_type][atA])
        #inp_train.append(Feat_train_scaled[at_type][atA])
        
        inp_val.append(val_scaled[at_type][atA])
        #inp_val.append(Feat_val_scaled[at_type][atA])
        
        inp_test.append(test_scaled[at_type][atA])
        #inp_test.append(Feat_test_scaled[at_type][atA])
        
def get_inp(at_idx_map, Gfunc_scaled, Feat_scaled):
    inp_arr = []
    for at_type in at_idx_map.keys():
        for at_idx in at_idx_map[at_type]:
            inp_arr.append(Gfunc_scaled[at_type][at_idx])
            inp_arr.append(Feat_scaled[at_type][at_idx])
    
    return pd.Series(inp_arr)
    
os.mkdir('model_2B_nUnit')
model_folder = "model_2B_nUnit"
#!mkdir $model_folder
trainer = netTrainer(model, verbose=1, folder=model_folder)
nUnit = 15

check1 = model_folder +'/' + str(nUnit) + '.hdf5'
checkpointer = ModelCheckpoint(filepath=check1, verbose=0,  monitor='val_mean_squared_error',\
                            mode='min', save_best_only=True)
earlystop = EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=100, verbose=0)

#Mode == 'NoFirstDrop':

def repeat(times, f):
        for i in range(times): f()
def session():
    model.compile(loss='mean_squared_error',
                    optimizer=Adam(lr=0.01,decay=0.001),
                    metrics=['mean_squared_error', 'mean_absolute_error'])
        
    history = model.fit(inp_train, y_train, \
                        callbacks=[checkpointer, earlystop],
                        batch_size=64, epochs=150, shuffle=True,
                        verbose=0, validation_data=(inp_val, y_val)
                        )
repeat(6, session)

                    
# Load model weights
model.load_weights(check1)

# Error on TEST set 
y_pred_scaled = model.predict(inp_test)
y_pred = y_pred_scaled.T[0]  # in kcal/mol unit
y_obs = y_test #/Eunit

err_test = np.sqrt(mean_squared_error(y_pred, y_obs))
errAbs_test = mean_absolute_error(y_pred, y_obs) 
print('RMSE_test:', '%.4f' % err_test)
print('MAE_test:','%.4f' % errAbs_test)
#print('The mean value of energies in test set: ', '%.4f' %E_ref_orig[test_idx].mean())

# Scatter plot of predicted and true values of energies in the test set
x2 = pd.Series(y_pred, name="Predicted")
x1 = pd.Series(y_obs, name="Observed")

data1 = pd.concat([x1, x2], axis=1)
g = sns.lmplot("Observed", "Predicted", data1,
        scatter_kws={"marker": ".", "color": "navy", "alpha": 0.4 },
        line_kws={"linewidth": 1, "color": "orange"},
        height=8, aspect=1);
plt.plot(ls="--", c=".1")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='purple')
plt.grid(which='minor', linestyle='-', linewidth='0.5', color='blue', alpha=0.25)
plt.minorticks_on()
plt.title('Test Set Prediction', fontsize=20)
plt.savefig("TestSet",figsize=[4,4], dpi=300)


for el in SUPPORTED_ELEMENTS:
    nm = f"{el}-subnet"
    write_subnet_text(model, save_dir, nm)

from keras.models import load_model

#TODO: Automatically collects all the sublayers. 
model.get_layer("H-subnet").save(os.path.join(save_dir, 'H-subnet.h5'))
model.get_layer("S-subnet").save(os.path.join(save_dir, 'S-subnet.h5'))
model.get_layer("C-subnet").save(os.path.join(save_dir, 'C-subnet.h5'))

os.mkdir('Model')
save_dir = "Model"

model.save(os.path.join(save_dir, 'model.h5'))

def write_np_arr_to_pkl(np_arr, save_dir, file_name):
    with open(os.path.join(save_dir, file_name), "wb") as pkl_file:
        pickle.dump(np_arr, pkl_file)    

write_np_arr_to_pkl(inp_test,save_dir,  "inp_test.pkl")
write_np_arr_to_pkl(y_pred,save_dir,  "y_pred.pkl")
write_np_arr_to_pkl(y_obs, save_dir, "y_obs.pkl")
