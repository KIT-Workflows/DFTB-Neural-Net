import yaml, tarfile

import os                                                             
import pickle
import math
import sys
import itertools
import numpy as np                                                     
import pandas as pd                                                     
import tarfile                                     
                                                                       
from sklearn.metrics import mean_absolute_error                        
from sklearn.metrics import mean_squared_error                                             
from sklearn.model_selection import train_test_split                   
                                                                       
import tensorflow.keras                                                           
from tensorflow.keras.optimizers import Adam                                      
from tensorflow.keras.models import Model, Sequential                             
from tensorflow.keras.layers import Input, Dense, Dropout                  
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping   
                                                                       
import ase                                                             
from ase import io
import ase.build                                                                                                    
from ase.atoms import Atoms                                            
from ase.io import read, write                                         
from ase.calculators.dftb import Dftb                                  
                   
                                                                     

my_tar = tarfile.open('src.tar')
my_tar.extractall() # specify which folder to extract to
my_tar.close()
os.mkdir('NN-parameter') 
print(os.getcwd())
                                                                                                                        
import pickle
from itertools import combinations_with_replacement as comb_replace

from src.Calculator import src_nogrd                                                                                         
from src.Calculator.src_nogrd import xyzArr_generator       
from src.Calculator.src_nogrd import dictonary_elements
from src.Calculator.src_nogrd import at_idx_map_generator  
from src.Calculator.store_models import write_subnet_text
                                                                    
                                            
from src.Utils.dftb_traj_io import read_scan_traj
import src.Utils.netBuilder
from src.Utils.netBuilder import netBuilder
from src.Utils.netTrainer import netTrainer

def get_elements(struct):
    num=list(dict.fromkeys(struct.get_atomic_numbers()))
    num.sort()
    atomic_num=[f'{str(i)}' for i in num]
    ELEMENTS=[]
    for element in atomic_num:
        ELEMENTS.append(dictonary_elements[element])
    return ELEMENTS

def activations(temp_val):
    if temp_val=="tanh":   
        var_1="tanh"
    elif temp_val=="sigmoid":   
        var_1="sigmoid"
    else:  
        var_1="relu"   
    return var_1

def drop(temp_val):
    if temp_val=="NoFirstDrop":   
        var_2="NoFirstDrop"
    else:  
        var_2="NoDrop"   
    return var_2

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
    neurons=wano_file['Neurons']
    act_funct=wano_file['Activation function']
    dropout_type=wano_file['Dropout']
    layerss=wano_file['Hidden Layers']
    lr=wano_file['Learning rate']
    descriptor=wano_file['Descriptor']

    data_energy = wano_file['Energies'] 
    data_set = pd.read_csv(data_energy)

    try:
        ref_orca_file = wano_file['Ref-Energy'] 
        ref = pd.read_csv(ref_orca_file)
    except KeyError:
        ref_turbomole_file = wano_file['Ref-Energy'] 
        ref = pd.read_csv(ref_orca_file)

    if wano_file['Corrective model']:
        ref_dftb_file = wano_file['Ref-Energy-DFTB'] 
        ref_dftb = pd.read_csv(ref_dftb_file)
        try:
            data_set['Delta'] = (data_set['orca_energy'] - data_set['dftb_energy'] + ref_dftb['dftb_energy'].sum() - ref['orca_energy'].sum())*627.5
        except KeyError:
            data_set['Delta'] = (data_set['turbomole_energy'] - data_set['dftb_energy'] + ref_dftb['dftb_energy'].sum() - ref['turbomole_energy'].sum())*627.5
    else:
        try:
            data_set['Delta'] = (data_set['orca_energy'] - ref['orca_energy'].sum())*627.5
            Reference_e=(ref['orca_energy'].sum())*627.5
            Level='DFT'
        except KeyError:
            data_set['Delta'] = (data_set['turbomole_energy'] - ref['turbomole_energy'].sum())*627.5
            Reference_e=(ref['turbomole_energy'].sum())*627.5
            Level='DFT'
        else:
            data_set['Delta'] = (data_set['dftb_energy'] - ref['dftb_energy'].sum())*627.5
            Reference_e=(ref['dftb_energy'].sum())*627.5
            Level='DFTB'            
        
        
    md_train_arr_origin = read_scan_traj(filename=sys.path[0]+'/'+ geom_filename)

    md_train_arr = md_train_arr_origin.copy(deep=False).reset_index(drop=True)
    md_rel_energy_arr = data_set['Delta']
    # Get rid of error values.
    nan_index = np.where(np.isnan(md_rel_energy_arr))

    for idx in nan_index:
        md_train_arr.drop(idx)
        md_rel_energy_arr = md_rel_energy_arr[~np.isnan(md_rel_energy_arr)]


    nAtoms, xyzArr = xyzArr_generator(md_train_arr)

    SUPPORTED_ELEMENTS=get_elements(md_train_arr[0])
    at_idx_map = at_idx_map_generator(md_train_arr[0])

    Gfunc_data = pickle.load(open(descriptor, "rb"))

    n_symm_func = Gfunc_data[SUPPORTED_ELEMENTS[0]][at_idx_map[SUPPORTED_ELEMENTS[0]][0]].shape[1]
    builder = netBuilder(SUPPORTED_ELEMENTS, n_symm_func)
    subnets = builder.build_subnets(n_dense_layers=layerss, n_units=neurons, 
                        hidden_activation=activations(act_funct),
                        dropout_type=drop(dropout_type), dropout_ratio=0.015)
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
                        optimizer=Adam(lr=lr,decay=0.00001),
                        metrics=['mean_squared_error', 'mean_absolute_error'])
            
        history = model.fit(inp_train, y_train, \
                            callbacks=[checkpointer, earlystop],
                            batch_size=64, epochs=1500, shuffle=True,
                            verbose=0, validation_data=(inp_val, y_val)
                            )
        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
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

    if wano_file['Corrective model']:
        fname = "output_dict.yml"
        thisdict = {
        "RMSE":'%.4f kcal/mol' % err_test,
        "MAE":'%.4f kcal/mol' % errAbs_test,
        "Train_Range":'%.5f to %.5f' %(y_train.min(), y_train.max() ),
        "Test_Range":'%.5f to %.5f' %(y_test.min(), y_test.max() ),
        "Training_structures":int(len(data_set)*0.8),
        "Validation_structures":int(len(data_set)*0.1),
        "Testing_structures":int(len(data_set)*0.1),
        "DFTB_reference_energy":'%.5f kcal/mol'%((ref_dftb['dftb_energy'].sum())*627.5),
        "ORCA_reference_energy":'%.5f kcal/mol'%((ref['orca_energy'].sum())*627.5),
        "Neurons":neurons,
        "Layers":layerss,
        "Elements":SUPPORTED_ELEMENTS,
        "Number_of_symmetries":n_symm_func,
        "Activation_function":activations(act_funct),
        "Corrective model":"true"
        }

        with open(fname, 'w') as yaml_file:
            yaml_file.write( yaml.dump(thisdict, default_flow_style=False))

    else:
        fname = "output_dict.yml"
        thisdict = {
        "RMSE":'%.4f kcal/mol' % err_test,
        "MAE":'%.4f kcal/mol' % errAbs_test,
        "Train_Range":'%.5f to %.5f' %(y_train.min(), y_train.max() ),
        "Test_Range":'%.5f to %.5f' %(y_test.min(), y_test.max() ),
        "Training_structures":int(len(data_set)*0.8),
        "Validation_structures":int(len(data_set)*0.1),
        "Testing_structures":int(len(data_set)*0.1),
        "Reference_energy":'%.5f kcal/mol'% Reference_e,
        "Neurons":neurons,
        "Layers":layerss,
        "Elements":SUPPORTED_ELEMENTS,
        "Number_of_symmetries":n_symm_func,
        "Activation_function":activations(act_funct),
        "Corrective model":"false",
        "Level of theory":Level
        }

        with open(fname, 'w') as yaml_file:
            yaml_file.write( yaml.dump(thisdict, default_flow_style=False))

    os.mkdir('Model')
    save_dir = "Model"

    for el in SUPPORTED_ELEMENTS:
        nm = f"{el}-subnet"
        write_subnet_text(model, save_dir, nm)
    
    archive = tarfile.open("Model.tar", "w")
    for el in SUPPORTED_ELEMENTS:
        archive.add((f"Model/{el}-subnet.param"))
    archive.close()

    for el in SUPPORTED_ELEMENTS:
        model.get_layer(f"{el}-subnet").save(os.path.join(save_dir, f'{el}-subnet.h5'))


  
    model.save(os.path.join(save_dir, 'model.h5'))

    def write_np_arr_to_pkl(np_arr, save_dir, file_name):
        with open(os.path.join(save_dir, file_name), "wb") as pkl_file:
            pickle.dump(np_arr, pkl_file)    

    write_np_arr_to_pkl(inp_test,save_dir,  "inp_test.pkl")
    write_np_arr_to_pkl(y_pred,save_dir,  "y_pred.pkl")
    write_np_arr_to_pkl(y_obs, save_dir, "y_obs.pkl")
