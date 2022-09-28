import pickle
import keras
import os
from keras.models import model_from_json


def read_data_from_pickle(save_dir, file_path):
    file_path_dir = os.path.join(save_dir, file_path)
    with open(file_path_dir, "rb") as pkl_file:
        data = pickle.load(pkl_file, encoding='latin1')
    return data

def read_keras_subnet_ind(save_dir, subnet_name):
    """
    Read the given subnet of keras model to file.

    For a given keras model (model), with the given Name of the subbnet (subnet_name, save
    the model to json file and the weight to a h5 file.

        Args:
            subnet_name: the name for the sub neural network as in model.summary()
                    (The element must be capital case)
                    e.g. 'H-subnet'

        Outputs:
            subnet: a keras model for the given subnet.

            Read the given sub neural network in the ../src.folder.
            The architecture is from 'subnet_name.json'
            The weight is from 'subnet_name-weight.h5'

    Comments:
    Needs to run the shell script to clean all the data
    Please also refer to write_keras_subnet_ind() function.
    """
    subnet_file = os.path.join(save_dir, subnet_name+".json")
    subnet_weight = os.path.join(save_dir, subnet_name+"-weight.h5")
    with open(subnet_file, "r") as json_file:
        loaded_subnet = json_file.read()
    subnet = model_from_json(loaded_subnet)
    subnet.load_weights(subnet_weight)
    #subnet.summary()
    return subnet
