import numpy as np
import tensorflow
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

def Nguyen_Widrow_init(shape, dtype=None):
    '''Nguyen-Widrow initialization for weights and biases
    Initialize the weight and biases for each layer.

            Args:
                shape: ???? (Guess: Shape of the layer to create)

            Output:
                [w,b]
                w: represents an array of weight
                b: represents an array of bias

    Comment:
    This is the function provided by Nguyen's original code.
    Junmian Zhu did the documentation. It is not 100% sure about the
    algorithm used here.

    '''
    n_input, n_unit = shape[0], shape[1]  # n_input: number of units of prev. layer without bias
    w_init = np.random.rand(n_input, n_unit) *2 -1
    norm = 0.7 * n_unit ** (1. / n_input)
    # normalize
    w = norm * w_init/np.sqrt(np.square(w_init).sum(axis=0).reshape(1, n_unit))
    if n_unit>1:
        b = norm * np.linspace(-1,1,n_unit) * np.sign(w[0,:])
    else:
        b = np.zeros((n_unit,))
    return [w, b]

class netBuilder():
    def __init__(self, supported_elements, input_dim):
        self.supported_elements = supported_elements
        self.input_dim = input_dim

    def build_subnets(self, n_dense_layers=3, n_units=34,
                      hidden_activation='tanh',
                      dropout_type='NoDrop', dropout_ratio=0.2):
        elemental_subnets = {}
        for element in self.supported_elements:
            subnet = self.build_elemental_subnet(n_dense_layers, n_units,
                                                 hidden_activation,
                                                 dropout_type, dropout_ratio)
            subnet._name = f"{element}-subnet"
            elemental_subnets[element] = subnet
        return elemental_subnets

    def build_elemental_subnet(self, n_dense_layers, n_units, hidden_activation, dropout_type, dropout_ratio):
        subnet = Sequential()
        for i in range(n_dense_layers):
            subnet.add(Dense(n_units, input_shape=(self.input_dim,),
                             activation=hidden_activation))
            self.initialize_weights(subnet.layers[-1])
            if dropout_type=="NoFirstDrop" and i!=0:
                subnet.add(Dropout(dropout_ratio))

        subnet.add(Dense(1, activation='linear'))
        self.initialize_weights(subnet.layers[-1])

        return subnet

    def initialize_weights(self, layer):
        param_shape = (layer.input_shape[1], layer.output_shape[1])
        layer.set_weights(Nguyen_Widrow_init(param_shape))


    def build_molecular_net(self, at_idx_map, elemental_subnets):
        inputs = []
        subnet_outputs = []
        atomic_nets = {}
        for type, atom_indices in at_idx_map.items():
            atomic_nets[type] = {}
            for idx in atom_indices:
                input = Input(shape=(self.input_dim,), dtype="float32",
                              name=f"{type}-{idx}-ele")
                inputs.append(input)
                atomic_subnet = elemental_subnets[type](input)
                subnet_outputs.append(atomic_subnet)
                #atomic_nets[type][atom_idx] = atomic_subnet
        total_output = keras.layers.add(subnet_outputs)
        molecular_net = Model(inputs, total_output)
        return molecular_net
