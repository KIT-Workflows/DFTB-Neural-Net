from Utils.netBuilder import netBuilder
from Utils.netTrainer import netTrainer
from sklearn.model_selection import ParameterGrid
from itertools import product

def gridSearcher():
    def __init__(self, supported_elements, input_dim):
        self.supported_elements = supported_elements
        self.input_dim = input_dim
        builder = netBuilder(self.supported_elements, self.input_dim)

    def fit(at_idx_map, train_inputs, targets, validation_data, build_param_grid, train_param_grid):
        build_grid = ParameterGrid(build_param_grid)
        train_grid = ParameterGrid(train_param_grid)

        for build_params, train_params in product(build_grid, train_grid):
            subnets = builder.build_subnets(**build_params)
            model = builder.build_molecular_net(at_idx_map, subnets)
