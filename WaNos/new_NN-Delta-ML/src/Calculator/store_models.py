import os
import numpy as np
import keras

def write_subnet_text(model, save_dir, subnet_name):
    subnet = model.get_layer(subnet_name)
    weights = subnet.get_weights()
    fname = f"{subnet_name}.param"
    f_contents = f""
    n_layers = 0
    for layer in subnet.layers:
        cfg = layer.get_config()
        sub_weights = layer.get_weights()
        if 'dense' in cfg['name']:
            n_layers += 1
            num_neurons = cfg['units']
            f_contents += str(num_neurons)+" "+cfg['activation']+"\n"
            for neuron in range(num_neurons):
                bias = sub_weights[1][neuron]
                weights = sub_weights[0][:, neuron]
                f_contents += str(bias)+" "
                for weight in weights:
                    f_contents += str(weight)+" "
                f_contents += "\n"

            f_contents += "\n"
        elif 'dropout' in cfg['name']:
            continue
        else:
            print(f"unexpected layer type: {cfg['name']}.")
            break
    f_contents = f"{n_layers}\n" + f_contents
    with open(os.path.join(save_dir, fname), "w") as out_f:
        out_f.write(f_contents)
