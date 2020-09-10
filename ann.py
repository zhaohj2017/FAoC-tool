import torch
import torch.nn as nn
import numpy as np
import superp



############################################
# generate nn architecture
############################################

def gen_nn(n_h_layers, n_h_neurons, dim_in, dim_out, act_fun, out_bound):
    # input layer and output layer
    layer_input = [nn.Linear(dim_in, n_h_neurons, bias=True)]
    layer_output = [act_fun, nn.Linear(n_h_neurons, dim_out, bias=True), nn.Hardtanh(-out_bound, out_bound)]

    # hidden layer
    module_hidden = [[act_fun, nn.Linear(n_h_neurons, n_h_neurons, bias=True)] for _ in range(n_h_layers - 1)]
    layer_hidden = list(np.array(module_hidden).flatten())

    # nn model
    layers = layer_input + layer_hidden + layer_output
    model = nn.Sequential(*layers)

    return model
