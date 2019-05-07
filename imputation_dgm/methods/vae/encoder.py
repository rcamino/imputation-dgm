from __future__ import print_function

import torch.nn as nn

from imputation_dgm.methods.general.multi_input import MultiInput


class Encoder(nn.Module):

    def __init__(self, input_size, code_size, hidden_sizes=[], variable_sizes=None):
        super(Encoder, self).__init__()

        layers = []

        if variable_sizes is None:
            previous_layer_size = input_size
        else:
            multi_input_layer = MultiInput(variable_sizes)
            layers.append(multi_input_layer)
            previous_layer_size = multi_input_layer.size

        layer_sizes = list(hidden_sizes) + [code_size]
        hidden_activation = nn.Tanh()

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.hidden_layers(inputs)
