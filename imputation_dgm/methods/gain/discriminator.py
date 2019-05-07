from __future__ import print_function

import torch
import torch.nn as nn

from imputation_dgm.methods.general.multi_input import MultiInput


class Discriminator(nn.Module):

    def __init__(self, size, hidden_sizes=[], hint_variables=False):
        super(Discriminator, self).__init__()

        hidden_activation = nn.Tanh()

        if type(size) is int:
            self.multi_input_layer = None
            previous_layer_size = size * 2
            output_size = size
        elif type(size) is list:
            self.multi_input_layer = MultiInput(size)

            if hint_variables:
                previous_layer_size = self.multi_input_layer.size + len(size)
                output_size = len(size)
            else:
                previous_layer_size = self.multi_input_layer.size + sum(size)
                output_size = sum(size)
        else:
            raise Exception("Invalid size.")

        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs, hints):
        if self.multi_input_layer is None:
            outputs = inputs
        else:
            outputs = self.multi_input_layer(inputs)

        outputs = torch.cat((outputs, hints), dim=1)

        return self.model(outputs)
