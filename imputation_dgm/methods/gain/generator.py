from __future__ import print_function

import torch
import torch.nn as nn

from imputation_dgm.methods.general.multi_input import MultiInput
from imputation_dgm.methods.general.multi_output import MultiOutput
from imputation_dgm.methods.general.single_output import SingleOutput


class Generator(nn.Module):

    def __init__(self, size, hidden_sizes=[], mask_variables=False, temperature=None):
        super(Generator, self).__init__()

        hidden_activation = nn.Tanh()

        if type(size) is int:
            self.multi_input_layer = None
            previous_layer_size = size * 2
        elif type(size) is list:
            self.multi_input_layer = MultiInput(size)

            if mask_variables:
                previous_layer_size = self.multi_input_layer.size + len(size)
            else:
                previous_layer_size = self.multi_input_layer.size + sum(size)
        else:
            raise Exception("Invalid size.")

        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(size) is int:
            self.output_layer = SingleOutput(previous_layer_size, size, activation=nn.Sigmoid())
        elif type(size) is list:
            assert temperature is not None
            self.output_layer = MultiOutput(previous_layer_size, size, temperature=temperature)

    def forward(self, inputs, mask, training=False):
        if self.multi_input_layer is None:
            outputs = inputs
        else:
            outputs = self.multi_input_layer(inputs)

        outputs = torch.cat((outputs, mask), dim=1)

        if self.hidden_layers is not None:
            outputs = self.hidden_layers(outputs)

        return self.output_layer(outputs, training=training)
