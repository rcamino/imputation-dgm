from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.one_hot_categorical import OneHotCategorical

from imputation_dgm.methods.general.output_layer import OutputLayer


class MultiOutput(OutputLayer):

    def __init__(self, input_size, variable_sizes):
        super(MultiOutput, self).__init__()

        self.output_layers = nn.ModuleList()
        self.output_activations = nn.ModuleList()

        numerical_size = 0
        for i, variable_size in enumerate(variable_sizes):
            # if it is a categorical variable
            if variable_size > 1:
                # first create the accumulated numerical layer
                if numerical_size > 0:
                    self.output_layers.append(nn.Linear(input_size, numerical_size))
                    self.output_activations.append(NumericalActivation())
                    numerical_size = 0
                # create the categorical layer
                self.output_layers.append(nn.Linear(input_size, variable_size))
                self.output_activations.append(CategoricalActivation())
            # if not, accumulate numerical variables
            else:
                numerical_size += 1

        # create the remaining accumulated numerical layer
        if numerical_size > 0:
            self.output_layers.append(nn.Linear(input_size, numerical_size))
            self.output_activations.append(NumericalActivation())

    def forward(self, inputs, training=True, temperature=None, concat=True):
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(inputs)
            output = output_activation(logits, training=training, temperature=temperature)
            outputs.append(output)

        if concat:
            return torch.cat(outputs, dim=1)
        else:
            return outputs


class CategoricalActivation(nn.Module):

    def __init__(self):
        super(CategoricalActivation, self).__init__()

    def forward(self, logits, training=True, temperature=None):
        # gumbel-softmax (training and evaluation)
        if temperature is not None:
            return F.gumbel_softmax(logits, hard=not training, tau=temperature)
        # softmax training
        elif training:
            return F.softmax(logits, dim=1)
        # softmax evaluation
        else:
            return OneHotCategorical(logits=logits).sample()


class NumericalActivation(nn.Module):

    def __init__(self):
        super(NumericalActivation, self).__init__()

    def forward(self, logits, training=True, temperature=None):
        return torch.sigmoid(logits)
