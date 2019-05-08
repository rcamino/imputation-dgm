from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.one_hot_categorical import OneHotCategorical

from imputation_dgm.methods.general.output_layer import OutputLayer


class MultiOutput(OutputLayer):

    def __init__(self, input_size, variable_sizes, temperature=None):
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
                self.output_activations.append(CategoricalActivation(temperature))
            # if not, accumulate numerical variables
            else:
                numerical_size += 1

        # create the remaining accumulated numerical layer
        if numerical_size > 0:
            self.output_layers.append(nn.Linear(input_size, numerical_size))
            self.output_activations.append(NumericalActivation())

    def forward(self, inputs, training=True, concat=True):
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(inputs)
            output = output_activation(logits, training=training)
            outputs.append(output)

        if concat:
            return torch.cat(outputs, dim=1)
        else:
            return outputs


class CategoricalActivation(nn.Module):

    def __init__(self, temperature):
        super(CategoricalActivation, self).__init__()

        self.temperature = temperature

    def forward(self, logits, training=True):
        # gumbel-softmax (training and evaluation)
        if self.temperature is not None:
            return F.gumbel_softmax(logits, hard=not training, tau=self.temperature)
        # softmax training
        elif training:
            return F.softmax(logits, dim=1)
        # softmax evaluation
        else:
            return OneHotCategorical(logits=logits).sample()


class NumericalActivation(nn.Module):

    def __init__(self):
        super(NumericalActivation, self).__init__()

    def forward(self, logits, training=True):
        return torch.sigmoid(logits)
