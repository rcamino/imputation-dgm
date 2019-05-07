from __future__ import print_function

import torch
import torch.nn as nn


class MultiInput(nn.Module):

    def __init__(self, variable_sizes, min_embedding_size=2, max_embedding_size=50):
        super(MultiInput, self).__init__()

        self.has_categorical = False
        self.size = 0

        embeddings = nn.ParameterList()
        for i, variable_size in enumerate(variable_sizes):
            # if it is a numerical variable
            if variable_size == 1:
                embeddings.append(None)
                self.size += 1
            # if it is a categorical variable
            else:
                # this is an arbitrary rule of thumb taken from several blog posts
                embedding_size = max(min_embedding_size, min(max_embedding_size, variable_size / 2))

                # the embedding is implemented manually to be able to use one hot encoding
                # PyTorch embedding only accepts as input label encoding
                embedding = nn.Parameter(data=torch.Tensor(variable_size, embedding_size).normal_(), requires_grad=True)

                embeddings.append(embedding)
                self.size += embedding_size
                self.has_categorical = True

        if self.has_categorical:
            self.variable_sizes = variable_sizes
            self.embeddings = embeddings

    def forward(self, inputs):
        if self.has_categorical:
            outputs = []
            start = 0
            for variable_size, embedding in zip(self.variable_sizes, self.embeddings):
                # extract the variable
                end = start + variable_size
                variable = inputs[:, start:end]

                # numerical variable
                if variable_size == 1:
                    # leave the input as it is
                    outputs.append(variable)
                # categorical variable
                else:
                    output = torch.matmul(variable, embedding).squeeze(1)
                    outputs.append(output)

                # move the variable limits
                start = end

            # concatenate all the variable outputs
            return torch.cat(outputs, dim=1)
        else:
            return inputs
