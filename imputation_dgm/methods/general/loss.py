import torch

from torch.nn import functional as F


def reconstruction_loss_function(reconstructed, original, variable_sizes, size_average=True):
    # by default use loss for binary variables
    if variable_sizes is None:
        return F.binary_cross_entropy(reconstructed, original, size_average=size_average)
    # use the variable sizes when available
    else:
        loss = 0
        start = 0
        numerical_size = 0
        for variable_size in variable_sizes:
            # if it is a categorical variable
            if variable_size > 1:
                # add loss from the accumulated continuous variables
                if numerical_size > 0:
                    end = start + numerical_size
                    batch_reconstructed_variable = reconstructed[:, start:end]
                    batch_target = original[:, start:end]
                    loss += F.mse_loss(batch_reconstructed_variable, batch_target, size_average=size_average)
                    start = end
                    numerical_size = 0
                # add loss from categorical variable
                end = start + variable_size
                batch_reconstructed_variable = reconstructed[:, start:end]
                batch_target = torch.argmax(original[:, start:end], dim=1)
                loss += F.cross_entropy(batch_reconstructed_variable, batch_target, size_average=size_average)
                start = end
            # if not, accumulate numerical variables
            else:
                numerical_size += 1

        # add loss from the remaining accumulated numerical variables
        if numerical_size > 0:
            end = start + numerical_size
            batch_reconstructed_variable = reconstructed[:, start:end]
            batch_target = original[:, start:end]
            loss += F.mse_loss(batch_reconstructed_variable, batch_target, size_average=size_average)

        return loss
