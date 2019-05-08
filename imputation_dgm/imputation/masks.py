import torch

from imputation_dgm.methods.general.loss import reconstruction_loss_function


def generate_mask_for(features, missing_probability, variable_sizes):
    num_samples = len(features)
    variable_masks = []
    for variable_size in variable_sizes:
        variable_mask = (torch.zeros(num_samples, 1).uniform_(0.0, 1.0) > missing_probability).float()
        if variable_size > 1:
            variable_mask = variable_mask.repeat(1, variable_size)
        variable_masks.append(variable_mask)
    return torch.cat(variable_masks, dim=1)


def compose_with_mask(positive, negative, mask):
    return mask * positive + (1.0 - mask) * negative


def masked_reconstruction_loss_function(reconstructed, original, mask, variable_sizes):
    return reconstruction_loss_function(mask * reconstructed,
                                        mask * original,
                                        variable_sizes,
                                        reduction="sum") / torch.sum(mask)
