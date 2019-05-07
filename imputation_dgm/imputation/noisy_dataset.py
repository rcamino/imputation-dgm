import torch

from torch.utils.data import Dataset

from imputation_dgm.cuda import to_cuda_if_available
from imputation_dgm.imputation.masks import compose_with_mask, generate_mask_for


class NoisyDataset(Dataset):

    def __init__(self, features, mask, return_all=False):
        self.features = features
        self.mask = mask
        self.return_all = return_all

    def __getitem__(self, index):
        features = self.features[index]
        mask = self.mask[index]
        noise = torch.randn_like(features)
        noisy_features = compose_with_mask(features, noise, mask)
        if self.return_all:
            return features, mask, noisy_features
        else:
            return noisy_features

    def __len__(self):
        return len(self.features)


def create_noisy_dataset(features, missing_probability, variable_sizes, return_all=False):
    mask = generate_mask_for(features, missing_probability, variable_sizes)
    mask = to_cuda_if_available(mask)
    return NoisyDataset(features, mask, return_all=return_all)
