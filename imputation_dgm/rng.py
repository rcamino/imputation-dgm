from __future__ import print_function

import numpy as np

import torch


def seed_all(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
