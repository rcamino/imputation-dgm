import torch

from torch.autograd.variable import Variable

from imputation_dgm.cuda import to_cuda_if_available


def generate_noise_like(batch, size):
    return generate_noise(len(batch), size)


def generate_noise(num_samples, num_features):
    noise = Variable(torch.FloatTensor(num_samples, num_features).normal_())
    return to_cuda_if_available(noise)


def label_zeros_like(batch):
    label_zeros = Variable(torch.zeros(len(batch)))
    return to_cuda_if_available(label_zeros)


def smooth_label_ones_like(batch):
    smooth_label_ones = Variable(torch.FloatTensor(len(batch)).uniform_(0.9, 1))
    return to_cuda_if_available(smooth_label_ones)
