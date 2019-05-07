from __future__ import print_function

import torch
import torch.nn as nn

from imputation_dgm.methods.vae.encoder import Encoder
from imputation_dgm.methods.vae.decoder import Decoder


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class VAE(nn.Module):

    def __init__(self, input_size, split_size, code_size, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 variable_sizes=None):

        super(VAE, self).__init__()

        self.encoder = Encoder(input_size, split_size, hidden_sizes=encoder_hidden_sizes, variable_sizes=variable_sizes)
        self.decoder = Decoder(code_size, input_size, hidden_sizes=decoder_hidden_sizes, variable_sizes=variable_sizes)

        self.mu_layer = nn.Linear(split_size, code_size)
        self.log_var_layer = nn.Linear(split_size, code_size)

    def forward(self, inputs, training=False, temperature=None):
        mu, log_var = self.encode(inputs)
        code = reparameterize(mu, log_var)
        reconstructed = self.decode(code, training=training, temperature=temperature)
        return code, reconstructed, mu, log_var

    def encode(self, inputs):
        outputs = self.encoder(inputs)
        return self.mu_layer(outputs), self.log_var_layer(outputs)

    def decode(self, code, training=False, temperature=None):
        return self.decoder(code, training=training, temperature=temperature)
