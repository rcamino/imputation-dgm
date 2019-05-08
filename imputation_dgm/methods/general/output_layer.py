from __future__ import print_function

import torch.nn as nn


class OutputLayer(nn.Module):
    """
    This is just a simple abstract class for single and multi output layers.
    Both need to have the same interface.
    """

    def forward(self, hidden, training=None):
        raise NotImplementedError
