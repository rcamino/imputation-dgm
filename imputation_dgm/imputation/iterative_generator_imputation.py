from __future__ import print_function

import time
import torch

from torch.autograd.variable import Variable
from torch.nn import MSELoss
from torch.optim import Adam

from imputation_dgm.cuda import to_cuda_if_available, to_cpu_if_available
from imputation_dgm.imputation.masks import compose_with_mask, masked_reconstruction_loss_function
from imputation_dgm.methods.general.batches import generate_noise_like
from imputation_dgm.methods.general.logger import Logger


def impute(generator,
           features,
           mask,
           output_loss_path,
           noise_size,
           noise_learning_rate,
           max_iterations=1000,
           tolerance=1e-3,
           variable_sizes=None,
           ):
    start_time = time.time()
    generator = to_cuda_if_available(generator)

    logger = Logger(output_loss_path, append=False)

    loss_function = MSELoss()

    inverted_mask = 1 - mask

    observed = features * mask
    missing = torch.randn_like(features)

    missing = Variable(missing, requires_grad=True)
    optim = Adam([missing], weight_decay=0, lr=noise_learning_rate)

    generator.train(mode=True)

    for iteration in range(max_iterations):
        logger.start_timer()

        optim.zero_grad()

        noisy_features = observed + missing * inverted_mask
        noise = generate_noise_like(noisy_features, noise_size)
        generated = generator(noise, training=True)

        observed_loss = masked_reconstruction_loss_function(generated,
                                                            features,
                                                            mask,
                                                            variable_sizes)

        missing_loss = masked_reconstruction_loss_function(generated,
                                                           features,
                                                           inverted_mask,
                                                           variable_sizes)

        loss = torch.sqrt(loss_function(compose_with_mask(features, generated, mask), features))

        observed_loss.backward()
        optim.step()

        observed_loss, missing_loss, loss = to_cpu_if_available(observed_loss, missing_loss, loss)
        observed_loss = observed_loss.data.numpy()
        missing_loss = missing_loss.data.numpy()
        loss = loss.data.numpy()

        logger.log(iteration, max_iterations, "generator", "observed_loss", observed_loss)
        logger.log(iteration, max_iterations, "generator", "missing_loss", missing_loss)
        logger.log(iteration, max_iterations, "generator", "loss", loss)

        if observed_loss < tolerance:
            break

    logger.close()
    print("Total time: {:02f}s".format(time.time() - start_time))
