from __future__ import print_function

import argparse
import time

import torch

from torch.autograd.variable import Variable
from torch.nn import MSELoss
from torch.optim import Adam

from imputation_dgm.commandline import parse_int_list, create_parent_directories_if_needed
from imputation_dgm.cuda import to_cuda_if_available, to_cpu_if_available, load_without_cuda
from imputation_dgm.formats import data_formats, load
from imputation_dgm.imputation.masks import generate_mask_for, compose_with_mask, masked_reconstruction_loss_function
from imputation_dgm.methods.general.logger import Logger
from imputation_dgm.methods.vae.vae import VAE
from imputation_dgm.rng import seed_all
from imputation_dgm.variables import load_variable_sizes_from_metadata


def impute(vae,
           features,
           mask,
           output_loss_path,
           max_iterations=1000,
           tolerance=1e-3,
           variable_sizes=None,
           noise_learning_rate=None
           ):
    start_time = time.time()
    vae = to_cuda_if_available(vae)

    logger = Logger(output_loss_path, append=False)

    loss_function = MSELoss()

    inverted_mask = 1 - mask

    observed = features * mask
    missing = torch.randn_like(features)

    if noise_learning_rate is not None:
        missing = Variable(missing, requires_grad=True)
        optim = Adam([missing], weight_decay=0, lr=noise_learning_rate)

    vae.train(mode=True)

    for iteration in range(max_iterations):
        logger.start_timer()

        if noise_learning_rate is not None:
            optim.zero_grad()

        noisy_features = observed + missing * inverted_mask
        _, reconstructed, _, _ = vae(noisy_features, training=True)

        observed_loss = masked_reconstruction_loss_function(reconstructed,
                                                            features,
                                                            mask,
                                                            variable_sizes)

        missing_loss = masked_reconstruction_loss_function(reconstructed,
                                                           features,
                                                           inverted_mask,
                                                           variable_sizes)

        loss = torch.sqrt(loss_function(compose_with_mask(features, reconstructed, mask), features))

        if noise_learning_rate is None:
            missing = reconstructed * inverted_mask
        else:
            observed_loss.backward()
            optim.step()

        observed_loss, missing_loss, loss = to_cpu_if_available(observed_loss, missing_loss, loss)
        observed_loss = observed_loss.data.numpy()
        missing_loss = missing_loss.data.numpy()
        loss = loss.data.numpy()

        logger.log(iteration, max_iterations, "vae", "observed_loss", observed_loss)
        logger.log(iteration, max_iterations, "vae", "missing_loss", missing_loss)
        logger.log(iteration, max_iterations, "vae", "loss", loss)

        if observed_loss < tolerance:
            break

    logger.close()
    print("Total time: {:02f}s".format(time.time() - start_time))


def main(args=None):
    options_parser = argparse.ArgumentParser(description="Impute missing values with iterative VAE. "
                                                         + "Define 'temperature' to use multi-output.")

    options_parser.add_argument("data", type=str, help="See 'data_format' parameter.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("model", type=str, help="Model input file.")

    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument(
        "--data_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array, a sparse csr matrix or any of those formats in split into several files."
    )

    options_parser.add_argument(
        "--split_size",
        type=int,
        default=128,
        help="Dimension of the VAE latent space."
    )

    options_parser.add_argument(
        "--code_size",
        type=int,
        default=128,
        help="Dimension of the VAE latent space."
    )

    options_parser.add_argument(
        "--encoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the encoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--decoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the decoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations."
    )

    options_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Minimum RMSE to continue iterating."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature. Only used if metadata is also provided."
    )

    options_parser.add_argument(
        "--missing_probability",
        type=float,
        default=0.5,
        help="Probability of a value being missing."
    )

    options_parser.add_argument(
        "--noise_learning_rate",
        type=float,
        help="Learning rate to use backpropagation and modify the noise."
    )

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options = options_parser.parse_args(args=args)

    seed_all(options.seed)

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    features = load(options.data, options.data_format)

    mask = generate_mask_for(features, options.missing_probability, variable_sizes)
    mask = to_cuda_if_available(mask)

    if options.temperature is not None:
        temperature = options.temperature
    else:
        temperature = None

    vae = VAE(
        features.shape[1],
        options.split_size,
        options.code_size,
        encoder_hidden_sizes=parse_int_list(options.encoder_hidden_sizes),
        decoder_hidden_sizes=parse_int_list(options.decoder_hidden_sizes),
        variable_sizes=(None if temperature is None else variable_sizes),  # do not use multi-output without temperature
        temperature=temperature
    )

    load_without_cuda(vae, options.model)

    impute(
        vae,
        features,
        mask,
        create_parent_directories_if_needed(options.output_loss),
        max_iterations=options.max_iterations,
        tolerance=options.tolerance,
        variable_sizes=variable_sizes,
        noise_learning_rate=options.noise_learning_rate
    )


if __name__ == "__main__":
    main()
