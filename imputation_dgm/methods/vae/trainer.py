from __future__ import print_function

import argparse
import time
import torch

import numpy as np

from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from imputation_dgm.commandline import parse_int_list, create_parent_directories_if_needed
from imputation_dgm.cuda import to_cuda_if_available, to_cpu_if_available
from imputation_dgm.formats import data_formats, load
from imputation_dgm.imputation.masks import compose_with_mask, masked_reconstruction_loss_function
from imputation_dgm.imputation.noisy_dataset import create_noisy_dataset
from imputation_dgm.methods.general.initialization import load_or_initialize
from imputation_dgm.methods.general.logger import Logger
from imputation_dgm.methods.general.saver import Saver
from imputation_dgm.methods.vae.vae import VAE
from imputation_dgm.rng import seed_all
from imputation_dgm.variables import load_variable_sizes_from_metadata


class Trainer(object):

    def __init__(self, vae, train_data, test_data, batch_size, optim, variable_sizes):
        self.vae = vae
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.optim = optim
        self.variable_sizes = variable_sizes

        self.test_loss_function = MSELoss()

    def train(self):
        self.vae.train(mode=True)
        losses = []
        for features, mask, noisy_features in DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True):
            losses.append(self.train_batch(features, mask, noisy_features))
        return losses

    def train_batch(self, features, mask, noisy_features):
        self.optim.zero_grad()

        _, reconstructed, mu, log_var = self.vae(noisy_features, training=True)

        # reconstruction of the non-missing values
        reconstruction_loss = masked_reconstruction_loss_function(reconstructed,
                                                                  features,
                                                                  mask,
                                                                  self.variable_sizes)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconstruction_loss + kld_loss
        loss.backward()

        self.optim.step()

        loss = to_cpu_if_available(loss)
        return loss.data.numpy()

    def test(self):
        self.vae.train(mode=False)

        # only one batch
        iterator = iter(DataLoader(self.test_data, batch_size=len(self.test_data), shuffle=True))
        features, mask, noisy_features = iterator.next()

        with torch.no_grad():
            _, reconstructed, _, _ = self.vae(noisy_features, training=False)

            imputed = compose_with_mask(noisy_features, reconstructed, mask)

            # transform MSE into RMSE (to report the same metrics as GAIN)
            loss = torch.sqrt(self.test_loss_function(imputed, features))

            loss = to_cpu_if_available(loss)
            loss = loss.data.numpy()

            return loss


def train(vae,
          train_data,
          test_data,
          output_path,
          output_loss_path,
          batch_size=100,
          start_epoch=0,
          num_epochs=100,
          l2_regularization=0.001,
          learning_rate=0.001,
          variable_sizes=None,
          max_seconds_without_save=300
          ):
    start_time = time.time()
    vae = to_cuda_if_available(vae)

    optim = Adam(vae.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path, append=start_epoch > 0)

    saver = Saver({vae: output_path}, logger, max_seconds_without_save)

    trainer = Trainer(vae, train_data, test_data, batch_size, optim, variable_sizes)

    for epoch_index in range(start_epoch, num_epochs):
        # train vae
        logger.start_timer()
        train_losses = trainer.train()
        logger.log(epoch_index, num_epochs, "vae", "train_mean_loss", np.mean(train_losses))

        # test imputation
        logger.start_timer()
        test_loss = trainer.test()
        logger.log(epoch_index, num_epochs, "vae", "test_loss", test_loss)

        # save models for the epoch
        saver.delayed_save()

    saver.save()
    logger.close()
    print("Total time: {:02f}s".format(time.time() - start_time))


def main(args=None):
    options_parser = argparse.ArgumentParser(description="Train VAE for imputation. "
                                                         + "Define 'temperature' to use multi-output.")

    options_parser.add_argument("train_data", type=str, help="Training data. See 'data_format' parameter.")
    options_parser.add_argument("test_data", type=str, help="Testing data. See 'data_format' parameter.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("output_model", type=str, help="Model output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_model", type=str, help="Model input file.", default=None)

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
        "--batch_size",
        type=int,
        default=100,
        help="Amount of samples per batch."
    )

    options_parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch."
    )

    options_parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs."
    )

    options_parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.001,
        help="L2 regularization weight for every parameter."
    )

    options_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Adam learning rate."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature. Only used if metadata is also provided."
    )

    options_parser.add_argument(
        "--max_seconds_without_save",
        type=int,
        default=300,
        help="Amount of seconds between model saving. The model always will be saved after the last epoch."
    )

    options_parser.add_argument(
        "--missing_probability",
        type=float,
        default=0.5,
        help="Probability of a value being missing."
    )

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options = options_parser.parse_args(args=args)

    seed_all(options.seed)

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    train_features = load(options.train_data, options.data_format)
    test_features = load(options.test_data, options.data_format)

    train_data = create_noisy_dataset(train_features, options.missing_probability, variable_sizes, return_all=True)
    test_data = create_noisy_dataset(test_features, options.missing_probability, variable_sizes, return_all=True)

    if options.temperature is not None:
        temperature = options.temperature
    else:
        temperature = None

    vae = VAE(
        train_features.shape[1],
        options.split_size,
        options.code_size,
        encoder_hidden_sizes=parse_int_list(options.encoder_hidden_sizes),
        decoder_hidden_sizes=parse_int_list(options.decoder_hidden_sizes),
        variable_sizes=(None if temperature is None else variable_sizes),  # do not use multi-output without temperature
        temperature=temperature
    )

    load_or_initialize(vae, options.input_model)

    train(
        vae,
        train_data,
        test_data,
        create_parent_directories_if_needed(options.output_model),
        create_parent_directories_if_needed(options.output_loss),
        batch_size=options.batch_size,
        start_epoch=options.start_epoch,
        num_epochs=options.num_epochs,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        variable_sizes=variable_sizes,
        max_seconds_without_save=options.max_seconds_without_save
    )


if __name__ == "__main__":
    main()
