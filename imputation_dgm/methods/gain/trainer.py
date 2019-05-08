from __future__ import print_function

import argparse
import time

import numpy as np

import torch

from torch.autograd.variable import Variable
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from imputation_dgm.commandline import parse_int_list, create_parent_directories_if_needed
from imputation_dgm.cuda import to_cuda_if_available, to_cpu_if_available
from imputation_dgm.formats import data_formats, load
from imputation_dgm.imputation.masks import generate_mask_for, compose_with_mask
from imputation_dgm.imputation.noisy_dataset import create_noisy_dataset
from imputation_dgm.methods.gain.discriminator import Discriminator
from imputation_dgm.methods.gain.generator import Generator
from imputation_dgm.methods.general.initialization import load_or_initialize
from imputation_dgm.methods.general.logger import Logger
from imputation_dgm.methods.general.loss import reconstruction_loss_function
from imputation_dgm.methods.general.saver import Saver
from imputation_dgm.rng import seed_all
from imputation_dgm.variables import load_variable_sizes_from_metadata


class Trainer(object):

    def __init__(self, train_data, test_data, generator, discriminator, optim_gen, optim_disc, batch_size,
                 variable_sizes, num_disc_steps, num_gen_steps, reconstruction_loss_weight, hint_probability,
                 temperature):

        self.train_data = train_data
        self.test_data = test_data
        self.generator = generator
        self.discriminator = discriminator
        self.optim_gen = optim_gen
        self.optim_disc = optim_disc
        self.batch_size = batch_size
        self.variable_sizes = variable_sizes
        self.num_disc_steps = num_disc_steps
        self.num_gen_steps = num_gen_steps
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.hint_probability = hint_probability
        self.temperature = temperature

        self.adversarial_loss_function = BCELoss()
        self.test_loss_function = MSELoss()

    def train(self):
        self.generator.train(mode=True)
        self.discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        more_batches = True

        # use the train data twice but with different shuffles
        disc_iterator = iter(DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True))
        gen_iterator = iter(DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True))

        while more_batches:
            # train discriminator
            for _ in range(self.num_disc_steps):
                try:
                    features, mask, noisy_features = next(disc_iterator)
                    disc_losses.append(self.train_discriminator(features, mask, noisy_features))
                except StopIteration:
                    break

            # train generator
            for _ in range(self.num_gen_steps):
                try:
                    features, mask, noisy_features = next(gen_iterator)
                    gen_losses.append(self.train_generator(features, mask, noisy_features))
                except StopIteration:
                    # stop when the generator has no more batches
                    # the discriminator could stop first if it uses more steps than the generator
                    more_batches = False
                    break

        return disc_losses, gen_losses

    def train_discriminator(self, features, mask, noisy_features):
        self.optim_disc.zero_grad()

        generated = self.generator(Variable(noisy_features), Variable(mask),
                                   training=True, temperature=self.temperature)
        imputed = compose_with_mask(noisy_features, generated, mask)
        hint = generate_hint_for(mask, self.hint_probability, self.variable_sizes)
        # detach to avoid back-propagation to the generator not needed for discriminator training (goes faster)
        predictions = self.discriminator(imputed.detach(), Variable(hint))

        # how good is the discriminator on detecting the fakes on each position
        loss = self.adversarial_loss_function(predictions, mask)

        loss.backward()
        self.optim_disc.step()

        loss = to_cpu_if_available(loss)
        return loss.data.item()

    def train_generator(self, features, mask, noisy_features):
        self.optim_gen.zero_grad()

        generated = self.generator(Variable(noisy_features), Variable(mask),
                                   training=True, temperature=self.temperature)
        imputed = compose_with_mask(noisy_features, generated, mask)
        hint = generate_hint_for(mask, self.hint_probability, self.variable_sizes)
        predictions = self.discriminator(imputed, Variable(hint))

        # how good is the generator on fooling the discriminator on each position
        adversarial_loss = self.adversarial_loss_function(predictions, (1 - mask))
        # reconstruction of the non-missing values (averaged by the number of non-missing values)
        reconstruction_loss = reconstruction_loss_function(mask * generated,
                                                           mask * features,
                                                           self.variable_sizes,
                                                           reduction="sum") / torch.sum(mask)
        # combine both losses with a weight on the reconstruction loss
        loss = adversarial_loss + reconstruction_loss * self.reconstruction_loss_weight

        loss.backward()
        self.optim_gen.step()

        loss = to_cpu_if_available(loss)
        return loss.data.item()

    def test(self):
        self.generator.train(mode=False)
        self.discriminator.train(mode=False)

        # only one batch
        iterator = iter(DataLoader(self.test_data, batch_size=len(self.test_data), shuffle=True))
        features, mask, noisy_features = iterator.next()

        with torch.no_grad():
            generated = self.generator(Variable(noisy_features), Variable(mask),
                                       training=True, temperature=self.temperature)
            imputed = compose_with_mask(noisy_features, generated, mask)

            # transform MSE into RMSE (to report the same metrics from the paper)
            loss = torch.sqrt(self.test_loss_function(imputed, features))

            loss = to_cpu_if_available(loss)
            loss = loss.data.cpu().item()

            return loss


def train(generator,
          discriminator,
          train_data,
          test_data,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          batch_size=64,
          start_epoch=0,
          num_epochs=10000,
          num_disc_steps=1,
          num_gen_steps=1,
          l2_regularization=0,
          learning_rate=0.001,
          variable_sizes=None,
          reconstruction_loss_weight=1,
          hint_probability=0.9,
          max_seconds_without_save=300,
          early_stopping_patience=100,
          temperature=None
          ):
    start_time = time.time()
    generator, discriminator = to_cuda_if_available(generator, discriminator)

    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path, append=start_epoch > 0)

    saver = Saver({generator: output_gen_path, discriminator: output_disc_path}, logger, max_seconds_without_save)

    trainer = Trainer(train_data, test_data, generator, discriminator, optim_gen, optim_disc, batch_size,
                      variable_sizes, num_disc_steps, num_gen_steps, reconstruction_loss_weight,
                      hint_probability, temperature)

    # initialize early stopping
    best_test_mean_loss = None
    bad_epochs = 0

    for epoch_index in range(start_epoch, num_epochs):
        # train discriminator and generator
        logger.start_timer()
        disc_losses, gen_losses = trainer.train()
        logger.log(epoch_index, num_epochs, "discriminator", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator", "train_mean_loss", np.mean(gen_losses))

        # test imputation
        logger.start_timer()
        reconstruction_losses = trainer.test()
        test_mean_loss = np.mean(reconstruction_losses)
        logger.log(epoch_index, num_epochs, "generator", "test_mean_loss", test_mean_loss)

        # check if the test loss is improving
        if best_test_mean_loss is None or test_mean_loss < best_test_mean_loss:
            best_test_mean_loss = test_mean_loss
            bad_epochs = 0

            # save models for the epoch
            saver.delayed_save(keep_parameters=True)

        # if the test loss is not improving check if early stopping should be executed
        else:
            bad_epochs += 1
            if bad_epochs >= early_stopping_patience:
                break

    saver.save(only_use_kept=True)
    logger.close()
    print("Total time: {:02f}s".format(time.time() - start_time))


def generate_hint_for(mask, hint_probability, variable_sizes):
    return mask * to_cuda_if_available(generate_mask_for(mask, 1.0 - hint_probability, variable_sizes))


def main(args=None):
    options_parser = argparse.ArgumentParser(description="Train GAIN. Define 'temperature' to use multi-output.")

    options_parser.add_argument("train_data", type=str, help="Training data. See 'data_format' parameter.")
    options_parser.add_argument("test_data", type=str, help="Testing data. See 'data_format' parameter.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("output_generator", type=str, help="Generator output file.")
    options_parser.add_argument("output_discriminator", type=str, help="Discriminator output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_generator", type=str, help="Generator input file.", default=None)
    options_parser.add_argument("--input_discriminator", type=str, help="Discriminator input file.", default=None)

    options_parser.add_argument(
        "--data_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
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
        default=10000,
        help="Number of epochs."
    )

    options_parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0,
        help="L2 regularization weight for every parameter."
    )

    options_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Adam learning rate."
    )

    options_parser.add_argument(
        "--generator_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the generator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--discriminator_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the discriminator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--num_discriminator_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the discriminator."
    )

    options_parser.add_argument(
        "--num_generator_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the generator."
    )

    options_parser.add_argument(
        "--max_seconds_without_save",
        type=int,
        default=300,
        help="Amount of seconds between model saving. The model always will be saved after the last epoch."
    )
    options_parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=1,
        help="Reconstruction loss weight for the generator training."
    )

    options_parser.add_argument(
        "--missing_probability",
        type=float,
        default=0.5,
        help="Probability of a value being missing."
    )

    options_parser.add_argument(
        "--hint_probability",
        type=float,
        default=0.9,
        help="Probability of giving the answer to the discriminator per value."
    )

    options_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=100,
        help="Maximum tolerable amount of epochs without bad results before stopping."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature."
    )

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options = options_parser.parse_args(args=args)

    seed_all(options.seed)

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    train_features = load(options.train_data, options.data_format)
    test_features = load(options.test_data, options.data_format)

    train_data = create_noisy_dataset(train_features, options.missing_probability, variable_sizes, return_all=True)
    test_data = create_noisy_dataset(test_features, options.missing_probability, variable_sizes, return_all=True)

    if options.temperature is None:
        size = train_features.shape[1]
    else:
        size = variable_sizes

    generator = Generator(size, hidden_sizes=parse_int_list(options.generator_hidden_sizes))
    load_or_initialize(generator, options.input_generator)

    discriminator = Discriminator(size, hidden_sizes=parse_int_list(options.discriminator_hidden_sizes))
    load_or_initialize(discriminator, options.input_discriminator)

    train(
        generator,
        discriminator,
        train_data,
        test_data,
        create_parent_directories_if_needed(options.output_generator),
        create_parent_directories_if_needed(options.output_discriminator),
        create_parent_directories_if_needed(options.output_loss),
        batch_size=options.batch_size,
        start_epoch=options.start_epoch,
        num_epochs=options.num_epochs,
        num_disc_steps=options.num_discriminator_steps,
        num_gen_steps=options.num_generator_steps,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        variable_sizes=variable_sizes,
        reconstruction_loss_weight=options.reconstruction_loss_weight,
        hint_probability=options.hint_probability,
        max_seconds_without_save=options.max_seconds_without_save,
        early_stopping_patience=options.early_stopping_patience,
        temperature=options.temperature
    )


if __name__ == "__main__":
    main()
