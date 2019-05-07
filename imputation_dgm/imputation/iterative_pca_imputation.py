from __future__ import print_function

import argparse
import time

import numpy as np

from imputation_dgm.commandline import create_parent_directories_if_needed
from imputation_dgm.formats import data_formats, loaders
from imputation_dgm.methods.general.logger import Logger

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def impute(features,
           mask,
           output_loss_path,
           max_iterations=1000,
           tolerance=1e-5
           ):
    start_time = time.time()

    logger = Logger(output_loss_path, append=False)

    inverted_mask = 1 - mask

    observed = features * mask
    missing = np.random.standard_normal(features.shape).astype(features.dtype)

    for iteration in range(max_iterations):
        logger.start_timer()

        noisy_features = observed + missing * inverted_mask

        standard_model = StandardScaler()
        standardized = standard_model.fit_transform(noisy_features)

        pca_model = PCA(n_components=0.99, svd_solver="full")
        transformed = pca_model.fit_transform(standardized)

        reconstructed = standard_model.inverse_transform(pca_model.inverse_transform(transformed))

        observed_loss = ((features * mask - reconstructed * mask) ** 2).sum() / mask.sum()

        missing_loss = ((features * inverted_mask - reconstructed * inverted_mask) ** 2).sum() / inverted_mask.sum()

        loss = np.sqrt(((features - (observed + reconstructed * inverted_mask)) ** 2).mean())

        logger.log(iteration, max_iterations, "pca", "observed_loss", observed_loss)
        logger.log(iteration, max_iterations, "pca", "missing_loss", missing_loss)
        logger.log(iteration, max_iterations, "pca", "loss", loss)

        missing = reconstructed * inverted_mask

        if observed_loss < tolerance:
            break

    logger.close()
    print("Total time: {:02f}s".format(time.time() - start_time))


def main(args=None):
    options_parser = argparse.ArgumentParser(description="Impute missing values with iterative PCA.")

    options_parser.add_argument("data", type=str, help="See 'data_format' parameter.")

    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument(
        "--data_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array, a sparse csr matrix or any of those formats in split into several files."
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
        "--missing_probability",
        type=float,
        default=0.5,
        help="Probability of a value being missing."
    )

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options = options_parser.parse_args(args=args)

    if options.seed is not None:
        np.random.seed(options.seed)

    loader = loaders[options.data_format]
    features = loader(options.data)

    mask = (np.random.uniform(0, 1, features.shape) > options.missing_probability).astype(features.dtype)

    impute(
        features,
        mask,
        create_parent_directories_if_needed(options.output_loss),
        max_iterations=options.max_iterations,
        tolerance=options.tolerance
    )


if __name__ == "__main__":
    main()
