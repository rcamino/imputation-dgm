from __future__ import print_function

import argparse

from sklearn.model_selection import train_test_split

from imputation_dgm.formats import data_formats, loaders, savers


def main():
    options_parser = argparse.ArgumentParser(description="Split features file into train and test files.")

    options_parser.add_argument("features", type=str, help="Input features file.")
    options_parser.add_argument("train_size", type=float, help="Number or proportion of samples for the train part.")
    options_parser.add_argument("train_features", type=str, help="Output train features file.")
    options_parser.add_argument("test_features", type=str, help="Output test features file.")

    options_parser.add_argument(
        "--features_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--labels", type=str, help="Input labels file.")
    options_parser.add_argument("--train_labels", type=str, help="Output train labels file.")
    options_parser.add_argument("--test_labels", type=str, help="Output test labels file.")

    options_parser.add_argument(
        "--labels_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--shuffle", default=False, action="store_true",
                                help="Shuffle the dataset before the split.")

    options_parser.add_argument("--stratify", default=False, action="store_true",
                                help="Split preserving class proportions (only valid with labels).")

    options = options_parser.parse_args()

    features_loader = loaders[options.features_format]
    features_saver = savers[options.features_format]
    features = features_loader(options.features, transform=False)

    if 0 < options.train_size < 1:
        train_size = options.train_size
        test_size = 1.0 - options.train_size  # a warning is thrown if not specified
    elif options.train_size < features.shape[0]:
        train_size = int(options.train_size)
        test_size = features.shape[0] - train_size  # a warning is thrown if not specified
    else:
        raise Exception("Invalid train size.")

    if options.labels is None:
        train_features, test_features = train_test_split(features,
                                                         train_size=train_size,
                                                         test_size=test_size,
                                                         shuffle=options.shuffle)
    else:
        labels_loader = loaders[options.labels_format]
        labels_saver = savers[options.labels_format]
        labels = labels_loader(options.labels, transform=False)

        if options.stratify:
            train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                        labels,
                                                                                        train_size=train_size,
                                                                                        test_size=test_size,
                                                                                        shuffle=True,  # mandatory
                                                                                        stratify=labels)
        else:
            train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                        labels,
                                                                                        train_size=train_size,
                                                                                        test_size=test_size,
                                                                                        shuffle=options.shuffle)

    features_saver(options.train_features, train_features)
    features_saver(options.test_features, test_features)

    print("Train features:", train_features.shape, "Test features:", test_features.shape)

    if options.labels is not None:
        labels_saver(options.train_labels, train_labels)
        labels_saver(options.test_labels, test_labels)

        print("Train labels:", train_labels.shape, "Test labels:", test_labels.shape)


if __name__ == "__main__":
    main()
