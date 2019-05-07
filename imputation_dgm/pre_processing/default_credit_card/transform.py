from __future__ import print_function

import argparse
import csv
import json

import numpy as np

from imputation_dgm.pre_processing.metadata import create_metadata

from sklearn.preprocessing.data import MinMaxScaler


NUM_SAMPLES = 30000

TYPES = {
    "LIMIT_BAL": "numerical",
    "SEX": "categorical",
    "EDUCATION": "categorical",
    "MARRIAGE": "categorical",
    "AGE": "numerical",
    "PAY_0": "categorical",
    "PAY_2": "categorical",
    "PAY_3": "categorical",
    "PAY_4": "categorical",
    "PAY_5": "categorical",
    "PAY_6": "categorical",
    "BILL_AMT1": "numerical",
    "BILL_AMT2": "numerical",
    "BILL_AMT3": "numerical",
    "BILL_AMT4": "numerical",
    "BILL_AMT5": "numerical",
    "BILL_AMT6": "numerical",
    "PAY_AMT1": "numerical",
    "PAY_AMT2": "numerical",
    "PAY_AMT3": "numerical",
    "PAY_AMT4": "numerical",
    "PAY_AMT5": "numerical",
    "PAY_AMT6": "numerical",
}

VALUES = {
    "SEX": {"1", "2"},
    "EDUCATION": {"0", "1", "2", "3", "4", "5", "6"},
    "MARRIAGE": {"0", "1", "2", "3"},
    "PAY_0": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
    "PAY_2": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
    "PAY_3": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
    "PAY_4": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
    "PAY_5": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
    "PAY_6": {"-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8"},
}

CLASSES = [
    "no default payment next month",
    "default payment next month",
]


def default_credit_card_transform(input_path, features_path, labels_path, metadata_path):
    input_file = open(input_path, "r")
    reader = csv.DictReader(input_file)

    variables = set(reader.fieldnames)
    variables.remove("ID")
    variables.remove("default payment next month")

    metadata = create_metadata(variables, TYPES, VALUES, NUM_SAMPLES, CLASSES)

    features = np.zeros((metadata["num_samples"], metadata["num_features"]), dtype=np.float32)
    labels = np.zeros(metadata["num_samples"], dtype=np.int32)

    # transform
    for i, row in enumerate(reader):
        # the categorical variables are already one hot encoded
        for j, variable in enumerate(metadata["variables"]):
            value = row[variable]
            if TYPES[variable] == "numerical":
                value = float(value)
                features[i, metadata["value_to_index"][variable]] = value
            elif TYPES[variable] == "categorical":
                value = value.replace(".0", "")
                assert value in VALUES[variable], \
                    "'{}' is not a valid value for '{}'".format(value, variable)
                features[i, metadata["value_to_index"][variable][value]] = 1.0

        # the class needs to be transformed
        labels[i] = int(row["default payment next month"].replace(".0", ""))

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(features)

    assert i == metadata["num_samples"] - 1

    num_positive_samples = int(labels.sum())
    num_negative_samples = labels.shape[0] - num_positive_samples

    print("Negative samples: ", num_negative_samples)
    print("Positive samples: ", num_positive_samples)
    print("Total samples: ", features.shape[0])
    print("Features: ", features.shape[1])

    np.save(features_path, features)
    np.save(labels_path, labels)

    input_file.close()

    metadata["features_min"] = scaler.data_min_.tolist()
    metadata["features_max"] = scaler.data_max_.tolist()

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)


def main(args=None):
    options_parser = argparse.ArgumentParser(
        description="Transform the Default of Credit Card Clients data into feature matrices."
                    + " Dataset: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
    )

    options_parser.add_argument("input", type=str, help="Input Default Credit Card data in text format.")
    options_parser.add_argument("features", type=str, help="Output features in numpy array format.")
    options_parser.add_argument("labels", type=str, help="Output labels in numpy array format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")

    options = options_parser.parse_args(args=args)

    default_credit_card_transform(options.input, options.features, options.labels, options.metadata)


if __name__ == "__main__":
    main()
