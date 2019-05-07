from __future__ import print_function

import argparse
import json

import numpy as np

from imputation_dgm.pre_processing.metadata import create_metadata, create_one_type_dictionary

from sklearn.preprocessing.data import MinMaxScaler


VARIABLES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]

NUM_SAMPLES = [
    2788,  # 0 = negative = not-spam
    1813,  # 1 = positive = spam
]

CLASSES = [
    "not-spam",
    "spam"
]


def spambase_transform(input_path, features_path, labels_path, metadata_path):
    metadata = create_metadata(VARIABLES,
                               create_one_type_dictionary("numerical", VARIABLES),
                               {},
                               sum(NUM_SAMPLES),
                               CLASSES)

    input_file = open(input_path, "r")

    features = np.zeros((metadata["num_samples"], metadata["num_features"]), dtype=np.float32)
    labels = np.zeros(metadata["num_samples"], dtype=np.int32)

    # transform
    i = 0
    line = input_file.readline()
    while line != "":
        line = line.rstrip("\n")
        values = line.split(",")

        assert len(values) - 1 == len(VARIABLES), str((len(values) - 1, len(VARIABLES)))

        for j, value in enumerate(values[:-1]):
            value = float(value)
            features[i, j] = value

        labels[i] = int(values[-1])

        i += 1

        line = input_file.readline()

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(features)

    assert i == metadata["num_samples"]

    num_positive_samples = int(labels.sum())
    num_negative_samples = labels.shape[0] - num_positive_samples

    assert num_negative_samples == NUM_SAMPLES[0]
    assert num_positive_samples == NUM_SAMPLES[1]

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
        description="Transform the Spambase data into feature matrices."
                    + " Dataset: https://archive.ics.uci.edu/ml/datasets/spambase"
    )

    options_parser.add_argument("input", type=str, help="Input Spambase data in text format.")
    options_parser.add_argument("features", type=str, help="Output features in numpy array format.")
    options_parser.add_argument("labels", type=str, help="Output labels in numpy array format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")

    options = options_parser.parse_args(args=args)

    spambase_transform(options.input, options.features, options.labels, options.metadata)


if __name__ == "__main__":
    main()
