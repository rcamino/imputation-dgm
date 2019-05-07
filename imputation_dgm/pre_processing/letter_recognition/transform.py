from __future__ import print_function

import argparse
import json

import numpy as np

from imputation_dgm.pre_processing.metadata import create_metadata, create_class_to_index, create_one_type_dictionary

from sklearn.preprocessing.data import MinMaxScaler


VARIABLES = [
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]

NUM_SAMPLES = [
    789,  # A
    766,  # B
    736,  # C
    805,  # D
    768,  # E
    775,  # F
    773,  # G
    734,  # H
    755,  # I
    747,  # J
    739,  # K
    761,  # L
    792,  # M
    783,  # N
    753,  # O
    803,  # P
    783,  # Q
    758,  # R
    748,  # S
    796,  # T
    813,  # U
    764,  # V
    752,  # W
    787,  # X
    786,  # Y
    734,  # Z
]

CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

CLASS_TO_INDEX = create_class_to_index(CLASSES)


def letter_recognition_transform(input_path, features_path, labels_path, metadata_path):
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

        for j, value in enumerate(values[1:]):
            value = float(value)
            features[i, j] = value

        labels[i] = CLASS_TO_INDEX[values[0]]

        i += 1

        line = input_file.readline()

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(features)

    assert i == metadata["num_samples"]

    for class_index in range(len(NUM_SAMPLES)):
        num_samples_class = (labels == class_index).sum()
        assert num_samples_class == NUM_SAMPLES[class_index]

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
        description="Transform the Letter Recognition data into feature matrices."
                    + " Dataset: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition"
    )

    options_parser.add_argument("input", type=str, help="Input Letter Recognition data in text format.")
    options_parser.add_argument("features", type=str, help="Output features in numpy array format.")
    options_parser.add_argument("labels", type=str, help="Output labels in numpy array format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")

    options = options_parser.parse_args(args=args)

    letter_recognition_transform(options.input, options.features, options.labels, options.metadata)


if __name__ == "__main__":
    main()
