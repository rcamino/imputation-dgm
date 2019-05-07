import json

import numpy as np


def load_variable_sizes_from_metadata(metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata["variable_sizes"]


def separate_variable(data, variable_sizes, selected_index):
    if selected_index == 0:
        features = data[:, variable_sizes[selected_index]:]
        labels = data[:, :variable_sizes[selected_index]]
    elif 0 < selected_index < len(variable_sizes) - 1:
        left_size = sum(variable_sizes[:selected_index])
        left = data[:, :left_size]
        labels = data[:, left_size:left_size + variable_sizes[selected_index]]
        right = data[:, left_size + variable_sizes[selected_index]:]
        features = np.concatenate((left, right), axis=1)
    else:
        left_size = sum(variable_sizes[:-1])
        features = data[:, :left_size]
        labels = data[:, left_size:]

    assert data.shape[1] == features.shape[1] + labels.shape[1]

    if variable_sizes[selected_index] > 1:
        labels = np.argmax(labels, axis=1)
    else:
        labels = labels.reshape(-1)

    return features, labels


def variable_start_features(variable_sizes):
    indices = []
    j = 0
    for variable_size in variable_sizes:
        indices.append(j)
        j += variable_size
    return indices


def round_categorical_variables(features, variable_sizes=None):
    if variable_sizes is None:
        return np.round(features)
    else:
        assert features.shape[1] == sum(variable_sizes)
        rounded = np.zeros_like(features)
        start = 0
        for variable_size in variable_sizes:
            if variable_size > 1:
                end = start + variable_size
                rounded[:, start:end] = np.round(features[:, start:end])
                start = end
        return rounded
