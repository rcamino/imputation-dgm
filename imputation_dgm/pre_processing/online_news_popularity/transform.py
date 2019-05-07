from __future__ import print_function

import argparse
import csv
import json

import numpy as np

from imputation_dgm.pre_processing.metadata import create_metadata

from sklearn.preprocessing.data import MinMaxScaler


NUM_SAMPLES = 39644

ORIGINAL_TYPES = {
    "n_tokens_title": "numerical",  # Number of words in the title
    "n_tokens_content": "numerical",  # Number of words in the content
    "n_unique_tokens": "numerical",  # Rate of unique words in the content
    "n_non_stop_words": "numerical",  # Rate of non-stop words in the content
    "n_non_stop_unique_tokens": "numerical",  # Rate of unique non-stop words in the content
    "num_hrefs": "numerical",  # Number of links
    "num_self_hrefs": "numerical",  # Number of links to other articles published by Mashable
    "num_imgs": "numerical",  # Number of images
    "num_videos": "numerical",  # Number of videos
    "average_token_length": "numerical",  # Average length of the words in the content
    "num_keywords": "numerical",  # Number of keywords in the metadata
    "data_channel_is_lifestyle": "categorical_split",  # Is data channel 'Lifestyle'?
    "data_channel_is_entertainment": "categorical_split",  # Is data channel 'Entertainment'?
    "data_channel_is_bus": "categorical_split",  # Is data channel 'Business'?
    "data_channel_is_socmed": "categorical_split",  # Is data channel 'Social Media'?
    "data_channel_is_tech": "categorical_split",  # Is data channel 'Tech'?
    "data_channel_is_world": "categorical_split",  # Is data channel 'World'?
    "kw_min_min": "numerical",  # Worst keyword (min. shares)
    "kw_max_min": "numerical",  # Worst keyword (max. shares)
    "kw_avg_min": "numerical",  # Worst keyword (avg. shares)
    "kw_min_max": "numerical",  # Best keyword (min. shares)
    "kw_max_max": "numerical",  # Best keyword (max. shares)
    "kw_avg_max": "numerical",  # Best keyword (avg. shares)
    "kw_min_avg": "numerical",  # Avg. keyword (min. shares)
    "kw_max_avg": "numerical",  # Avg. keyword (max. shares)
    "kw_avg_avg": "numerical",  # Avg. keyword (avg. shares)
    "self_reference_min_shares": "numerical",  # Min. shares of referenced articles in Mashable
    "self_reference_max_shares": "numerical",  # Max. shares of referenced articles in Mashable
    "self_reference_avg_sharess": "numerical",  # Avg. shares of referenced articles in Mashable
    "weekday_is_monday": "categorical_split",  # Was the article published on a Monday?
    "weekday_is_tuesday": "categorical_split",  # Was the article published on a Tuesday?
    "weekday_is_wednesday": "categorical_split",  # Was the article published on a Wednesday?
    "weekday_is_thursday": "categorical_split",  # Was the article published on a Thursday?
    "weekday_is_friday": "categorical_split",  # Was the article published on a Friday?
    "weekday_is_saturday": "categorical_split",  # Was the article published on a Saturday?
    "weekday_is_sunday": "categorical_split",  # Was the article published on a Sunday?
    "is_weekend": "binary",  # Was the article published on the weekend?
    "LDA_00": "numerical",  # Closeness to LDA topic 0
    "LDA_01": "numerical",  # Closeness to LDA topic 1
    "LDA_02": "numerical",  # Closeness to LDA topic 2
    "LDA_03": "numerical",  # Closeness to LDA topic 3
    "LDA_04": "numerical",  # Closeness to LDA topic 4
    "global_subjectivity": "numerical",  # Text subjectivity
    "global_sentiment_polarity": "numerical",  # Text sentiment polarity
    "global_rate_positive_words": "numerical",  # Rate of positive words in the content
    "global_rate_negative_words": "numerical",  # Rate of negative words in the content
    "rate_positive_words": "numerical",  # Rate of positive words among non-neutral tokens
    "rate_negative_words": "numerical",  # Rate of negative words among non-neutral tokens
    "avg_positive_polarity": "numerical",  # Avg. polarity of positive words
    "min_positive_polarity": "numerical",  # Min. polarity of positive words
    "max_positive_polarity": "numerical",  # Max. polarity of positive words
    "avg_negative_polarity": "numerical",  # Avg. polarity of negative  words
    "min_negative_polarity": "numerical",  # Min. polarity of negative  words
    "max_negative_polarity": "numerical",  # Max. polarity of negative  words
    "title_subjectivity": "numerical",  # Title subjectivity
    "title_sentiment_polarity": "numerical",  # Title polarity
    "abs_title_subjectivity": "numerical",  # Absolute subjectivity level
    "abs_title_sentiment_polarity": "numerical",  # Absolute polarity level
}

CAN_BE_EMPTY = {
    "data_channel": True,
    "weekday": False,
}


def read_binary(value):
    return int(float(value.strip()))


def online_news_popularity_transform(input_path, features_path, labels_path, metadata_path):
    variables = []
    types = {}
    values = {}

    for original_variable, original_type in ORIGINAL_TYPES.items():
        if "_is_" in original_variable:
            index = original_variable.index("_is_")
            variable = original_variable[:index]
            value = original_variable[index + 4:]
            if variable not in types:
                assert variable not in values
                types[variable] = "categorical"
                if CAN_BE_EMPTY[variable]:
                    values[variable] = ["none"]
                else:
                    values[variable] = []
                variables.append(variable)
            values[variable].append(value)
        else:
            variables.append(original_variable)
            types[original_variable] = original_type

    metadata = create_metadata(variables, types, values, NUM_SAMPLES)

    input_file = open(input_path, "r")
    reader = csv.DictReader(input_file)

    reader.fieldnames = [variable.strip() for variable in reader.fieldnames]

    features = np.zeros((metadata["num_samples"], metadata["num_features"]), dtype=np.float32)
    labels = np.zeros(metadata["num_samples"], dtype=np.float32)

    # transform
    for i, row in enumerate(reader):
        # the categorical variables are already one hot encoded
        for j, variable in enumerate(metadata["variables"]):
            if types[variable] == "numerical":
                value = float(row[variable])
                features[i, metadata["value_to_index"][variable]] = value
            elif types[variable] == "categorical":
                value = None
                for possible_value in values[variable]:
                    if possible_value == "none":
                        continue
                    real_variable = "{}_is_{}".format(variable, possible_value)
                    if read_binary(row[real_variable]) == 1:
                        if value is None:
                            value = possible_value
                        else:
                            raise Exception("'{}' was already defined".format(variable))
                if value is None:
                    if "none" in values[variable]:
                        value = "none"
                    else:
                        for possible_value in values[variable]:
                            if possible_value == "none":
                                continue
                            real_variable = "{}_is_{}".format(variable, possible_value)
                            print(possible_value, real_variable, read_binary(row[real_variable]))
                        raise Exception("'{}' has no valid value".format(variable))
                features[i, metadata["value_to_index"][variable][value]] = 1.0
            elif types[variable] == "binary":
                value = read_binary(row[variable])
                assert value in [0, 1], "'{}' is not a valid value for '{}'".format(value, variable)
                features[i, metadata["value_to_index"][variable][value]] = 1.0
            else:
                raise Exception("Unknown variable type.")

        labels[i] = row["shares"]

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit_transform(features)

    label_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    label_scaler.fit_transform(labels.reshape(-1, 1))

    assert i == metadata["num_samples"] - 1

    print("Total samples: ", features.shape[0])
    print("Features: ", features.shape[1])

    np.save(features_path, features)
    np.save(labels_path, labels)

    input_file.close()

    metadata["features_min"] = scaler.data_min_.tolist()
    metadata["features_max"] = scaler.data_max_.tolist()

    metadata["labels_min"] = label_scaler.data_min_.tolist()
    metadata["labels_max"] = label_scaler.data_max_.tolist()

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)


def main(args=None):
    options_parser = argparse.ArgumentParser(
        description="Transform the Online News Popularity data into feature matrices."
                    + " Dataset: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity"
    )

    options_parser.add_argument("input", type=str, help="Input Online News Popularity data in text format.")
    options_parser.add_argument("features", type=str, help="Output features in numpy array format.")
    options_parser.add_argument("labels", type=str, help="Output labels in numpy array format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")

    options = options_parser.parse_args(args=args)

    online_news_popularity_transform(options.input, options.features, options.labels, options.metadata)


if __name__ == "__main__":
    main()
