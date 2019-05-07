def create_one_type_dictionary(variable_type, variables):
    return dict([(variable, variable_type) for variable in variables])


def types_to_sorted_lists(variable_types, variables=None):
    if variables is None:
        variables = variable_types.keys()

    binary_variables = []
    categorical_variables = []
    numerical_variables = []
    for variable in variables:
        variable_type = variable_types[variable]
        if variable_type == "categorical":
            categorical_variables.append(variable)
        elif variable_type == "binary":
            binary_variables.append(variable)
        elif variable_type == "numerical":
            numerical_variables.append(variable)
        else:
            raise Exception("Invalid type: '{}'.".format(variable_type))

    binary_variables = sorted(binary_variables)
    categorical_variables = sorted(categorical_variables)
    numerical_variables = sorted(numerical_variables)
    return binary_variables, categorical_variables, numerical_variables


def create_metadata(variables, variable_types, categorical_values={}, num_samples=None, classes=None):
    binary_variables, categorical_variables, numerical_variables = types_to_sorted_lists(variable_types, variables)

    feature_number = 0
    value_to_index = {}
    index_to_value = []
    variable_sizes = []
    variable_types = []

    for variable in categorical_variables:
        variable_types.append("categorical")
        values = sorted(categorical_values[variable])
        variable_sizes.append(len(values))
        value_to_index[variable] = {}
        for value in values:
            index_to_value.append((variable, value))
            value_to_index[variable][value] = feature_number
            feature_number += 1

    for variable in binary_variables:
        variable_types.append("categorical")
        values = [0, 1]
        variable_sizes.append(2)
        value_to_index[variable] = {}
        for value in values:
            index_to_value.append((variable, value))
            value_to_index[variable][value] = feature_number
            feature_number += 1

    for variable in numerical_variables:
        variable_types.append("numerical")
        variable_sizes.append(1)
        value_to_index[variable] = feature_number
        feature_number += 1

    num_features = feature_number

    metadata = {
        "variables": binary_variables + categorical_variables + numerical_variables,
        "variable_sizes": variable_sizes,
        "variable_types": variable_types,
        "index_to_value": index_to_value,
        "value_to_index": value_to_index,
        "num_features": num_features,
    }

    if num_samples is not None:
        metadata["num_samples"] = num_samples

    if classes is not None:
        metadata["classes"] = classes

    return metadata


def create_class_to_index(classes):
    return dict([(c, i) for i, c in enumerate(classes)])
