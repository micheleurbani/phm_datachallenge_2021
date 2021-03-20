
import os
import sys
import pandas as pd
from tqdm import tqdm
from csv import reader, field_size_limit
from ast import literal_eval


field_size_limit(sys.maxsize)


def read(file_path):
    """
    Read the csv file.

    Parameters
    ----------
    file_path : str
        The path to the file complete of folder name.

    Returns
    -------
    data : list
        The list of rows in the file as strings.
    """
    with open(file_path, "r", newline="") as csv_file:
        raw_data = reader(csv_file)
        data = [row for row in raw_data]
    return data[1:]


def parse(raw_data):
    """
    Parse the list of strings containing raw data from csv by
    `ast.literal_eval`. The `nan` values are replaced with strings (`NaN`).

    Parameters
    ----------
    data : list
        The list of string containig raw data.

    Returns
    -------
    data : dict
        A dictionary containing signals' names as keyworkds and a list of
        tuples as values. Each list contains some tuples corresponding to
        signal's features.

    """
    # Parse the raw data and return a dictionary.
    data = {row[0]: literal_eval(row[1].replace('nan', '"NaN"'))
            for row in raw_data}
    # Transform each signal in a list of tuples, each containing a
    # timeseries.
    for k, v in data.items():
        data[k] = [i for i in zip(*v)]

    return data


def sanity_checks(data):
    """
    The followign sanity checks are performed:

        1) Check the length of each timeseris and cut them to the shortes.
        2) Check the number of fetures: if vCnt of the signal
        NumberFuseDetected is not present, then add a column containing
        `np.nan` objects for compliance of indexing.

    Parameters
    ----------
    data : dict
        The dicts obtained from parsing of raw data using `parse()`.

    Returns
    -------
    data : dict
        The dictionary containing "sanitized" data.

    """
    min_len = min([min([len(i) for i in v]) for v in data.values()])
    # Check the length of timeseries
    for signal, features in data.items():
        for i, _ in enumerate(features):
            data[signal][i] = data[signal][i][:min_len]
    # Check for the existence of vCnt feature for the signal NumberFuseDetected
    if sum([len(i) for i in data.values()]) == 246:
        # The feature vFreq is located in position 1 in the structure of
        # features
        data["NumberFuseDetected"].insert(1, tuple(["NaN" for _ in
                                                    range(min_len)]))
    return data


def read_indices():
    # Load column names and define indexing
    with open("training_validation_2/fields.csv", "r") as csvfile:
        csv_reader = reader(csvfile)
        csv_reader = [i for i in csv_reader]
        indices = {i[0]: [j for j in i[1:] if j] for i in csv_reader[1:]}
    return indices


def read_dataset(file_path):
    """
    Read and parse a dataset of the competion.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    df : pandas DataFrame
        The dataframe complete of MultiIndex.
    """
    # Read raw data from csv.
    X = read(file_path=file_path)
    # Parse row data.
    X = parse(X)
    # Performa sanity checks
    X = sanity_checks(X)
    # Add subindexes
    array, indices = {}, read_indices()
    for signal in X:
        for i, feature in enumerate(X[signal]):
            array[(signal, indices[signal][i])] = feature
    # Generate pd.MultiIndex
    index = pd.MultiIndex.from_tuples(
        list(array.keys()),
        names=["signal", "feature"]
    )
    # Force all data to be float or NaN if inference to float does not work
    df = pd.DataFrame(
        data=array,
        dtype="float",
        columns=index,
    )
    return df
