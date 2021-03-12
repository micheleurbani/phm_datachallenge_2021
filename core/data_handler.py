
import os
import sys
import pandas as pd
from csv import reader, field_size_limit
from ast import literal_eval


field_size_limit(sys.maxsize)


class DataHandler():
    """
    The class parses the csv files and provides easy access to data and
    utilities to hadle datasets.

    :param str file_path: the path to the file containig data.
    :return: an instance of the object (which can be transformed into a
    :class:`pandas.DataFrame`.

    """
    def __init__(self, file_name):
        self.f_path = file_name
        self.error_code = self.read_error_class()
        self.indices = self.read_indices()
        self.data = self.parse()

    def read(self):
        """
        Returns a list of lists, which contains the data to be parsed.
        The first row of the csv file is cut out.
        """
        with open(self.f_path, "r", newline="") as csv_file:
            raw_data = reader(csv_file)
            data = [row for row in raw_data]
        return data[1:]

    @staticmethod
    def read_indices():
        # Load column names and define indexing
        with open("training_validation_2/fields.csv", "r") as csvfile:
            csv_reader = reader(csvfile)
            csv_reader = [i for i in csv_reader]
            indices = {i[0]: [j for j in i[1:] if j] for i in csv_reader[1:]}
        return indices

    def read_error_class(self):
        """
        Returns the error class from the name of the file.
        """
        start = self.f_path.find("_", self.f_path.find("class"))
        end = self.f_path.find("_", start + 1)
        return int(self.f_path[start + 1:end])

    def parse(self):
        """
        Returns a :class:`pandas.DataFrame` with multi-indexing divided by
        signals and features.
        """
        raw_data = self.read()
        # Parse the raw data and return a dictionary.
        data = {row[0]: literal_eval(row[1].replace('nan', '"NaN"'))
                for row in raw_data}
        # Transform each signal in a list of tuples, each containing a
        # timeseries.
        for k, v in data.items():
            data[k] = [i for i in zip(*v)]
        # Create subindexes
        array = {}
        for signal in data:
            for i, feature in enumerate(data[signal]):
                array[(signal, self.indices[signal][i])] = feature
        # Check the length of the timeseries and cut them to the length of the
        # shortest
        min_len = min([len(v) for v in array.values()])
        for k, v in array.items():
            array[k] = v[:min_len]
        # Force all data to be float or NaN if inference to float does not work
        df = pd.DataFrame(array, dtype="float")
        return df


def load_training_dataset():
    """
    Utility function that returns a numpy array containing all the data in the
    `training_validation_1` folder.
    """
    folder = "training_validation_1"
    file_names = os.listdir(folder)
    X = []
    for fname in file_names:
        try:
            x = DataHandler(os.path.join(folder, fname)).data
            X.append(x)
        except ValueError as e:
            print("During processigng of", fname,
                  "the following error occured:")
            print(e)
    return pd.concat(X)


def assess_NA(data):
    """
    Returns a pandas dataframe denoting the total number of NA values and the
    percentage of NA values in each column.
    The column names are noted on the index.

    Parameters
    ----------
    data: dataframe
    """
    # pandas series denoting features and the sum of their null values
    null_sum = data.isnull().sum()  # instantiate columns for missing data
    total = null_sum.sort_values(ascending=False)
    percent = (((null_sum / len(data.index))*100).round(2))\
        .sort_values(ascending=False)

    # concatenate along the columns to create the complete dataframe
    df_NA = pd.concat([total, percent], axis=1,
                      keys=['Number of NA', 'Percent NA'])

    # drop rows that don't have any missing data; omit if you want to keep all
    # rows
    df_NA = df_NA[(df_NA.T != 0).any()]

    return df_NA
