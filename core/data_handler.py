
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
            # indices = [(i, j) for i in indices for j in indices[i]]
            # indices = pd.MultiIndex.from_tuples(indices,
                                                # names=["Signal", "Feature"])
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
        data = {row[0]: literal_eval(row[1].replace('nan', '"nan"'))
                for row in raw_data}
        # Transform each signal in a list of tuples, each containing a
        # timeseries.
        for k, v in data.items():
            data[k] = [i for i in zip(*v)]
        array = {}
        for signal in data:
            for i, feature in enumerate(data[signal]):
                array[(signal, self.indices[signal][i])] = feature
        df = pd.DataFrame(array)
        return df
