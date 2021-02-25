
from csv import reader
from ast import literal_eval


class DataHandler:
    """
    The class parses the csv files, provides easy access to data and utilities
    to hadle datasets.

    :param str file_path: the path to the file containig data.
    :return: an instance of the object (which can be transformed into a
    :class:`pandas.DataFrame`.

    """

    def __init__(self, file_name):
        self.f_path = file_name
        self.error_code = self.read_error_class()
        self.data = self.parse()

    def read(self):
        """
        Returns a list of lists, which contains the data to be parsed.
        The first row of the csv file is cut out.
        """
        with open(self.f_path, "r") as csv_file:
            raw_data = reader(csv_file)
            data = [row for row in raw_data]
        return data[1:]

    def read_error_class(self):
        """
        Returns the error class from the name of the file.
        """
        try:
            start = self.f_path.find("_", self.f_path.find("class"))
            end = self.f_path.find("_", start + 1)
            return int(self.f_path[start + 1:end])
        except:
            raise ValueError

    def parse(self):
        """
        Returns a dictionary containg the field and the corresponding
        timeseries.
        """
        raw_data = self.read()
        # Parse the raw data and return a dictionary.
        data = {row[0]: literal_eval(row[1].replace('nan', '"nan"'))
                for row in raw_data}
        return data
