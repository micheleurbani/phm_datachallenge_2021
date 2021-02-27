
import unittest
import pandas as pd
from .data_handler import DataHandler


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        self.data = DataHandler("training_validation_1/class_0_0_data.csv")

    def test_read_error_class(self):
        self.assertIsInstance(self.data.error_code, int)

    def test_parse(self):
        parsed_data = self.data.parse()
        for i in parsed_data:
            self.assertIsInstance(i, str)
            self.assertIsInstance(parsed_data[i], list)

    def test_data_frame(self):
        df = pd.DataFrame(self.data.data)
        print(self.data.indices)
