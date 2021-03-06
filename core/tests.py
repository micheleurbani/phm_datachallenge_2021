
import unittest
import pandas as pd
from .data_handler import DataHandler


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        self.data = DataHandler("training_validation_1/class_0_0_data.csv")

    def test_read_error_class(self):
        self.assertIsInstance(self.data.error_code, int)

    def test_classes(self):
        self.data = DataHandler("training_validation_2/class_0_101_data.csv")

