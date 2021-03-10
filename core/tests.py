
import unittest
import pandas as pd
from .data_handler import DataHandler, load_training_dataset, assess_NA


class TestDataHandler(unittest.TestCase):

    def setUp(self):
        self.data = DataHandler("training_validation_1/class_0_0_data.csv")

    def test_read_error_class(self):
        self.assertIsInstance(self.data.error_code, int)

    def test_classes(self):
        self.data = DataHandler("training_validation_2/class_0_101_data.csv")


class TestTrainingDataloader(unittest.TestCase):

    def test_data_loader(self):
        X = load_training_dataset()
        print(X.info())
        na_df = assess_NA(X)
        print(na_df)
