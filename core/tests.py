
import unittest
import numpy as np
import pandas as pd
from .data_handler import DataHandler, load_training_dataset, assess_NA
from .aakr import AAKR


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


class TestAAKR(unittest.TestCase):

    def setUp(self):
        training_dataset = load_training_dataset().dropna(axis=1).head(5000).\
                           to_numpy()
        self.aakr = AAKR(training_data=training_dataset)

    def test_covariance(self):

        print(np.cov(self.aakr.X))
