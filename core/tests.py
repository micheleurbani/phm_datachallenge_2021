
import os
import unittest
import numpy as np
from random import choice, seed
from itertools import chain
from core.data_handler import (
    read,
    parse,
    sanity_checks,
    read_dataset,
    load_training_dataset,
)
from core.aakr import AAKR
from sklearn.feature_selection import VarianceThreshold


seed("294845")


class TestReadDataset(unittest.TestCase):

    def setUp(self):
        folder = "training_validation_1"
        datasets = os.listdir(folder)
        self.file_path = os.path.join(folder, choice(datasets))

    def test_read(self):
        X = read(self.file_path)
        for signal in X:
            self.assertIsInstance(signal, list)
            for i in signal:
                self.assertIsInstance(i, str)

    def test_parse(self):
        X = parse(read(self.file_path))
        self.assertIsInstance(X, dict)
        for k, v in X.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, list)
        self.assertEqual(len(X.keys()), 50)

    def test_sanity_checks(self):
        X = sanity_checks(parse(read(self.file_path)))
        self.assertIsInstance(X, dict)
        # Check the length of the timeseries
        self.assertEqual(len(set(chain(*[[len(f) for f in s] for s
                         in X.values()]))), 1)
        # Check the number of features
        self.assertEqual(
            sum([len(i) for i in X.values()]),
            247
        )

    def test_read_dataset(self):
        X = read_dataset(self.file_path)
        self.assertEqual(len(list(X.columns)), 247)

    def test_load_training_dataset(self):
        X = load_training_dataset(percent_data=0.3)
        self.assertEqual(len(list(X.columns)), 247)
        self.assertGreaterEqual(len(X), 1)


class TestAAKR(unittest.TestCase):

    def setUp(self):
        self.X = load_training_dataset(percent_data=0.2)
        self.aakr = AAKR()

    def test_fit(self):

        np.set_printoptions(threshold=1000)
        # Drop columns with missing values
        X = self.X.dropna(axis=1)
        # Pre-processing: drop features with almost zero variance
        f_sel = VarianceThreshold(threshold=0.01)
        X = f_sel.fit_transform(X.to_numpy())
        self.aakr.predict(X, np.expand_dims(X[2,:], axis=0))
        self.aakr.predict(X, X[2:5,:])

