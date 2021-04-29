
import os
import unittest
import numpy as np
from random import choice, seed, sample
from itertools import chain
from core.data_handler import (
    read,
    parse,
    sanity_checks,
    read_dataset,
    load_training_dataset,
)
from core.aakr import AAKR, ModifiedAAKR


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
        # Select a random dataset containing failures
        self.Y = read_dataset(
            os.path.join(
                "training_validation_2",
                *sample(os.listdir("training_validation_2"), 1)
            )
        )
        self.aakr = AAKR()

    def test_fit(self):
        self.aakr.fit(self.X, self.Y)
        self.assertIsNotNone(self.aakr.pipe)
        self.assertIsNotNone(self.aakr.features)

    def test_transform(self):
        self.aakr.fit(self.X, self.Y)
        X, Y = self.aakr.transform(self.X, self.Y)
        self.assertTrue(type(X) is np.ndarray)
        self.assertTrue(type(Y) is np.ndarray)
        self.assertEqual(self.X.shape[0], len(X))
        self.assertEqual(self.Y.shape[0], len(Y))
        self.assertGreaterEqual(self.X.shape[1], X.shape[1])
        self.assertGreaterEqual(self.Y.shape[1], Y.shape[1])

    def test_predict(self):
        Y, Y_hat = self.aakr.predict(self.X, self.Y)
        self.assertIsNotNone(self.aakr.VI)

#     def test_features_retrieval(self):
#         Y = read_dataset("training_validation_2/class_0_101_data.csv")
#         Y = self.aakr.fit(Y)
#         mask = self.aakr.pipe.named_steps["variancethreshold"].get_support()
#         self.assertEqual(len(mask), Y.shape[1])


# class TestModifiedAAKR(unittest.TestCase):

#     def setUp(self):
#         self.X = load_training_dataset(percent_data=0.2)
#         self.aakr = ModifiedAAKR()
#         self.Y = read_dataset("training_validation_2/class_0_101_data.csv")

#     def test_abs_normalized_distance(self):
#         X, Y = self.aakr.fit_transform(self.X, self.Y)
#         dist_shape = self.aakr.abs_normalized_distance(X, Y).shape
#         self.assertEqual(
#             dist_shape,
#             (Y.shape[0], self.X.shape[0], Y.shape[1])
#         )

#     def test_permutation_matrix(self):
#         X, Y = self.aakr.fit_transform(self.X, self.Y)
#         rnd_obs = Y[np.random.randint(Y.shape[0]), :]
#         dist = self.aakr.abs_normalized_distance(X, rnd_obs)
#         P = self.aakr.permutation_matrix(dist)
#         p_rnd_obs = np.dot(rnd_obs, P)
#         self.assertGreaterEqual(p_rnd_obs[1], p_rnd_obs[-1])


# class TestTraining(unittest.TestCase):

#     def setUp(self):
#         self.X = load_training_dataset(percent_data=0.2)
#         self.aakr = ModifiedAAKR
#         self.Y = read_dataset("training_validation_2/class_0_101_data.csv")

#     def test_training(self):
#         cross_validation(
#             classifier=self.aakr,
#             data=self.X,
#         )
