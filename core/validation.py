
import numpy as np
from sklearn.model_selection import train_test_split


def cross_validation(classifier, data):
    # Preprocess data
    clf = classifier()
    clf.fit(data)
    X = clf.transform(data)
    # Split the dataset into training and validation datasets
    train, val = train_test_split(X, test_size=0.3)
    print(train)
    print(val)
    # Test the value of h
    Y, Y_hat = clf.predict(train, val)
    # Compute the mean square error of the prediction
    MSE = (Y - Y_hat)**2 / val.shape[0]