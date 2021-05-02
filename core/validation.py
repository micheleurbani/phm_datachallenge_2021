
from sklearn.model_selection import train_test_split


def cross_validation(classifier, data):
    """
    Perform cross-validation of the classifier using the provided data.

    Parameters
    ----------
    classifier : a classifier object
        The classifier must implement the methods fit, transfor, and
        fit_transform.
    data
        Either a :class:`pandas.DataFrame` or a :class:`numpy.array`

    Returns
    -------
    h : scalar
        The bandwidth parameter of the classifier.

    """
    # Split the dataset into training and validation datasets
    train, val = train_test_split(data, test_size=0.3)
    # Test the value of h
    clf = classifier(h=5)
    Y, Y_hat = clf.predict(train, val)
    # Compute the mean square error of the prediction
    mse = clf.score(Y, Y_hat)
    print(mse)
