
import numpy as np
from sklearn.model_selection import train_test_split


def cross_validation_score(classifier, data, cv, h):
    """
    Perform cross-validation of the classifier using the provided data.
    The function detects `h` parameters that yields numerical errors and
    returns a score of 0.0 for that cross validation cycle.

    Parameters
    ----------
    classifier : a classifier object
        The classifier must implement the methods fit, transfor, and
        fit_transform.
    data : :class:`pandas.DataFrame` or :class:`numpy.array`
        The training data used to cross-validate the estimator.
    cv : scalar
        The number of cross-validation folds used per each bandwidth paramter.
    h: scalar
        The bandwidth parameters `h` to be cross-validated.

    Returns
    -------
    scores : :class:`numpy.array`
        The vector containing the scores obtained with each bandwidth
        parameter.
    """
    assert type(cv) is int
    # Start testing the bandwidth paramters
    scores = []
    clf = classifier(h=h)
    clf.fit(data)
    data, _ = clf.transform(data)
    # Use the 5-fold procedure
    for i in range(cv):
        # Split the dataset into training and validation datasets
        train, val = train_test_split(data, test_size=0.3, random_state=i)
        # ZeroDivisionError is possible if `h` is too small.
        try:
            # Test the value of h
            Y, Y_hat = clf.predict(train, val)
            # Compute the mean square error of the prediction
            scores.append(clf.score(Y, Y_hat))
        # When `h` is too small, returns 0.0 score to lower the average score
        except ZeroDivisionError:
            scores.append(0.0)

    return scores

def grid_search(classifier, values, cv):
    """
    Perform a grid search over the parameters provided.

    Parameters
    ----------
    classifier : a classifier objecet
        Either a :class:`core.AAKR` or :class:`core.ModifiedAAKR`.
    values : dict
        The dictionary containing the parameter names as keywords and the
        values to be tested (provided within a list) as values.
    cv : scalar
        The number of folds performed during the cross validation process.

    Returns
    -------
    scores : a pandas DataFrame
        The average scores obtained with combination of paramters.
    """
    # Check if the hyper-parameters belongs to the classifier
    clf = classifier()
    for param in values:
        try:
            getattr(clf, param)
        except AttributeError:
            raise AttributeError(f"The classifier has no '{param}' attribute.")
    # Create a list of tuples with all the combinations of hyper-parameters
    
