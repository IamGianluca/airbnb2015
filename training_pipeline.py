"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd
from sklearn import cross_validation


def prepare_dataset():

    # set variables
    path = "./data/"
    train_file = "training_features.csv"
    train_full_path = "".join((path, train_file))
    test_file = "test_features.csv"
    test_full_path = "".join((path, test_file))

    # load training data set
    train = pd.read_csv(train_full_path)
    test = pd.read_csv(test_full_path)

    # TODO: make sure training and test sets have the same number of features (this must be checked in the featurizer flow)
    # remove features in test set which are not present in train set
    not_in_train = [feature for feature in test.columns if feature not in train.columns]
    test = test.drop(not_in_train, axis=1)

    # add features missing
    not_in_test = [feature for feature in train.columns if feature not in test.columns]
    not_in_test.remove("country_destination")
    train = train.drop(not_in_test, axis=1)

    # separate outcome from independent variables
    y_train = train.country_destination.tolist()
    X_train = train.drop(["id", "country_destination"], axis=1).as_matrix()
    ids_test = test.id.tolist()
    X_test = test.drop(["id"], axis=1).as_matrix()
    del train, test

    # feature selection; remove all features that are either one or zero (on or off) in more than 80% of the samples
    from sklearn.feature_selection import VarianceThreshold

    print("Training set size before feature selection:", X_train.shape)
    print("Test set size before feature selection:", X_test.shape)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_train = sel.fit_transform(X_train, y_train)
    X_test = sel.transform(X_test)
    print("Training set size after feature selection:", X_train.shape)

    return X_train, X_test, y_train, ids_test


def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    """
    We want to fit 12 logistic regression models to predict each possible outcome: 'US', 'FR', 'CA', 'GB', 'ES', 'IT',
      'PT', 'NL','DE', 'AU', 'NDF' and 'other'. The general idea is that we want to assess which are the 5 more likely
      outcomes given the information we have on each user. This will constitute our naive solution for 2nd, 3rd, 4th
      and 5th guesses. We will then improve the 1st prediction using other non-linear classifiers (decision trees,
      Random Forest, XGBoost, KNN, NN, etc..)
    """
    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf = lr.fit(X_train, y_train)

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_decision_tree(X_train, y_train):

    from sklearn import tree

    # train decision trees
    # TODO: generally max_depth=3 is a good starting point. Assess if different value could lead to a better CV score
    tree = tree.DecisionTreeClassifier(max_depth=3)
    clf = tree.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_random_forest(X_train, y_train):

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
    clf = clf.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf