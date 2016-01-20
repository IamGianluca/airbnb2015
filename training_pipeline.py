"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd
from sklearn import cross_validation
import numpy as np
import time


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
    X_train_full = train.drop(["id", "country_destination"], axis=1).as_matrix()
    ids_test = test.id.tolist()
    X_test = test.drop(["id"], axis=1).as_matrix()

    # feature selection; remove all features that are either one or zero (on or off) in more than 80% of the samples
    from sklearn.feature_selection import VarianceThreshold

    print("Training set size before feature selection:", X_train_full.shape)
    print("Test set size before feature selection:", X_test.shape)

    # TODO: try less naive approaches to do features selection
    # feature selection
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_train = sel.fit_transform(X_train_full, y_train)
    X_test = sel.transform(X_test)

    # keep track of the features remaining after the feature selection step
    idxs = sel.get_support(indices=False)
    selected_features = train.drop(["id", "country_destination"], axis=1).columns[idxs].values

    del train, test

    print("Training set size after feature selection:", X_train.shape)
    print("Selected features are:", selected_features)

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
    tree = tree.DecisionTreeClassifier(max_depth=3, random_state=23)
    clf = tree.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_random_forest(X_train, y_train):

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=2013, min_samples_leaf=50,
                                 oob_score=True)
    clf = clf.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_svm(X_train, y_train):

    from sklearn.svm import SVC
    from sklearn import preprocessing

    # scale and centre variables as Support Vector Machine algorithms are not scale invariant
    # sample = np.random.choice(X_train.shape[0], 5000, replace=False)
    X_train_c = preprocessing.scale(X_train[:20000, :])
    y_train = y_train[:20000]

    # TODO: try Stochastic Gradient Descent (SGD)
    # The advantages of Stochastic Gradient Descent are:
    # - Efficiency.
    # - Ease of implementation (lots of opportunities for code tuning).
    # The disadvantages of Stochastic Gradient Descent include:
    # - SGD requires a number of hyper-parameters such as the regularization parameter and the number of iterations.
    # - SGD is sensitive to feature scaling.

    # fit svm
    # we tried also a linear kernel but it was too slow and didn't show any performance improvement
    svc = SVC(kernel="rbf", probability=True, decision_function_shape="ovr", shrinking=True, random_state=16)
    svc = svc.fit(X_train_c, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(svc, X_train_c, y_train, cv=5)
    print("RBF with Shrinkage Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return svc


def train_knn(X_train, y_train):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import preprocessing

    # scale and centre independent variables
    # sample = np.random.choice(X_train.shape[0], 5000, replace=False)
    X_train_c = preprocessing.scale(X_train[:20000, :])
    y_train = y_train[:20000]

    # fit knn
    knn = KNeighborsClassifier(n_neighbors=12, n_jobs=-1)
    knn = knn.fit(X_train_c, y_train)

    # cross-validation
    # scores = cross_validation.cross_val_score(knn, X_train_c, y_train, cv=5)
    # print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return knn


def train_neural_net(X_train, y_train):

    # sigmoid function
    def nonlin(x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # input dataset
    X = np.array(X_train)

    # output dataset
    y_dummies = pd.get_dummies(y_train)
    y = np.array(y_dummies).T

    # seed random numbers to make calculation deterministic (just a good practice)
    np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2 * np.random.random((92, 12)) - 1

    for iter in range(10000):

        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))

        # how much did we miss?
        l1_error = y.T - l1

        # multiply how much we missed by the slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1, True)

        # update weights
        syn0 += np.dot(l0.T, l1_delta)

    print("Output After Training:")
    print(l1)

    return l1



