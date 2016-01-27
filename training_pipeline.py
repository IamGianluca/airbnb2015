"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd
from sklearn import cross_validation


def prepare_dataset(t):

    # set variables
    path = "./data/"
    train_file = "training_features.csv"
    train_full_path = "".join((path, train_file))
    test_file = "test_features.csv"
    test_full_path = "".join((path, test_file))

    # load training data set
    train = pd.read_csv(train_full_path)
    test = pd.read_csv(test_full_path)

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
    X_test_full = test.drop(["id"], axis=1).as_matrix()

    # feature selection; remove all features that are either one or zero (on or off) in more than 80% of the samples
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

    print("Training set size before feature selection:", X_train_full.shape)
    print("Test set size before feature selection:", X_test_full.shape)

    # TODO: try less naive approaches to do features selection
    """
    Using Grid Search we found the best Variance Threshold to be 80-85%. Values above of below result in performance
    loss. We tested the following values [.50, .60, .70, .80, .85, .90]. The best performance run resulted in the
    following accuracies value:

    Logistic Regression Accuracy: 0.62 (+/- 0.01)
    Decision Tree Accuracy: 0.66 (+/- 0.01)
    Random Forest Accuracy: 0.65 (+/- 0.03)



    I should read a little bit more into how to perform feature selection before training, in particular the Strong Rule
    approach proposed by Robert Tibshirani et al. (http://tinyurl.com/hu8lzxn). Chi-square, even though it's a more
    sophisticated method, doesn't improve performance compared to a variance threshold method. This is to me a bit
    suspicious.
    """
    # feature selection
    # sel = VarianceThreshold(threshold=(t * (1 - t)))
    sel = SelectKBest(chi2, k=t)
    X_train = sel.fit_transform(X_train_full, y_train)
    X_test = sel.transform(X_test_full)

    # keep track of the features remaining after the feature selection step
    idxs = sel.get_support(indices=False)
    selected_features = train.drop(["id", "country_destination"], axis=1).columns[idxs].values

    del train, test, X_train_full, X_test_full

    print("Number of features included:", t)
    print("Training set size after feature selection:", X_train.shape)
    print("Selected features are:", selected_features)

    return X_train, X_test, y_train, ids_test


def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf = lr.fit(X_train, y_train)

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_decision_tree(X_train, y_train):

    from sklearn import tree

    # train decision trees
    # TODO: generally max_depth=3 is a good starting point. Assess if different value could lead to a better CV score
    tree = tree.DecisionTreeClassifier(max_depth=3, random_state=23)
    clf = tree.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def train_random_forest(X_train, y_train, msl=50):

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=2013, min_samples_leaf=msl,
                                 oob_score=True)
    clf = clf.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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



