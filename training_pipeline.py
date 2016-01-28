"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd
from sklearn import cross_validation
import numpy as np
import math
import pickle


def ndcg_n(predictions, truth, k=5):
    # idcg = [1.0, 0.6309297535714575, 0.5, 0.43067655807339306, 0.38685280723454163]
    users = truth.id.unique()
    ndcg = []
    for user in users:
        print('user:', user)
        res = []
        guesses = predictions[predictions.user_id == user]['country'].reset_index(drop=True)
        observed = truth[truth.user_id == user]['country'].reset_index(drop=True)
        for i in range(0, k, 1):
            if guesses.loc[i] == observed.loc[0]:
                rel = 1
            else:
                rel = 0
            nom = math.pow(2, rel) - 1
            den = math.log2(i+2)
            inter = nom / den
            res.append(inter)
        ndcg.append(sum(res))
    return np.mean(ndcg)


def format_predictions(X_test, model, training_features):
    X_test = pd.DataFrame(X_test, columns=training_features)
    predictions = model.predict_proba(X_test)
    ids = X_test.user_id.uniques()
    probs = pd.DataFrame(predictions, columns=model.classes_, index=ids)

    # TODO: this should be a function, need refactoring
    # select top 5 destinations for each user
    results = {}
    n = 1
    tot = len(probs.index) + 1
    for id in probs.index:
        df = probs[probs.index == id]
        destinations = []
        count = 0
        while count < 5:
            destination = df.idxmax(axis=1)
            destinations.append(destination[0])
            df.drop(destination, inplace=True, axis=1)
            count += 1
        results[id] = destinations
        n += 1

    # TODO: there is no need to create a data frame for this. Should output to text file directly from the for loop
    submission = pd.DataFrame(columns=["id", "country"])
    for key, value in results.items():
        submission = pd.concat([submission, pd.DataFrame({"id": key, "country": value})])

    return submission


def prepare_dataset(t):
    # set variables
    path = "./data/"
    train_file = "training_features.pickle"
    train_full_path = "".join((path, train_file))
    test_file = "test_features.pickle"
    test_full_path = "".join((path, test_file))

    # read data
    with open(train_full_path, 'rb') as f:
        train = pickle.load(f)
    with open(test_full_path, 'rb') as f:
        test = pickle.load(f)

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

    """
    I should read a little bit more into how to perform feature selection before training, in particular the Strong Rule
    approach proposed by Robert Tibshirani et al. (http://tinyurl.com/hu8lzxn). Chi-square, even though it's a more
    sophisticated method, doesn't improve performance compared to a variance threshold method. This is to me a bit
    suspicious.
    """
    # feature selection
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


def train_logistic_regression(X, y, training_features=[]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split

    # train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf = lr.fit(X_train, y_train)

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # normalized discounted cumulative gain
    predictions = format_predictions(X_test, clf, training_features)
    ndcg = ndcg_n(predictions, y_test)
    print('NDCG on test set:', ndcg)

    return clf


def train_decision_tree(X, y):

    from sklearn import tree
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split

    # train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # train decision tree
    tree = tree.DecisionTreeClassifier(random_state=23)
    param_grid = {'max_depth': [3, 4, 5, 6, 10]}
    CV_tree = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)
    CV_tree.fit(X_train, y_train)
    print(CV_tree.best_params_)
    print('Decision Tree Accuracy: %0.2f'.format(CV_tree.best_score_))

    # normalized discounted cumulative gain
    predictions = format_predictions(X_test, CV_tree)
    ndcg = ndcg_n(predictions, y_test)
    print('NDCG on test set:', ndcg)

    return CV_tree


def train_random_forest(X_train, y_train):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV

    rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=2013, oob_score=True)

    # grid search
    param_grid = {
        'n_estimators': [1000, 2000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [20, 35, 50]
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)

    # cross-validation
    scores = CV_rfc.best_score_
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return CV_rfc


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



