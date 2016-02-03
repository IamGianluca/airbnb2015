"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""


from evaluation_metric import ndcg_n, format_predictions
from sklearn import cross_validation
import pickle


def prepare_dataset():
    # TODO: optimise this parameter for each model
    t = 150

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
    not_in_test.remove('country_destination')
    train = train.drop(not_in_test, axis=1)

    # separate outcome from independent variables
    y_train = train.country_destination.tolist()

    # dumb training and test user ids
    train_ids, test_ids = train.id, test.id
    with open('./data/training_ids.pickle', 'wb') as f:
        pickle.dump(train_ids, f)
    with open('./data/test_ids.pickle', 'wb') as f:
        pickle.dump(test_ids, f)

    # TODO: there should be either user_id or id, not both! This must be addressed in the featurizer flow
    X_train_before_feature_selection = train.drop(['user_id', 'id', 'country_destination'], axis=1).as_matrix()
    X_test_before_feature_selection = test.drop(['user_id', 'id'], axis=1).as_matrix()

    # feature selection; remove all features that are either one or zero (on or off) in more than 80% of the samples
    from sklearn.feature_selection import SelectKBest, chi2

    print("Training set size before feature selection:", X_train_before_feature_selection.shape)
    print("Test set size before feature selection:", X_test_before_feature_selection.shape)

    """
    Read into how to perform feature selection before training, in particular the Strong Rule approach proposed
    by Robert Tibshirani et al. (http://tinyurl.com/hu8lzxn). Chi-square, even though it's a more sophisticated
    method, doesn't improve performance compared to a variance threshold method. This is to me a bit suspicious (?)
    """
    # feature selection
    sel = SelectKBest(chi2, k=t)
    X_train = sel.fit_transform(X_train_before_feature_selection[:, 1:], y_train)
    X_test = sel.transform(X_test_before_feature_selection[:, 1:])
    del X_train_before_feature_selection, X_test_before_feature_selection

    # keep track of the features remaining after the feature selection step
    idxs = sel.get_support(indices=False)
    # TODO: verify id and user_id are not included
    selected_features = train.drop(["id", "user_id", "country_destination"], axis=1).columns[idxs].values
    del train, test

    print("Number of features included:", t)
    print("Training set size after feature selection:", X_train.shape)
    print("Selected features are:", selected_features)

    # write features which passed the feature selection step
    with open('./data/selected_features.pickle', 'wb') as f:
        pickle.dump(selected_features, f)

    return X_train, X_test, y_train


def train_logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split

    # train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=395)

    # TODO: separate feature selection step

    # fit logistic regression
    # TODO: Lasso regularization
    lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf = lr.fit(X_train, y_train)

    # normalized discounted cumulative gain
    predictions = format_predictions(X_test, clf)
    ndcg = ndcg_n(predictions, y_test)
    print('Logistic regression classifier NDCG (on test set): : {0:.2f}'.format(ndcg))

    return clf


def train_decision_tree(X, y):

    from sklearn import tree
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split

    # train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1093)

    # TODO: separate feature selection step

    # fit decision tree
    tree = tree.DecisionTreeClassifier(random_state=23)
    param_grid = {'max_depth': [3, 4, 5, 6, 10]}
    CV_tree = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)
    CV_tree.fit(X_train, y_train)
    print('Decision tree classifier best parameters:', CV_tree.best_params_)

    # normalized discounted cumulative gain
    predictions = format_predictions(X_test, CV_tree)
    ndcg = ndcg_n(predictions, y_test)
    print('Decision tree classifier NDCG (on test set): {0:.2f}'.format(ndcg))

    return CV_tree


def train_random_forest(X, y):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split

    # train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=357)

    # TODO: separate feature selection step

    # fit random forest classifier
    rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=2013, oob_score=True)
    param_grid = {
        'n_estimators': [1000, 2000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [20, 35, 50]
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)
    print('Random Forest classifier best parameters:', CV_rfc.best_params_)

    # normalized discounted cumulative gain
    predictions = format_predictions(X_test, CV_rfc)
    ndcg = ndcg_n(predictions, y_test)
    print('Random Forest classifier NDCG (on test set): {0:.2f}'.format(ndcg))

    return CV_rfc


def train_svm(X_train, y_train):

    from sklearn.svm import SVC
    from sklearn import preprocessing

    # scale and centre variables as Support Vector Machine algorithms are not scale invariant
    # sample = np.random.choice(X_train.shape[0], 5000, replace=False)
    X_train_c = preprocessing.scale(X_train[:20000, :])
    y_train = y_train[:20000]

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



