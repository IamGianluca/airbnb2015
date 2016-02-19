"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Pipeline launcher
"""


import featurizer as featurize
import training_pipeline as train
import predicting_pipeline as predict

if __name__ == "__main__":
    # pre-processing and feature extraction step
    # featurize.create_features(True)
    # featurize.create_features(False)

    # split training and test sets
    X_train, X_test, y_train = train.prepare_dataset()

    # train logistic regressions
    lr = train.train_logistic_regression(X_train, y_train)
    predict.make_predictions(test_set=X_test, model=lr, centre=False, filename="results_lr.csv")

    # # train decision trees classifier
    tree = train.train_decision_tree(X_train, y_train)
    predict.make_predictions(test_set=X_test, model=tree, centre=False, filename="results_tree.csv")
    #
    # # train random forest classifier
    rf = train.train_random_forest(X_train, y_train)
    predict.make_predictions(test_set=X_test, model=rf, centre=False, filename="results_rf.csv")

    # train knn; accuracy close to tree models but very slow to fit
    # knn = train.train_svm(X_train, y_train)
    # predict.make_predictions(X_test, ids_test, knn, centre=True, filename="results_knn.csv")

    # train svm; accuracy close to tree models but very slow to fit
    # svc = train.train_svm(X_train, y_train)
    # predict.make_predictions(X_test, ids_test, svc, centre=True, filename="results_svm_rbf.csv")
