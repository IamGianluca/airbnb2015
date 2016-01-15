"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Control board to conveniently launch featurizer, training and prediction flows
"""

import featurizer as featurize
import training_pipeline as train
import predicting_pipeline as predict

if __name__ == "__main__":

    # featurizer flow
    training_features = featurize.create_features(True)
    featurize.create_features(False, training_features)

    # create training and test sets
    X_train, X_test, y_train, ids_test = train.prepare_dataset()

    # train decision trees classifier and pull results
    tree_model = train.train_decision_tree(X_train, y_train)
    predict.predict_decision_tree(X_test, ids_test, tree_model)

    # train random forest classifier and pull results
    # rf_model = tp.train_random_forest(X_train, X_test, y_train, y_test)
    # predict.predict_random_forest(rf_model, features)

    # train xgboost classifier and pull results