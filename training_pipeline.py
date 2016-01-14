"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd
import numpy as np


def prepare_dataset():

    # set variables
    path = "/home/gianluca/Kaggle/airbnb2015/data/"
    data_file = "training_features.csv"
    data_file_path = "".join((path, data_file))

    # load dataset
    data = pd.read_csv(data_file_path, nrows=1000)

    # separate outcome from independent variables
    y = data.country_destination.tolist()
    # TODO avoid returning list of features used in training. This should be done in the featurizer
    features = data.columns
    features = [feature for feature in features if feature not in ["id", "country_destination"]]
    X = data.drop(["id", "country_destination"], axis=1).as_matrix()
    del data

    # split into training and test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    del X, y

    return X_train, X_test, y_train, y_test, features


def train_decision_tree(X_train, X_test, y_train, y_test):

    from sklearn import tree
    from sklearn import cross_validation

    # train decision trees
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # cross-validation
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf


def predict_decision_tree(model, features):

    # set variables
    path = "/home/gianluca/Kaggle/airbnb2015/data/"
    data_file = "test_features.csv"
    data_file_path = "".join((path, data_file))

    test_data = pd.read_csv(data_file_path)
    ids = test_data["id"]

    # add to test data set missing features, this is only to get a working submission by end of today
    # TODO: remove this dirty hack!
    missing_features = [feature for feature in features if feature not in test_data.columns]
    for missing_feature in missing_features:
        test_data[missing_feature] = np.zeros(test_data.shape[0])

    test_data = test_data[features]
    #test_data.drop(["id"], axis=1, inplace=True)
    predictions = model.predict(test_data)
    # result = pd.DataFrame({'aid': ids, 'country': predictions})

    # save predictions in a dictionary
    dict_results = {}
    i = 0
    for user in ids:
        dict_results[user] = predictions[i]
        i += 1

    # use as first guess the prediction from the decision tree and use the other best guesses as remaining choises
    top_destinations = ["NDF", "US", "other", "FR", "IT"]
    result = pd.DataFrame(columns=['aid', 'country'])
    c = 0
    tot = len(ids)

    for user in dict_results.keys():
        remaining_predictions = [country for country in top_destinations if country not in dict_results[user]]
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': dict_results[user]})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': remaining_predictions[0]})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': remaining_predictions[1]})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': remaining_predictions[2]})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': remaining_predictions[3]})])

        # print loop status
        c += 1
        print(c, "/", tot)

    file_name = "result.csv"
    result.to_csv(file_name, index=False)
    print("Output saved in", file_name)

    # TODO: need to output results in the required format
    result_path = "/home/gianluca/Kaggle/airbnb2015/predictions/"
    result_file_name = "results_decision_trees.csv"
    result_file_path = "".join((result_path, result_file_name))

    result.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, features = prepare_dataset()
    model = train_decision_tree(X_train, X_test, y_train, y_test)
    # predict_decision_tree(model, features)
