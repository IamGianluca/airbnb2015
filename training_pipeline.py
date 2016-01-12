"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Training pipeline
"""

import pandas as pd


def prepare_dataset():

    # set variables
    path = "/home/gianluca/Kaggle/airbnb2015/data/"
    data_file = "training_features.csv"
    data_file_path = "".join((path, data_file))

    # load dataset
    data = pd.read_csv(data_file_path, nrows=1000)

    # separate outcome from independent variables
    y = data.country_destination.tolist()
    X = data.drop(["id", "country_destination"], axis=1).as_matrix()
    del data

    # split into training and test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    del X, y

    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, X_test, y_train, y_test):

    from sklearn import tree

    # train decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # make predictions
    predictions = clf.predict(X_test)

    # compute misclassification rate
    error_rate = (y_test != predictions).mean()
    print("Misclassifcation error: {0:.2%}".format(error_rate))

    return clf


def predict_decision_tree(model):

    # set variables
    path = "/home/gianluca/Kaggle/airbnb2015/data/"
    data_file = "test_features.csv"
    data_file_path = "".join((path, data_file))

    test_data = pd.read_csv(data_file_path)
    result = model.predict(test_data)

    result_file_name = "result.csv"
    result.to_csv(result_file_name, index=False)
    print("Output saved in", result_file_name)


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_dataset()
    model = train_decision_tree(X_train, X_test, y_train, y_test)
    predict_decision_tree(model)
