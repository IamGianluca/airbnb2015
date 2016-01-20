"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Predicting pipeline
"""

import pandas as pd
import datetime


def format_output_namefile(fname, fmt='%Y%m%d-%H%M%S_{fname}'):
    # add current datetime to the result file name; useful to keep track of multiple solutions
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def make_predictions(X_test, ids, model, centre=False, filename="results.csv"):
    # if the model is not scale independent, scale the test dataset
    if centre is True:
        from sklearn import preprocessing
        X_test = preprocessing.scale(X_test)

    # predict most likely destination for each user, using decision tree classifier
    predictions = model.predict_proba(X_test)
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
        # print("Completed:", n, "/", tot)

    # TODO: there is no need to create a data frame for this. Should output to text file directly from the for loop
    submission = pd.DataFrame(columns=["aid", "country"])
    for key, value in results.items():
        submission = pd.concat([submission, pd.DataFrame({"aid": key, "country": value})])

    result_path = "./predictions/"
    result_file_name = format_output_namefile(filename)
    result_file_path = "".join((result_path, result_file_name))

    submission.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)
