"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Predicting pipeline
"""

import pandas as pd
import datetime
import pickle


def format_output_filename(filename, fmt='%Y%m%d-%H%M%S_{fname}'):
    # add current datetime to the result file name
    return datetime.datetime.now().strftime(fmt).format(fname=filename)


def make_predictions(test_set, model, centre=False, filename="results.csv"):
    with open('./data/test_ids.pickle', 'rb') as f:
        ids = pickle.load(f)

    # if the model is not scale independent, scale the test dataset
    if centre is True:
        from sklearn import preprocessing
        test_set = preprocessing.scale(test_set)

    # predict most likely destination for each user, using decision tree classifier
    predictions = model.predict_proba(test_set)
    probs = pd.DataFrame(predictions, columns=model.classes_, index=ids)

    # TODO: this should be a function, need refactoring
    # select top 5 destinations for each user
    results = {}
    for idx in probs.index:
        row = probs.iloc[idx]
        destinations = []
        count = 0
        while count < 5:
            destination = row.idxmax(axis=1)
            destinations.append(destination)
            row.drop(destination, inplace=True, axis=0)
            count += 1
        results[idx] = destinations

    # TODO: there is no need to create a data frame for this. Should output to text file directly from the for loop
    submission = pd.DataFrame(columns=["aid", "country"])
    for key, value in results.items():
        submission = pd.concat([submission, pd.DataFrame({"aid": key, "country": value})])

    result_path = "./predictions/"
    result_file_name = format_output_filename(filename)
    result_file_path = "".join((result_path, result_file_name))

    submission.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)
