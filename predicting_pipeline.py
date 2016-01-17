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


def predict_decision_tree(X_test, ids, model):
    # predict most likely destination for each user, using decision tree classifier
    predictions = model.predict(X_test)

    # save predictions in a dictionary
    dict_results = {}
    i = 0
    for user in ids:
        dict_results[user] = predictions[i]
        i += 1

    # TODO: improve the way we pull remaining guesses
    # use as first guess the prediction from the decision tree and add top destinations as remaining guesses
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

    # TODO: need to output results in the required format ("aid" workaround)
    result_path = "./predictions/"
    result_file_name_end = "results_decision_trees.csv"
    result_file_name = format_output_namefile(result_file_name_end)
    result_file_path = "".join((result_path, result_file_name))

    result.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)


def predict_logistic_regression(X_test, ids, model):
    # return likelihood on a probability scale for each destination
    predictions = model.predict_proba(X_test)
    probs = pd.DataFrame(predictions, columns=model.classes_, index=ids)

    # select top 5 destinations for each user
    results = {}
    n = 1
    tot = len(probs.index) + 1
    for id in probs.index:
        df = probs[probs.index==id]
        destinations = []
        count = 0
        while count < 5:
            destination = df.idxmax(axis=1)
            destinations.append(destination[0])
            df.drop(destination, inplace=True, axis=1)
            count += 1
        results[id] = destinations
        n += 1
        print("Completed:", n, "/", tot)

    submission = pd.DataFrame(columns=["aid", "country"])
    for key, value in results.items():
        submission = pd.concat([submission, pd.DataFrame({"aid": key, "country": value})])

    result_path = "./predictions/"
    result_file_name_end = "results_logistic_regression.csv"
    result_file_name = format_output_namefile(result_file_name_end)
    result_file_path = "".join((result_path, result_file_name))

    submission.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)
