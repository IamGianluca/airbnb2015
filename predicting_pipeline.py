"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Predicting pipeline
"""

import pandas as pd


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
    result_path = "/Users/grossi/"
    result_file_name = "results_decision_trees.csv"
    result_file_path = "".join((result_path, result_file_name))

    result.to_csv(result_file_path, index=False)
    print("Output saved in", result_file_name)
