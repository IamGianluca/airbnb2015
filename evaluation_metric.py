"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Evaluation metric
"""


import pandas as pd
import numpy as np
import math


def ndcg_n(predictions, truth, k=5):
    # TODO: validate results using leaderboard and manual computations
    # idcg = [1.0, 0.6309297535714575, 0.5, 0.43067655807339306, 0.38685280723454163]
    users = predictions.id.unique()
    ndcg = []
    n = 0
    for user in users:
        res = []
        guesses = predictions[predictions.id == user]['country'].reset_index(drop=True)
        observed = truth[n]
        for i in range(0, k, 1):
            if guesses.loc[i] == observed:
                rel = 1
            else:
                rel = 0
            nom = math.pow(2, rel) - 1
            den = math.log2(i+2)
            inter = nom / den
            res.append(inter)
        ndcg.append(sum(res))
        n += 1
    return np.mean(ndcg)


def format_predictions(X_test, model):
    predictions = model.predict_proba(X_test)
    try:
        probs = pd.DataFrame(predictions, columns=model.classes_)
    except:
        probs = pd.DataFrame(predictions, columns=model.best_estimator_.classes_)

    # select top 5 destinations for each user
    results = {}
    for idx in probs.index:
        row = probs.iloc[idx, :]
        destinations = []
        count = 0
        while count < 5:
            destination = row.idxmax(axis=1)
            destinations.append(destination)
            row.drop(destination, inplace=True, axis=0)
            count += 1
        results[idx] = destinations

    submission = pd.DataFrame(columns=["id", "country"])
    for key, value in results.items():
        submission = pd.concat([submission, pd.DataFrame({"id": key, "country": value})])

    return submission