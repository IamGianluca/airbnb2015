"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Feature Engineering pipeline
"""

import pandas as pd
import numpy as np
import collections

def create_features(is_training_set=True):

    # set variables
    directory = "/home/gianluca/Kaggle/airbnb2015/data/"
    session_file = "sessions.csv"
    if is_training_set:
        user_file = "train_users_2.csv"
        destination_file = "training_features.csv"
    else:
        user_file = "test_users.csv"
        destination_file = "test_features.csv"
    session_full_path = "".join((directory, session_file))
    user_full_path = "".join((directory, user_file))
    destination_full_path = "".join((directory, destination_file))
    chunk_size = 10000

    # load only 100,000 lines at a time because of memory restrictions (the file weights over 630MB)
    session_data = pd.DataFrame()
    reader = pd.read_csv(session_full_path, chunksize=chunk_size)
    n = 0
    for chunk in reader:
        n += 1
        chunk["mins_elapsed"] = chunk.secs_elapsed / 60

        # process session data (transform from long to wide)
        df = pd.DataFrame({"count": chunk.groupby(["user_id", "action"]).action.count()}).reset_index()
        chunk_session_data = df.pivot(index="user_id", columns="action", values="count").fillna(0).reset_index()

        # add `chunk_session_data` to `session_data`
        session_data = pd.concat([session_data, chunk_session_data], axis=0)

        # remove chunk_session_data to save memory
        del chunk_session_data
        print("I've done with chunk {}".format(n))

    # make sure there is one line for each user_id in the final `session_data` data frame
    """
    # this method should work but I get into memory troubles on my laptop with just 4GB of RAM
    session_features = session_data.fillna(0).groupby("user_id")
    session_features = session_features.apply(np.sum, axis=0)
    """
    duplicates = [item for item, count in collections.Counter(session_data["user_id"]).items() if count > 1]
    duplicates_features = session_data[session_data.user_id.isin(duplicates)].groupby("user_id").apply(np.sum).drop("user_id", axis=1).reset_index()
    non_duplicates_features = session_data[~session_data.user_id.isin(duplicates)]
    session_features = pd.concat([non_duplicates_features, duplicates_features], axis=0)

    # create user features matrix
    user = pd.read_csv(user_full_path)
    if is_training_set:
        user_features = user[["id", "age", "signup_flow", "country_destination"]]
    else:
        user_features = user[["id", "age", "signup_flow"]]

    # TODO: create features out of 'date_account_created', 'timestamp_first_active', 'date_first_booking'

    # add to user feature data frame dummy variables for categorical data
    cols = ["gender", "signup_method", "language", "affiliate_channel", "affiliate_provider",
            "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"]
    for col in cols:
        dummies = pd.get_dummies(user[col], prefix=col)
        user_features = pd.concat([user_features, dummies.ix[:, 1:]], axis=1)

    # join user session and user features
    features = pd.merge(user_features, session_features, how="inner", left_on="id", right_on="user_id")
    features = features.fillna(0)
    features.drop("user_id", axis=1, inplace=True)

    # save output
    features.to_csv(destination_full_path, index=False)
    print("Featurizer has completed its job and saved the results in a file named '{}'".format(destination_file))


if __name__ == "__main__":
    create_features(False)
