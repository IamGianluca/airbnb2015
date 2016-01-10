"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Feature Engineering pipeline
"""

import pandas as pd


def create_features():

    # set variables
    directory = "/home/gianluca/Kaggle/airbnb2015/data/"
    session_file = "sessions.csv"
    user_file = "train_users_2.csv"
    destination_file = "training_features.csv"
    session_full_path = "".join((directory, session_file))
    user_full_path = "".join((directory, user_file))
    destination_full_path = "".join((directory, destination_file))
    chunk_size = 100000

    # load only first 100,000 lines because of memory restrictions
    session = pd.read_csv(session_full_path, nrows=chunk_size)
    session["mins_elapsed"] = session.secs_elapsed / 60

    # process session data (transform from long to wide)
    session_data = pd.DataFrame({'count': session.groupby(["user_id", "action"]).action.count()}).reset_index().pivot(index='user_id', columns='action', values='count').fillna(0).reset_index()

    # create user features matrix
    user = pd.read_csv(user_full_path)
    user_features = user[['id', 'age', 'signup_flow', 'country_destination']]

    # TODO: create features out of 'date_account_created', 'timestamp_first_active', 'date_first_booking'

    # add to user feature data frame dummy variables for categorical data
    cols = ["gender", "signup_method", "language", "affiliate_channel", "affiliate_provider",
            "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"]
    for col in cols:
        dummies = pd.get_dummies(user[col], prefix=col)
        user_features = pd.concat([user_features, dummies.ix[:, 1:]], axis=1)

    # join user session and user features
    features = pd.merge(user_features, session_data, how="inner", left_on="id", right_on="user_id")
    features = features.drop('user_id', axis=1)

    # save output
    features.fillna(0).to_csv(destination_full_path, index=False)
    print("Featurizer has completed its job and saved the results in a file named '{}'".format(destination_file))


if __name__ == "__main__":
    create_features()
