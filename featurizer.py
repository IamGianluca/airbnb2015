"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Feature Engineering pipeline
"""

import pandas as pd
import numpy as np
import collections
import pickle


def user_session_count(time_series):
    sessions = 0
    for t in time_series:
        if t > 20:
            sessions += 1
    return sessions


def median_page_per_visit(time_series):
    page_per_visit = []
    pages = 0
    for t in time_series:
        if t < 20:
            pages += 1
        else:
            page_per_visit.append(pages)
            pages = 0
    return np.nanmedian(page_per_visit, axis=0)


def create_features(is_training_set=True, training_features=[]):
    # set variables
    directory = "./data/"
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
    chunk_size = 100000

    # load only 100,000 lines at a time because of memory restrictions (the file weights over 630MB)
    session_data = pd.DataFrame()
    reader = pd.read_csv(session_full_path, chunksize=chunk_size)
    n = 0
    for chunk in reader:
        n += 1
        chunk["mins_elapsed"] = chunk.secs_elapsed / 60

        # extract user sessions count and median number of pages browsed per session
        chunk_user_session_counts = pd.DataFrame({"user_id": chunk.user_id,
                                                  "sessions": chunk.groupby("user_id").mins_elapsed.apply(user_session_count),
                                                  "median_pages_per_visit": chunk.groupby("user_id").mins_elapsed.apply(median_page_per_visit)})
        chunk_user_session_counts.set_index('user_id')

        # process session data (transform from long to wide)
        df = pd.DataFrame({"count": chunk.groupby(["user_id", "action_type", "action_detail", "action"]).action.count()}).reset_index()

        cols = []
        for i in range(0, len(df)):
            cols.append(str(df["action"].values[i] + "_" + df["action_type"].values[i] + "_" +
                            df["action_detail"].values[i]))
        cols = np.array(cols)
        df["grouping"] = cols

        df = pd.DataFrame({"count": df.groupby(["user_id", "grouping"])['count'].sum()}).reset_index()
        chunk_action_data = df.pivot(index="user_id", columns="grouping", values="count").fillna(0)
        del df

        # join user session count data with activity data
        to_concatenate = chunk_action_data.join(chunk_user_session_counts, how="outer")
        # to_concatenate = pd.concat([chunk_action_data, chunk_user_session_counts], axis=1)

        to_concatenate = to_concatenate.drop_duplicates(keep='first', inplace=False)

        del chunk_action_data, chunk_user_session_counts

        # concatenate new chunk to existing results
        session_data = pd.concat([session_data, to_concatenate], axis=0)
        del to_concatenate

        print("Processed chunk {0}".format(n))

    # hack! exclude from test data set all features not seen in the training set
    if is_training_set is False and len(training_features) > 1:
        session_data = session_data[list(training_features)]

    # make sure there is one line for each user_id in the final `session_data` data frame
    duplicates = [item for item, count in collections.Counter(session_data["user_id"]).items() if count > 1]
    duplicates_features = session_data[session_data.user_id.isin(duplicates)].groupby("user_id").apply(np.sum).drop("user_id", axis=1).reset_index()
    non_duplicates_features = session_data[~session_data.user_id.isin(duplicates)]
    session_features = pd.concat([non_duplicates_features, duplicates_features], axis=0)

    # create empty data frame to store user features
    user_data = pd.read_csv(user_full_path)
    if is_training_set is True:
        user_features = user_data[["id", "age", "signup_flow", "country_destination"]]
    else:
        user_features = user_data[["id", "age", "signup_flow"]]

    # create features out of 'date_account_created' and 'timestamp_first_active'
    user_features.loc[:, 'account_created_day'] = pd.to_datetime(user_data.date_account_created).map(lambda x: x.day)
    user_features.loc[:, 'account_created_month'] = pd.to_datetime(user_data.date_account_created).map(lambda x: x.month)
    user_features.loc[:, 'account_created_year'] = pd.to_datetime(user_data.date_account_created).map(lambda x: x.year)

    user_features.loc[:, 'account_created_day'] = pd.to_datetime(user_data.timestamp_first_active, format="%Y%m%d%H%M%S").map(lambda x: x.day)
    user_features.loc[:, 'account_created_month'] = pd.to_datetime(user_data.timestamp_first_active, format="%Y%m%d%H%M%S").map(lambda x: x.month)
    user_features.loc[:, 'account_created_year'] = pd.to_datetime(user_data.timestamp_first_active, format="%Y%m%d%H%M%S").map(lambda x: x.year)

    # crete dummy variables for categorical user features
    cols = ["gender", "signup_method", "language", "affiliate_channel", "affiliate_provider",
            "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"]
    for col in cols:
        dummies = pd.get_dummies(user_data[col], prefix=col)
        user_features = pd.concat([user_features, dummies.ix[:, 1:]], axis=1)

    # join user session and user features
    features = pd.merge(user_features, session_features, how="inner", left_on="id", right_on="user_id")
    features = features.fillna(0)
    features.drop(["user_id"], axis=1, inplace=True)

    # debug
    if is_training_set:
        print("Training set size:", features.shape)
    else:
        print("Test set size:", features.shape)

    # save output
    features.to_csv(destination_full_path, index=False)
    print("Featurizer has completed its job and saved the results in a file named '{}'".format(destination_file))

    # hack! keep track of the features we need in order to drop them from test data set
    if is_training_set is True:
        return session_features.columns


if __name__ == "__main__":

    # process training data
    training_features = create_features(True)

    # process test data
    create_features(False, training_features)

