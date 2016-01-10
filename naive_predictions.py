"""
@author: Gianluca Rossi
@project: Airbnb Recruiting Challenge (New User Bookings)
@title: Naive model predictions
"""

import pandas as pd


def predict():
    test = pd.read_csv("data/test_users.csv")
    # TODO: flag as a bug in pandas, to_csv doesn't preserve columns order. As a workaround I've renamed `id` as `aid`
    result = pd.DataFrame(columns=['aid', 'country'])

    users = test.id.unique()
    tot = len(users)
    c = 0

    for user in users:
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': ['NDF']})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': ['US']})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': ['other']})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': ['FR']})])
        result = pd.concat([result, pd.DataFrame({'aid': [user], 'country': ['IT']})])
        
        # print loop status
        c += 1
        print(c, "/", tot)

    file_name = "result.csv"    
    result.to_csv(file_name, index=False)
    print("Output saved in", file_name)

if __name__ == '__main__':
    predict()