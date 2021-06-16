# Split and standardize data based on timestamps

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilities.data_dict import DataDict

# data_dict_list = user specified subset of data_dict
# targets = desired transpiration dictionary

DataDict = DataDict()


def sanitizer(data_dict_list, targets):
    # Store available data into X (features) and Y (target) matrices
    x = []
    y = []
    timestamp = []
    for row in targets:
        features = []
        date = row[0]
        for data_dict in data_dict_list:
            feature = data_dict.get(date, False)  # search dictionary for time stamp
            if data_dict == DataDict.rainfall:
                feature = data_dict.get(date[:10], False)  # rainfall dictionary
            features.append(feature)
        if all(features):
            x.append(features)
            y.append(row[1])
            timestamp.append(date)

    x = np.array(x)
    y = np.array(y) * 86400  # conversion factor to [mm/d]
    print(f"Mean overall transpiration is {round(np.mean(y), 4)} mm/d")
    print(f'\nStandard deviation:', np.std(y))
    return x, y, timestamp


def split_data(x, y):
    # Standardize and split data
    # Set aside testing and validation sets: 80% training, 10% testing, 10% validation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    x = scaler.transform(x)

    # Define m and n using input features
    m = len(y_train)
    n = len(x_train[0]) + 1
    print("m = ", m, "instances in training set")
    print("n = ", n, "number of features")
    return x, x_train, x_test, x_val, y, y_train, y_test, y_val
