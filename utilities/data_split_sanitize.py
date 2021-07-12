# Split and standardize data based on timestamps

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilities.data_import import GatumData

# data_dict_list = user specified subset of data_dict
# targets = desired transpiration dictionary


def sanitizer(data_dict_list, targets):
    # Store available data into X (features) and Y (target) matrices
    x = []
    y = []
    timestamp = []
    for key in targets:
        features = []
        time = key
        date = time.split(' ')[0]
        for data_dict in data_dict_list:
            feature = data_dict.get(time, 'missing')  # search dictionary for time stamp
            if feature == 'missing':
                feature = data_dict.get(date, 'missing')  # check if the dictionary is using the date only
            features.append(feature)
        if all(feature != 'missing' for feature in features):
            x.append(features)
            y.append(targets[key])
            timestamp.append(time)

    x = np.array(x)
    y = np.array(y)  # * 86400 conversion factor to [mm/d]
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


def season_split(x, y, times, seasons):
    # Split x and y into seasons based on the month of the corresponding timestamps
    # Based on the input season(s), return x, y, and the timestamps for that season(s)
    new_x = []
    new_y = []
    new_times = []
    for i, time in enumerate(times):
        month = time[5:7]
        if 'spring' in seasons:
            if month == '09' or month == '10' or month == '11':
                new_x.append(x[i])
                new_y.append(y[i])
                new_times.append(time)
        if 'summer' in seasons:
            if month == '12' or month == '01' or month == '02':
                new_x.append(x[i])
                new_y.append(y[i])
                new_times.append(time)
        if 'fall' in seasons:
            if month == '03' or month == '04' or month == '05':
                new_x.append(x[i])
                new_y.append(y[i])
                new_times.append(time)
        if 'winter' in seasons:
            if month == '06' or month == '07' or month == '08':
                new_x.append(x[i])
                new_y.append(y[i])
                new_times.append(time)
    return new_x, new_y, new_times
