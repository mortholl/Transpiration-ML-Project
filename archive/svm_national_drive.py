# Build and visualize the model for the National Drive data

from utilities.data_import import NationalDriveData
from archive.data_split_sanitize import sanitizer, split_data
from archive import svm

DataDict = NationalDriveData()

data_dict_list = [DataDict.gwl, DataDict.air_temp, DataDict.rainfall, DataDict.humidity, DataDict.wind, DataDict.solar, DataDict.vpd]
X, Y, timestamp = sanitizer(data_dict_list, DataDict.transpiration)
X, X_train, X_test, X_val, Y, Y_train, Y_test, Y_val = split_data(X, Y)

params = {
    "kernel": ['rbf'],  # 'linear', 'poly'
    "epsilon": [0.1],  # 1, 5
    "C": [10, 100]  # 1
}

# Search for best model
svr = svm.svm_search(X_train, X_test, Y_train, Y_test, params)

# Visualize model performance
fig = svm.svm_visualize(X, Y, svr, timestamp, name="National Drive model for hourly transpiration")
