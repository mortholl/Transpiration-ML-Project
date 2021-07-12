# Build and visualize the model for the '17-'18 data
# Input features: solar, temp, humidity, vpd, rainfall, soil, gwl_1718, gwl_1920, gwl
# Targets: transpiration_1718, transpiration_1920, transpiration

from utilities.data_import import GatumData
from utilities.data_split_sanitize import sanitizer, split_data, season_split
from utilities import svm

DataDict = GatumData()

data_dict_list = [DataDict.solar, DataDict.temp, DataDict.humidity, DataDict.vpd, DataDict.rainfall, DataDict.gwl_1718]
X_1718, Y_1718, timestamp_1718 = sanitizer(data_dict_list, DataDict.transpiration_1718)
# X_1718, Y_1718, timestamp_1718 = season_split(X_1718, Y_1718, timestamp_1718, ['winter', 'spring'])
X_1718, X_train_1718, X_test_1718, X_val_1718, Y_1718, Y_train_1718, Y_test_1718, Y_val_1718 = split_data(X_1718, Y_1718)

params = {
    "kernel": ['rbf'],  # 'linear', 'poly'
    "epsilon": [0.1],  # 1, 5
    "C": [10, 100]  # 1
}

# Search for best model
svr_1718 = svm.svm_search(X_train_1718, X_test_1718, Y_train_1718, Y_test_1718, params)

# Visualize model performance
fig = svm.svm_visualize(X_1718, Y_1718, svr_1718, timestamp_1718, name="'17-'18 model")
