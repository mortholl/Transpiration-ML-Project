# Build and visualize the model for the '19-20 data
# Input features: solar, temp, humidity, vpd, rainfall, soil, gwl_1718, gwl_1920, gwl
# Targets: transpiration_1718, transpiration_1920, transpiration

from utilities.data_import import GatumData
from archive.data_split_sanitize import sanitizer, split_data, season_split
from utilities import svm

DataDict = GatumData()
# Import data by desired features, split by season into dry and wet models
data_dict_list = [DataDict.solar, DataDict.temp, DataDict.humidity, DataDict.vpd, DataDict.rainfall, DataDict.soil]
X_1920, Y_1920, timestamp_1920 = sanitizer(data_dict_list, DataDict.transpiration_1920)
X_wet, Y_wet, timestamp_wet = season_split(X_1920, Y_1920, timestamp_1920, ['fall', 'winter', 'spring'])
X_dry, Y_dry, timestamp_dry = season_split(X_1920, Y_1920, timestamp_1920, ['summer'])
X_wet, X_train_wet, X_test_wet, X_val_wet, Y_wet, Y_train_wet, Y_test_wet, Y_val_wet = split_data(X_wet, Y_wet)
X_dry, X_train_dry, X_test_dry, X_val_dry, Y_dry, Y_train_dry, Y_test_dry, Y_val_dry = split_data(X_dry, Y_dry)

params = {
    "kernel": ['rbf'],  # 'linear', 'poly'
    "epsilon": [0.1],  # 1, 5
    "C": [10, 100]  # 1
}

# Search for best wet model
svr_1920_wet = svm.svm_search(X_train_wet, X_test_wet, Y_train_wet, Y_test_wet, params)

# Visualize model performance
wet_fig = svm.svm_visualize(X_wet, Y_wet, svr_1920_wet, timestamp_wet, name="Wet model")

# Search for best dry model
svr_1920_dry = svm.svm_search(X_train_dry, X_test_dry, Y_train_dry, Y_test_dry, params)

# Visualize model performance
dry_fig = svm.svm_visualize(X_dry, Y_dry, svr_1920_dry, timestamp_dry, name="Dry model")