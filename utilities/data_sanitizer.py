import pandas as pd
import numpy as np
import os
import datetime
import csv

# This file needs optimizing, takes way too long - think about implementing np.loadtxt instead
# Could read the column names into a list, record the indices, then load the relevant indices into a numpy array


def feature_generator(feature_list, file_list):   # Generates X and Y arrays based on lists of feature names and files
    # The feature names list must exactly match the column names in the SAPFLUXNET database
    feature_directory = 'data/modeling_data/features'
    target_directory = 'data/modeling_data/targets'
    x_array = np.empty((1, 1))
    y_array = np.empty((1, 1))
    for filename in os.listdir(feature_directory):   # pull one file at a time from the features directory
        if filename in file_list:
            feature_path = feature_directory + '/' + filename
            with open(feature_path) as f:
                reader = csv.reader(f, delimiter=',')
                first_row = next(reader)
                ncols = len(first_row)
            data = np.loadtxt(feature_path, skiprows=1, usecols=range(2, ncols))
            for row in data:
                row = row[1]
                for feature in feature_list:
                    x_array.append(row[feature])  # almost definitely not doing what I want here
            # Need to look up corresponding target file, make sure all the timestamps match, and add to y_array
    return x_array, y_array

# Pandas implementation:
# df = pd.read_csv(feature_directory + '/' + filename, header=0)
# for row in df.iterrows():
#     row = row[1]
#     for feature in feature_list:
#         x_array.append(row[feature])

begin_time = datetime.datetime.now()
feature_names = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']
file_names = ['ARG_MAZ_env_data.csv']
X, Y = feature_generator(feature_names, file_names)
end_time = datetime.datetime.now()

print(f'The runtime was {end_time - begin_time}.')
print(f'The number of entries in X_dict is {len(X_dict)}.')

# Organizes features/targets into X and Y numpy arrays referenced by timestamp and location
# Adds wind speed, MAP, MAT and species data to each feature entry?? Or uses a clustering algorithm for these??
# Use np.loadtext to find column names
# Read the first line and create a dictionary showing the position of each desired columns
# Move to a numpy array using
# Removes extraneous/irrelevant data
# Removes entries with missing data
# Splits data into training and validation sets

