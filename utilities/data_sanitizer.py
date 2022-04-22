import pandas as pd
import numpy as np
import os
import datetime
from utilities.cluster_creator import ClusterCreator
import csv

# Removes entries with missing data
# Organizes features and targets into csv files


def data_import(feature_list, file_list, verbose=True):  # Returns two numpy arrays, x features and y targets
    # The feature names list must exactly match the column names in the SAPFLUXNET database
    # Can pass an empty list to use all files in the directory
    feature_directory = 'data/modeling_data/features'
    target_directory = 'data/modeling_data/targets'
    target = 'Average Sap Flux'
    x_dict = {}
    for feature in feature_list:
        x_dict.update({feature: []})
    y_dict = {target: []}

    for filename in os.listdir(feature_directory):   # Pull one location at a time from the features directory
        location = filename.split('_env')[0]
        if location in file_list or len(file_list) == 0:
            combined_df = pd.read_csv(feature_directory + '/' + filename, usecols=feature_list)
            target_df = pd.read_csv(target_directory + '/' + location +'_sapf_data.csv')
            target_df[target] = target_df.iloc[:, 2:].mean(axis=1)
            combined_df[target] = target_df[target]
            combined_df = combined_df.dropna()  # Removes rows with missing values
            # Remove large sap flux values that are likely errors
            combined_df = combined_df.drop(combined_df[combined_df[target] > 80000].index)
            for feature in feature_list:
                x_dict[feature].extend(np.ndarray.tolist(combined_df[feature].values))
            y_dict[target].extend(np.ndarray.tolist(combined_df[target].values))
            if verbose:
                print(f'{location} data loaded')
    df_out = pd.DataFrame()
    for feature in feature_list:
        value_list = x_dict[feature]
        df_out[feature] = value_list
    df_out['Average Sap Flux'] = y_dict['Average Sap Flux']
    df_out.to_csv('data/modeling_data/working_data.csv', index=False)
    data = np.loadtxt('data/modeling_data/working_data.csv', skiprows=1, delimiter=',')
    x = data[:, 0:-1]
    y = data[:, -1]
    if verbose:
        print(f'The number of data points is {len(x)}.')
    return x, y


# Test code below

# begin_time = datetime.datetime.now()
#
# cluster_creator = ClusterCreator.build_clusters()
# biome_clusters = cluster_creator.biome_cluster_dict
# k_clusters = cluster_creator.k_cluster_dict
#
# feature_names = ['ppfd_in']
# file_names = k_clusters[1]
# X, Y = data_import(feature_names, file_names, verbose=False)
#
# end_time = datetime.datetime.now()
# print(f'The runtime was {end_time - begin_time}.')

# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot()
# bp = ax.boxplot(X[:, 0], whis='range')
# plt.ylabel('PPFD')
# plt.show()
