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
    feature_directory = 'data/modeling_data/resampled/features'
    target_directory = 'data/modeling_data/resampled/targets'
    target = 'Average Sap Flux'
    site_frames = []

    for filename in os.listdir(feature_directory):   # Pull one location at a time from the features directory
        location = filename.split('_env')[0]
        if location in file_list or len(file_list) == 0:
            feature_df = pd.read_csv(feature_directory + '/' + filename, index_col='TIMESTAMP',
                                     usecols=['TIMESTAMP', 'interpolated'] + feature_list)
            target_df = pd.read_csv(target_directory + '/' + location + '_sapf_data.csv', index_col='TIMESTAMP')
            sensor_columns = [name for name in target_df.columns if name != 'interpolated']  # one column per tree

            target_frame = pd.DataFrame({target: target_df[sensor_columns].mean(axis=1),  # average sap flux at the site
                                         'target_interpolated': target_df['interpolated']})
            feature_df = feature_df.rename(columns={'interpolated': 'drivers_interpolated'})
            combined_df = feature_df.join(target_frame, how='inner')  # joined on timestamp rather than row position
            combined_df = combined_df.dropna(subset=feature_list + [target])  # Removes rows with missing values
            # Remove large sap flux values that are likely errors
            combined_df = combined_df.drop(combined_df[combined_df[target] > 80000].index)
            combined_df['Site'] = location
            site_frames.append(combined_df)
            if verbose:
                print(f'{location} data loaded')

    df_out = pd.concat(site_frames)
    df_out = df_out[['Site'] + feature_list + [target, 'drivers_interpolated', 'target_interpolated']]
    df_out.to_csv('data/modeling_data/working_data.csv')
    x = df_out[feature_list].values
    y = df_out[target].values
    if verbose:
        print(f'The number of data points is {len(x)}.')
    return x, y


# Test code below
# Guarded so that importing data_import does not rebuild the clusters and reload the data

if __name__ == "__main__":
    begin_time = datetime.datetime.now()

    cluster_creator = ClusterCreator.build_clusters()
    biome_clusters = cluster_creator.biome_cluster_dict
    k_clusters = cluster_creator.k_cluster_dict

    feature_names = ['ppfd_in']
    file_names = biome_clusters['Temperate forest']
    X, Y = data_import(feature_names, file_names, verbose=True)

    end_time = datetime.datetime.now()
    print(f'The runtime was {end_time - begin_time}.')

    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # bp = ax.boxplot(X[:, 0], whis='range')
    # plt.ylabel('PPFD')
    # plt.show()
