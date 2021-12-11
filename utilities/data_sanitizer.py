import pandas as pd
import numpy as np
import os
import datetime
import csv

# Removes entries with missing data
# Organizes features and targets into csv files


def feature_generator(feature_list, file_list):  # Returns two dictionaries, features and targets
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
            for feature in feature_list:
                x_dict[feature].extend(np.ndarray.tolist(combined_df[feature].values))
            y_dict[target].extend(np.ndarray.tolist(combined_df[target].values))
            print(f'{location} complete')
    df_out = pd.DataFrame()
    for feature in feature_list:
        value_list = x_dict[feature]
        df_out[feature] = value_list
    df_out['Average Sap Flux'] = y_dict['Average Sap Flux']
    df_out.to_csv('data/modeling_data/working_data.csv', index=False)
    return x_dict, y_dict  # optional return values, can be useful for debugging


begin_time = datetime.datetime.now()

# Test code below
# feature_names = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']
# file_names = ['ARG_MAZ']
# feature_generator(feature_names, file_names)

end_time = datetime.datetime.now()
print(f'The runtime was {end_time - begin_time}.')
