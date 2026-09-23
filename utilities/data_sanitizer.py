import pandas as pd
import numpy as np
import datetime
from utilities.cluster_creator import ClusterCreator
import csv

# Removes entries with missing data
# Organizes features and targets into csv files


def cap_by_location(df, quantile, seed, verbose):
    # Cap each location at the cluster's quantile of location record lengths
    cap = int(df.groupby('Location').size().quantile(quantile))
    rng = np.random.default_rng(seed)
    kept = []
    for name, rows in df.groupby('Location'):
        if len(rows) <= cap:
            kept.append(rows)
        else:  # thin at random, keeping what survives in time order
            kept.append(rows.iloc[np.sort(rng.choice(len(rows), cap, replace=False))])
    out = pd.concat(kept)
    if verbose:
        print(f'Location cap is {cap} rows, {len(df)} rows down to {len(out)}.')
    return out


def data_import(feature_list, file_list, verbose=True, cap_quantile=0.75, seed=51, return_info=False):
    # Returns two numpy arrays, x features and y targets, and the site and location of every row
    # The feature names list must exactly match the column names in the SAPFLUXNET database
    # Can pass an empty list to use every site in site_locations.csv
    feature_directory = 'data/modeling_data/resampled/features'
    target_directory = 'data/modeling_data/resampled/targets'
    target = 'Average Sap Flux'
    site_locations = pd.read_csv('data/modeling_data/site_locations.csv',
                                 index_col='Site')['Location']  # co-located sites share one name
    site_frames = []

    # Pull one site at a time from the site list rather than the directory, so a file left behind by an
    # earlier site list or folded into a merge by site_merger.py is never read
    for site in site_locations.index:
        if site in file_list or len(file_list) == 0:
            feature_df = pd.read_csv(f'{feature_directory}/{site}_env_data.csv', index_col='TIMESTAMP',
                                     usecols=['TIMESTAMP', 'interpolated'] + feature_list)
            target_df = pd.read_csv(target_directory + '/' + site + '_sapf_data.csv', index_col='TIMESTAMP')
            sensor_columns = [name for name in target_df.columns if name != 'interpolated']  # one column per tree
            target_df[sensor_columns] = target_df[sensor_columns].mask(target_df[sensor_columns] > 200)  # a
            # reading above 200 cm/h is instrument error

            target_frame = pd.DataFrame({target: target_df[sensor_columns].mean(axis=1),  # average sap flux at the site
                                         'target_interpolated': target_df['interpolated']})
            feature_df = feature_df.rename(columns={'interpolated': 'drivers_interpolated'})
            combined_df = feature_df.join(target_frame, how='inner')  # joined on timestamp rather than row position
            combined_df = combined_df.dropna(subset=feature_list + [target])  # Removes rows with missing values
            if len(combined_df) < 336:  # drops sites with less than a week of valid data
                continue
            if (combined_df[target] < 0).mean() > 0.5:  # a record more than half below zero is not net
                continue  # water transport, it is a zero flow baseline set too high
            combined_df['Site'] = site
            combined_df['Location'] = site_locations[site]
            site_frames.append(combined_df)
            if verbose:
                print(f'{site} data loaded')

    df_out = pd.concat(site_frames)
    df_out = cap_by_location(df_out, cap_quantile, seed, verbose)
    df_out = df_out[['Site', 'Location'] + feature_list + [target, 'drivers_interpolated', 'target_interpolated']]
    x = df_out[feature_list].values
    y = df_out[target].values
    if verbose:
        print(f'The number of data points is {len(x)}.')
    if return_info:
        return x, y, df_out[['Site', 'Location']].reset_index()
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
