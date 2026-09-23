import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import

# Draws the held out test rows once per cluster and writes them where the model scripts read them
# rf_optimization.py and ann_optimization.py share one split, so both models are scored on the same rows
# Run after site_merger.py and before either optimization script

split_directory = 'data/modeling_data/splits'
test_size = 0.1
seed = 51

# Must match the feature list in the optimization scripts, a different list drops a different set of rows
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']


def cluster_key(identifier, cluster):  # the name the optimization scripts build their model names from
    return f'{identifier}{cluster}'.replace('/', '')


def split_path(key):
    return f'{split_directory}/{key}_test.csv'


def build_split(key, files, features, verbose=False):  # Writes one cluster's held out rows, returns all
    x, y, info = data_import(features, files, verbose=verbose, return_info=True)  # of them labelled
    if 'TIMESTAMP' not in info.columns:
        raise ValueError('data_import no longer returns a TIMESTAMP column, the split cannot be keyed')
    train_rows, test_rows = train_test_split(np.arange(len(info)), test_size=test_size, random_state=seed)
    info = info.copy()
    info['split'] = 'train'
    info.loc[test_rows, 'split'] = 'test'
    os.makedirs(split_directory, exist_ok=True)
    held_out = info.loc[info['split'] == 'test', ['Site', 'TIMESTAMP']]  # only the test rows are stored,
    held_out.to_csv(split_path(key), index=False)  # everything else is training data by omission
    return info


def load_test_mask(key, info):  # Returns a boolean array marking the rows of info that were held out
    held_out = pd.read_csv(split_path(key), dtype=str)
    held_out = set(zip(held_out['Site'], held_out['TIMESTAMP']))
    mask = np.array([(site, stamp) in held_out for site, stamp in zip(info['Site'], info['TIMESTAMP'])])
    if mask.sum() != len(held_out):  # the feature list or the site data changed after the split was drawn
        raise ValueError(f'{key}: {len(held_out)} rows were held out but {mask.sum()} are still in the '
                         f'data, rerun test_set_builder.py')
    return mask


def summarise(key, info):  # Returns one row describing the cluster and one row per site within it
    counts = info.groupby(['Site', 'split']).size().unstack(fill_value=0)
    for column in ('train', 'test'):
        if column not in counts.columns:
            counts[column] = 0
    counts = counts[['train', 'test']].reset_index().sort_values('test', ignore_index=True)
    counts['test share'] = (counts['test'] / (counts['train'] + counts['test'])).round(4)
    counts['test days'] = (counts['test'] / 48).round(1)  # half hourly, so 48 rows make a day
    counts.insert(0, 'Data set', key)
    overall = {'Data set': key,
               'n sites': info['Site'].nunique(),
               'n locations': info['Location'].nunique(),
               'n rows': len(info),
               'n train': int((info['split'] == 'train').sum()),
               'n test': int((info['split'] == 'test').sum()),
               'smallest site test rows': int(counts['test'].min()),
               'median site test rows': int(counts['test'].median()),
               'sites under 100 test rows': int((counts['test'] < 100).sum())}
    return overall, counts


def build_all(verbose=True):  # Writes a split for every cluster the optimization scripts loop over
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['func_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    overall_rows, site_rows = [], []
    for identifier, cluster_group in groups:  # k_clusters is left out, KMeans has no seed so its labels
        for cluster in cluster_group:         # move between runs and a stored split would not survive one
            key = cluster_key(identifier, cluster)
            info = build_split(key, cluster_group[cluster], my_features)
            overall, counts = summarise(key, info)
            overall_rows.append(overall)
            site_rows.append(counts)
            if verbose:
                print(f'{key}: {overall["n train"]} train and {overall["n test"]} test rows over '
                      f'{overall["n sites"]} sites, smallest site has {overall["smallest site test rows"]}')
    pd.DataFrame(overall_rows).to_csv(f'{split_directory}/test_set_summary.csv', index=False)
    pd.concat(site_rows, ignore_index=True).to_csv(f'{split_directory}/test_set_by_site.csv', index=False)


if __name__ == "__main__":
    build_all()
