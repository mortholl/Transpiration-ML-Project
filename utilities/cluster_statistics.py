from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key
import pandas as pd

# Climate and data averages for every PFT and biome cluster, and for every site within each cluster, so the
# feature importance figures can plot importance against climate without loading the data themselves
# The averages are taken over the rows the models were trained and tested on, after the location cap; a site
# can keep a different number of rows in its PFT and its biome cluster, so the site table is keyed by both
# MAP and MAT are the site values in cluster_info.csv, a cluster's are the average over its sites
# Run from the project folder after site_merger.py

cluster_file = 'data/modeling_data/cluster_statistics.csv'
site_file = 'data/modeling_data/site_statistics.csv'

# Must match the feature list in the optimization scripts
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
average_names = {feature: feature + '_avg' for feature in my_features}  # the column names the notebook used

if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    site_climate = pd.read_csv('data/modeling_data/cluster_info.csv', index_col='Site')[['MAP', 'MAT']]
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    cluster_rows, site_tables = [], []
    for identifier, cluster_group in groups:  # the same clusters the optimization scripts loop over, k-means
        for data_cluster in cluster_group:    # is left out, its labels move between runs
            key = cluster_key(identifier, data_cluster)
            X, Y, info = data_import(my_features, cluster_group[data_cluster], verbose=False, return_info=True)
            data = pd.DataFrame(X, columns=my_features).rename(columns=average_names)
            data['Sap flux average'] = Y
            data['Site'] = info['Site'].values

            sites = data.groupby('Site').mean().join(site_climate)
            sites.insert(0, 'Data set', key)
            site_tables.append(sites.reset_index())

            cluster = data.drop(columns='Site').mean()
            cluster_rows.append({'Data set': key, 'n sites': len(sites), 'MAP': sites['MAP'].mean(),
                                 'MAT': sites['MAT'].mean(), **cluster.to_dict()})
            print(f'{key} complete', flush=True)

    pd.DataFrame(cluster_rows).to_csv(cluster_file, index=False)
    pd.concat(site_tables, ignore_index=True).to_csv(site_file, index=False)
    print(f'Written to {cluster_file} and {site_file}')
