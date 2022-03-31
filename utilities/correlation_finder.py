from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict


# Pick relevant features
my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow']
cor_all = []
model_list = []

for identifier, cluster_group in zip(['k_means_', 'func_', 'biome_'], [k_clusters, func_clusters, biome_clusters]):
    for data_cluster in cluster_group:
        # Get data
        my_files = cluster_group[data_cluster]
        n_files = len(my_files)
        model_name = f'{identifier}{data_cluster}'
        model_name = model_name.replace('/', '')
        model_list.append(model_name)
        X, Y = data_import(my_features, my_files)
        ta = X[:, 0]
        cor_list = []
        for i in range(len(my_features)):
            cor, p = spearmanr(X[:, i], Y)
            cor_list.append(cor)
        cor_all.append(cor_list)

model_list.append('All')
cor_list = []
X, Y = data_import(my_features, [])
for i in range(len(my_features)):
    cor, p = spearmanr(X[:, i], Y)
    cor_list.append(cor)
cor_all.append(cor_list)
cor_all = np.asarray(cor_all)
np.transpose(cor_all)

cor_df = pd.DataFrame({'Model Name': model_list, 'Ta': cor_all[:, 0], 'RH': cor_all[:, 1], 'VPD': cor_all[:, 2],
                       'PPFD': cor_all[:, 3], 'SWC': cor_all[:, 4]})
cor_df.to_csv('data/modeling_data/spearman_correlations.csv', index=False)
