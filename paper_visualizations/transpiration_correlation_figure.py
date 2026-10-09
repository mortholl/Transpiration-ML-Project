# -*- coding: utf-8 -*-
"""
Spearman correlation between each model input and sap flux, for all data and for every PFT and biome cluster.

Replaces cell 8 of 'SAPFLUXNET transpiration ML project visualizations for paper.ipynb' (transpiration_correlation.tiff),
whose correlations were typed in from an earlier set of clusters. Loads the data, so it is slow; run it once from any
folder. The correlations are also written to data/modeling_data/transpiration_correlation.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)
os.chdir(project)  # data_import reads the data folders relative to the project
sys.path.insert(0, project)

from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key

# Must match the feature list in the optimization scripts
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
feature_labels = ['T$_a$', 'VPD', 'PPFD', 'SWC']
output_table = os.path.join(project, 'data', 'modeling_data', 'transpiration_correlation.csv')

cluster_names = {
    'all': 'All data',
    'pft_deciduous': 'Deciduous',
    'pft_evergreen': 'Evergreen',
    'pft_mixed': 'Mixed',
    'biome_WoodlandShrubland': 'Woodland/Shrubland',
    'biome_Temperate forest': 'Temperate forest',
    'biome_Tropical rain forest': 'Tropical rainforest',
    'biome_Tropical forest savanna': 'Tropical forest savanna',
    'biome_Temperate grassland desert': 'Grassland desert',
}


def correlations(X, Y):
    return [spearmanr(X[:, i], Y)[0] for i in range(len(my_features))]


cluster_creator = ClusterCreator.build_clusters()
rows = []
X, Y = data_import(my_features, [], verbose=False)  # every site
rows.append({'Data set': 'all', 'n locations': np.nan, **dict(zip(my_features, correlations(X, Y)))})
for identifier, cluster_group in zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict,
                                                         cluster_creator.biome_cluster_dict]):
    for data_cluster in cluster_group:
        X, Y, info = data_import(my_features, cluster_group[data_cluster], verbose=False, return_info=True)
        rows.append({'Data set': cluster_key(identifier, data_cluster), 'n locations': info['Location'].nunique(),
                     **dict(zip(my_features, correlations(X, Y)))})
        print(f'{rows[-1]["Data set"]} complete', flush=True)
table = pd.DataFrame(rows)
table.to_csv(output_table, index=False)

# All data first, then the PFT and the biome clusters, each largest first as in the performance figures
table['group'] = table['Data set'].str.split('_').str[0].map({'all': 0, 'pft': 1, 'biome': 2})
table = table.sort_values(['group', 'n locations'], ascending=[True, False])

matrix = table[my_features].to_numpy()
fig = plt.figure(figsize=(5, 5))
plt.imshow(matrix, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(label='Spearman correlation with sap flux')
for (i, j), value in np.ndenumerate(matrix):
    plt.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8)
plt.xticks(range(len(my_features)), feature_labels, rotation=45, ha='right')
plt.yticks(range(len(table)), [cluster_names[key] for key in table['Data set']])
fig.tight_layout()
fig.savefig(os.path.join(folder, 'transpiration_correlation.png'), dpi=300)
