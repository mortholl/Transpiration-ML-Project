from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict

my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']

avg_dict = {}
var_dict = {}

for k, k_cluster in k_clusters.items():
    X, Y = data_import(my_features, k_cluster)
    feat_avg = [float(x) for x in np.average(X, axis=0)]
    feat_var = [float(x) for x in np.var(X, axis=0)]
    sf_avg = float(np.average(Y))
    sf_var = float(np.var(Y))
    avg_dict.update({k: [feat_avg, sf_avg]})
    var_dict.update({k: [feat_var, sf_var]})

for func, func_cluster in func_clusters.items():
    X, Y = data_import(my_features, func_cluster)
    feat_avg = [float(x) for x in np.average(X, axis=0)]
    feat_var = [float(x) for x in np.var(X, axis=0)]
    sf_avg = float(np.average(Y))
    sf_var = float(np.var(Y))
    if func == 0:
        func = "deciduous"
    if func == 1:
        func = "evergreen"
    if func == 2:
        func = "mixed"
    avg_dict.update({func: [feat_avg, sf_avg]})
    var_dict.update({func: [feat_var, sf_var]})

for biome, biome_cluster in biome_clusters.items():
    X, Y = data_import(my_features, biome_cluster)
    feat_avg = [float(x) for x in np.average(X, axis=0)]
    feat_var = [float(x) for x in np.var(X, axis=0)]
    sf_avg = float(np.average(Y))
    sf_var = float(np.var(Y))
    avg_dict.update({biome: [feat_avg, sf_avg]})
    var_dict.update({biome: [feat_var, sf_var]})

avg_df = pd.DataFrame(avg_dict)
var_df = pd.DataFrame(var_dict)
df = pd.concat([avg_df, var_df], axis=0)
df = df.T

df.to_csv("data/modeling_data/cluster_statistics.csv",
          header=['Feature average', 'Sap flux average', 'Feature variance', 'Sap flux variance'])
