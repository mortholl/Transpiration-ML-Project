import pandas as pd
import numpy as np
from utilities.cluster_creator import ClassCreator
from utilities.data_sanitizer import feature_generator

#  Creates a support vector machine to predict sap flux from environmental variables

#  Import data
creator = ClassCreator()
creator.k_means_classes()
creator.func_type_classes()
creator.biome_classes()
k_clusters = creator.k_cluster_dict
func_clusters = creator.func_cluster_dict
biome_clusters = creator.biome_cluster_dict

features = ['ta', 'rh', 'vpd']  # Specify desired features
files = biome_clusters['Tropical rain forest']  # Can choose by biome, functional type, K-Means, or include all using []
feature_generator(features, files)
data = np.loadtxt('data/modeling_data/working_data.csv', skiprows=1, delimiter=',')
X = data[:, 0:-1]
Y = data[:, -1]

#  Split to training/validation sets

#  Build SVM models
