import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# This file creates clusters of locations sorted into sets related by either climate or plant functional type


class ClassCreator:
    def __init__(self):
        self.site_df = []  # working site dataframe
        self.k_cluster_dict = {}  # dictionary of site clusters based on K-Means labels
        self.func_cluster_dict = {}  # dictionary of site clusters by plant functional type
        self.biome_cluster_dict = {}  # dictionary of site clusters by biome

    def preprocess(self):
        # Compile csv with location name, MAP, MAT, average wind speed, and plant functional type
        self.site_df = pd.read_csv('data/modeling_data/site_locations.csv')  # Use this .csv file
        site_list = self.site_df['Unnamed: 0'].values
        func_type_list = []

        # Add plant functional type
        for filename in os.listdir('data/plant'):
            for site in site_list:
                if site in filename and '_species_md' in filename:
                    func_type_df = pd.read_csv('data/plant/'+filename)
                    func_type = func_type_df['sp_leaf_habit'].values[0]
                    if isinstance(func_type, str):
                        if 'deciduous' in func_type:
                            func_type = 0
                        elif 'evergreen' in func_type:
                            func_type = 1
                    func_type_list.append(func_type)
        self.site_df['Functional Type'] = func_type_list
        self.site_df = self.site_df.rename(columns={"Unnamed: 0": "Site"})

        # Add average wind speed
        wind_df = pd.read_csv('data/modeling_data/avg_wind_speed.csv')
        wind_df['Average Wind Speed'] = wind_df.iloc[:, 3:].mean(axis=1)
        wind_sites = wind_df['Site Name'].values
        wind_speeds = wind_df['Average Wind Speed'].values
        wind_dict = dict(zip(wind_sites, wind_speeds))
        wind_speeds = [wind_dict[site] for site in site_list]
        self.site_df['Average Wind Speed'] = wind_speeds
        self.site_df.to_csv('data/modeling_data/cluster_info.csv', index=False)

    def k_means_classes(self):  # Implement K-means to come up with clusters of similar climate statistics
        self.site_df = pd.read_csv('data/modeling_data/cluster_info.csv')
        sites = self.site_df['Site'].values
        maps = self.site_df['MAP'].values
        mats = self.site_df['MAT'].values
        wind_speeds = self.site_df['Average Wind Speed'].values
        data = np.asarray([maps, mats, wind_speeds])
        data = data.transpose()
        k = 5
        # for k in range(2, 15):  # use to test different numbers of clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=6)
        kmeans.fit(data)
        # print(f'Best inertia when k = {k} was {kmeans.score(data)}.')
        labels = kmeans.labels_
        self.site_df['K-Means Label'] = labels
        self.site_df.to_csv('data/modeling_data/cluster_info.csv', index=False)
        k_means_labels = self.site_df['K-Means Label'].values
        k_means = np.unique(k_means_labels)
        for k in k_means:
            self.k_cluster_dict.update({k: []})
            for i, k_mean in enumerate(k_means_labels):
                if k_mean == k:
                    self.k_cluster_dict[k].append(sites[i])

    def func_type_classes(self):  # Return dictionary of functional types with sites
        self.site_df = pd.read_csv('data/modeling_data/cluster_info.csv')
        sites = self.site_df['Site'].values
        func_types = self.site_df['Functional Type'].values
        func_type_unique = np.unique(func_types)
        func_type_unique = func_type_unique[np.logical_not(np.isnan(func_type_unique))]  # remove nan values
        for func_type in func_type_unique:
            self.func_cluster_dict.update({int(func_type): []})
            for i, func in enumerate(func_types):
                if func == func_type:
                    self.func_cluster_dict[int(func_type)].append(sites[i])

    def biome_classes(self):  # Return dictionary of biomes with sites
        self.site_df = pd.read_csv('data/modeling_data/cluster_info.csv')
        sites = self.site_df['Site'].values
        biomes = self.site_df['Biome'].values
        biomes_unique = []
        for biome in biomes:
            if biome not in biomes_unique:
                biomes_unique.append(biome)
        for biome in biomes_unique:
            self.biome_cluster_dict.update({biome: []})
            for i, b in enumerate(biomes):
                if b == biome:
                    self.biome_cluster_dict[biome].append(sites[i])


if __name__ == "__main__":
    creator = ClassCreator()
    creator.preprocess()
    creator.k_means_classes()
    creator.func_type_classes()
    creator.biome_classes()
