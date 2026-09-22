import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utilities.site_merger import merge_groups

# This file creates clusters of sites sorted into sets related by climate or plant functional type

# Columns used as the k-means inputs, edit this list to try a different set
cluster_features = ['MAP', 'MAT', 'Average Wind Speed']


class ClusterCreator:

    def __init__(self):
        self.site_df = []  # working site dataframe
        self.k_cluster_dict = {}  # dictionary of site clusters based on K-Means labels
        self.func_cluster_dict = {}  # dictionary of site clusters by plant functional type
        self.biome_cluster_dict = {}  # dictionary of site clusters by biome

    def preprocess(self):
        # Compile csv with site name, MAP, MAT, average wind speed, functional type
        self.site_df = pd.read_csv('data/modeling_data/site_locations.csv')  # Use this .csv file
        site_list = self.site_df['Site'].values
        func_type_list = []

        # Add plant functional type
        types = pd.read_csv("data/site_types.csv")
        for site in site_list:
            for row in types.iterrows():
                if site == row[1]['Site']:
                    func_type = row[1]['Type']
                    if type(func_type) == str:
                        if 'deciduous' in func_type:
                            func_type = 0
                        elif 'evergreen' in func_type:
                            func_type = 1
                        elif 'mixed' in func_type:
                            func_type = 2
                    func_type_list.append(func_type)
        self.site_df['Functional Type'] = func_type_list

        # Add average wind speed
        wind_df = pd.read_csv('data/modeling_data/avg_wind_speed.csv')
        wind_df['Average Wind Speed'] = wind_df.iloc[:, 3:].mean(axis=1)
        wind_sites = wind_df['Site Name'].values
        wind_speeds = wind_df['Average Wind Speed'].values
        wind_dict = dict(zip(wind_sites, wind_speeds))
        for merged, sources in merge_groups.items():  # a merged site takes the wind of its sources, which
            present = [wind_dict[s] for s in sources if s in wind_dict]  # sit in one grid cell and so
            if present:                                                  # carry the same value
                wind_dict[merged] = sum(present) / len(present)
        wind_speeds = [wind_dict[site] for site in site_list]
        self.site_df['Average Wind Speed'] = wind_speeds

        # Print new data to csv
        self.site_df.to_csv('data/modeling_data/cluster_info.csv', index=False)

    def cluster_data(self):  # Returns the scaled k-means inputs
        self.site_df = pd.read_csv('data/modeling_data/cluster_info.csv')
        data = self.site_df[cluster_features].values
        return StandardScaler().fit_transform(data)

    def elbow_plot(self):  # Run on its own to choose k
        data = self.cluster_data()
        k_list = []
        inertia_list = []
        for k in range(2, 15):  # use to test different numbers of clusters
            kmeans = KMeans(n_clusters=k, n_init=6)
            kmeans.fit(data)
            k_list.append(k)
            inertia_list.append(kmeans.inertia_)
            print(f'Inertia when k = {k} was {kmeans.inertia_}.')
        plt.scatter(k_list, inertia_list)
        plt.xlabel('Number of clusters k')
        plt.ylabel('Inertia score')
        plt.savefig('data/k-means-elbow.png')
        plt.show()
        plt.clf()

    def k_means_clusters(self):  # Implement K-means to come up with clusters of similar climate statistics
        data = self.cluster_data()
        sites = self.site_df['Site'].values
        k = 7
        kmeans = KMeans(n_clusters=k, n_init=6)
        kmeans.fit(data)
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

    def func_type_clusters(self):  # Return dictionary of functional types with sites
        self.site_df = pd.read_csv('data/modeling_data/cluster_info.csv')
        sites = self.site_df['Site'].values
        func_types = self.site_df['Functional Type'].values
        func_type_unique = np.unique(func_types)
        func_type_unique = func_type_unique[np.logical_not(func_type_unique == 'missing')]  # remove missing value
        for func_type in func_type_unique:
            self.func_cluster_dict.update({int(func_type): []})
            for i, func in enumerate(func_types):
                if func == func_type:
                    self.func_cluster_dict[int(func_type)].append(sites[i])

    def biome_clusters(self):  # Return dictionary of biomes with sites
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

    @classmethod
    def build_clusters(cls):
        creator = ClusterCreator()
        creator.preprocess()
        creator.k_means_clusters()
        creator.func_type_clusters()
        creator.biome_clusters()
        return creator


if __name__ == "__main__":
    creator = ClusterCreator()
    creator.preprocess()
    creator.k_means_clusters()
    creator.func_type_clusters()
    creator.biome_clusters()
    # creator.elbow_plot()
