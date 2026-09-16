# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:46:46 2024

@author: Samantha
"""

# import data_sanitizer
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
# from cluster_creator import ClusterCreator
import csv

sites = pd.read_csv('AppendixA_dataclusters_table.csv')

site_names= sites['Site']

target = 'Average Sap Flux'

fig, ax = plt.subplots(1, 1, figsize = (7.5,4.5))

ax1 = ax

ax1.set_ylim(0, 8000)
ax1.set_ylabel('Sap flux (cm$^3$ h$^{-1}$)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# feature_names = ['ppfd_in', 'swc_shallow', 'vpd', 'ta']
# file_names = biome_clusters['Temperate forest']
# X, Y = data_import(feature_names, file_names, verbose=True)

# ax1.violinplot([X], [0], showmeans = True, showextrema = True)

FT_list = ['Deciduous', 'Evergreen', 'Mixed']
Biome_list = ['Woodland/Shrubland', 'Temperate forest', 'Tropical rain forest', 'Tropical forest savanna', 'Subtropical desert', 'Boreal forest', 'Temperate grassland desert']
    
wp = 0.75 #plotwidths

for j in [0, 1, 2]:
    sf = []

    for i in np.arange(95):
        if sites['Functional Type'][i] == FT_list[j]:
            url = 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/targets/' + site_names[i] +'_sapf_data.csv?raw=true'
            df = pd.read_csv(url)
            df = df.dropna() # gets rid of rows with missing values
            df[target] = df.iloc[:, 2:].mean(axis=1)  # takes the average sap flux at the site
            # Remove large sap flux values that are likely errors, and negative sap flux values
            df = df.drop(df[df[target] > 80000].index)
            df = df.drop(df[df[target] < 0].index)
            sf = sf + df[target].values.tolist()

        print('i=', i)
    ax1.violinplot(sf, [j], widths = wp, showmeans = True, showextrema = True)

    print('j=', j)
    
for j in np.arange(3,10):
    sf = []
    
    for i in np.arange(95):
        if sites['Biome'][i] == Biome_list[j-3]:
            url = 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/targets/' + site_names[i] +'_sapf_data.csv?raw=true'
            df = pd.read_csv(url)
            df = df.dropna() # gets rid of rows with missing values
            df[target] = df.iloc[:, 2:].mean(axis=1)  # takes the average sap flux at the site
            # Remove large sap flux values that are likely errors, and negative sap flux values
            df = df.drop(df[df[target] > 80000].index)
            df = df.drop(df[df[target] < 0].index)
            sf = sf + df[target].values.tolist()
        print('i=', i)
    ax1.violinplot(sf, [j], widths = wp, showmeans = True, showextrema = True)
    print('j=', j)



ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax1.set_xticklabels(['Deciduous', 'Evergreen', 'Mixed', 'Woodland/Shrubland', 'Temperate forest', 'Tropical rain forest', 'Tropical forest savanna', 'Subtropical desert', 'Boreal forest', 'Temperate grassland \n desert'], rotation = 45)

fig.tight_layout()
fig.savefig('site_sapf_AppendixC_full_cropped.png')