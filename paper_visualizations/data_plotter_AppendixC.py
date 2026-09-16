# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:50:38 2024

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

# url= 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/features/ARG_MAZ_env_data.csv?raw=true'
# df = pd.read_csv(url)

#fig, ax = plt.subplots(2, 2, figsize = (7.5,4.5))

fig, ax = plt.subplots(4, 1, figsize = (8,12))

ax1 = ax[0]


ax1.set_ylim(-30, 50)
ax1.set_ylabel('T$_a$ (C)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

ax2 = ax[1]
ax2.set_ylim(0,8)
ax2.set_ylabel('VPD (kPa)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

ax3 = ax[2]
ax3.set_ylim(0,0.7)
ax3.set_ylabel('SWC (-)')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

ax4 = ax[3]
ax4.set_ylim(0,3000)
ax4.set_ylabel('PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# feature_names = ['ppfd_in', 'swc_shallow', 'vpd', 'ta']
# file_names = biome_clusters['Temperate forest']
# X, Y = data_import(feature_names, file_names, verbose=True)

# ax1.violinplot([X], [0], showmeans = True, showextrema = True)

FT_list = ['Deciduous', 'Evergreen', 'Mixed']
Biome_list = ['Woodland/Shrubland', 'Temperate forest', 'Tropical rain forest', 'Tropical forest savanna', 'Subtropical desert', 'Boreal forest', 'Temperate grassland desert']

# ta_in = []
# for i in np.arange(15):
#     if sites['Functional Type'][i] == FT_list[0]:
#         url = 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/features/' + site_names[i] +'_env_data.csv?raw=true'
#         df = pd.read_csv(url)
#         ta_in = ta_in + df['ta'].values.tolist()
#     print('i=', i)
# ax1.violinplot(ta_in, [0], showmeans = True, showextrema = True)

wp = 0.75 #plotwidths

for j in [0, 1, 2]:
    ta_in = []
    vpd_in = []
    swc_in = []
    ppfd_in = []
    for i in np.arange(95):
        if sites['Functional Type'][i] == FT_list[j]:
            url = 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/features/' + site_names[i] +'_env_data.csv?raw=true'
            df = pd.read_csv(url)
            df = df.dropna() # gets rid of rows with missing values
            ta_in = ta_in + df['ta'].values.tolist()
            vpd_in =vpd_in + df['vpd'].values.tolist()
            swc_in = swc_in + df['swc_shallow'].values.tolist()
            ppfd_in = ppfd_in + df['ppfd_in'].values.tolist()
        print('i=', i)
    ax1.violinplot(ta_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax2.violinplot(vpd_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax3.violinplot(swc_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax4.violinplot(ppfd_in, [j], widths = wp, showmeans = True, showextrema = True)
    print('j=', j)
    
for j in np.arange(3,10):
    ta_in = []
    vpd_in = []
    swc_in = []
    ppfd_in = []
    for i in np.arange(95):
        if sites['Biome'][i] == Biome_list[j-3]:
            url = 'https://github.com/mortholl/Transpiration-ML-Project/blob/main/data/modeling_data/features/' + site_names[i] +'_env_data.csv?raw=true'
            df = pd.read_csv(url)
            df = df.dropna() # gets rid of rows with missing values
            ta_in = ta_in + df['ta'].values.tolist()
            vpd_in =vpd_in + df['vpd'].values.tolist()
            swc_in = swc_in + df['swc_shallow'].values.tolist()
            ppfd_in = ppfd_in + df['ppfd_in'].values.tolist()
        print('i=', i)
    ax1.violinplot(ta_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax2.violinplot(vpd_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax3.violinplot(swc_in, [j], widths = wp, showmeans = True, showextrema = True)
    ax4.violinplot(ppfd_in, [j], widths = wp, showmeans = True, showextrema = True)
    print('j=', j)
    
    
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax1.set_xticklabels([])

ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax2.set_xticklabels([])

ax3.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax3.set_xticklabels([])
# ax3.set_xticklabels(['Deciduous', 'Evergreen', 'Mixed', 'Woodland/Shrubland', 'Temperate forest', 'Tropical rain forest', 'Tropical forest savanna', 'Subtropical desert', 'Boreal forest', 'Temperate grassland \n desert'], rotation = 45)

ax4.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax4.set_xticklabels(['Deciduous', 'Evergreen', 'Mixed', 'Woodland/Shrubland', 'Temperate forest', 'Tropical rain forest', 'Tropical forest savanna', 'Subtropical desert', 'Boreal forest', 'Temperate grassland \n desert'], rotation = 45)

ax1.text(-1.5, 52, 'a', fontweight = 'bold', fontsize = 'large')
ax2.text(-1.5, 8.5, 'b', fontweight = 'bold', fontsize = 'large')
ax3.text(-1.5, 0.74, 'c', fontweight = 'bold', fontsize = 'large')
ax4.text(-1.5, 3150, 'd', fontweight = 'bold', fontsize = 'large')

fig.tight_layout()
fig.savefig('site_stats_AppendixC_column.png')