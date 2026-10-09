# -*- coding: utf-8 -*-
"""
Site level feature importance figures, RF vs NN, one figure per model input, and all four stacked as panels a to d
(site_feature_importance_all.png).

Drawn like the site test R2 figure in model_performance_figures.py: every site's held out rows are scored
separately, clusters sit in panels of similar size (number of locations), RF and NN side by side.
Each model's boxes show the sites where that model has site test R2 above 0.5, the sites its importances are
tested on between clusters.
Reads data/modeling_data/site_feature_importance_r2_05.csv, feature_importance.csv and cluster_statistics.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)
modeling_data = os.path.join(project, 'data', 'modeling_data')

my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
feature_labels = {'ta': 'T$_a$', 'vpd': 'VPD', 'ppfd_in': 'PPFD', 'swc_shallow': 'SWC'}

site_importance = pd.read_csv(os.path.join(modeling_data, 'site_feature_importance_r2_05.csv'))
site_importance = site_importance[site_importance['above floor']]

# One row per cluster with its size, largest cluster first
clusters = pd.read_csv(os.path.join(modeling_data, 'feature_importance.csv'))
clusters = clusters[clusters['Model'] == 'RF'][['Data set', 'n locations']]
clusters = clusters.merge(pd.read_csv(os.path.join(modeling_data, 'cluster_statistics.csv'))[['Data set', 'n sites']])
clusters = clusters.rename(columns={'Data set': 'key'}).sort_values('n locations', ascending=False)
for model in ('RF', 'ANN'):
    kept = site_importance[site_importance['Model'] == model].groupby('Data set')['Site'].nunique()
    clusters[f'n sites {model}'] = clusters['key'].map(kept).fillna(0).astype(int)

cluster_names = {
    'pft_deciduous': 'Deciduous',
    'pft_evergreen': 'Evergreen',
    'pft_mixed': 'Mixed',
    'biome_WoodlandShrubland': 'Woodland/\nShrubland',
    'biome_Temperate forest': 'Temperate\nforest',
    'biome_Tropical rain forest': 'Tropical\nrainforest',
    'biome_Tropical forest savanna': 'Tropical forest\nsavanna',
    'biome_Temperate grassland desert': 'Grassland\ndesert',
}

# Panels of similarly sized clusters, by number of locations, as in the performance figures
size_panels = [
    ('Large (14-21 locations)', lambda n: n >= 14),
    ('Medium (5-9 locations)', lambda n: (n >= 5) & (n < 14)),
    ('Small (1-2 locations)', lambda n: n < 5),
]
panel_widths = [4, 2, 2]

color1 = '#0072B2'  # RF, Okabe-Ito blue (colour blind safe)
color2 = '#D55E00'  # NN, Okabe-Ito vermillion


def cluster_label(row):  # 'Temperate\nforest\n(RF 13, NN 12\nof 25 sites)', the sites shown for each model out of
    return (f"{cluster_names[row['key']]}\n(RF {row['n sites RF']}, NN {row['n sites ANN']}\n"  # the sites in the
            f"of {row['n sites']} sites)")                                                      # cluster


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_feature(ax, feature, titles=True, ticklabels=True, legend=True):  # ax is a row of the three size panels
    rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible
    for ax1, (title, in_panel) in zip(ax, size_panels):
        panel = clusters[in_panel(clusters['n locations'])]
        for i, key in enumerate(panel['key']):
            sites = site_importance[site_importance['Data set'] == key]
            for offset, model, color in ((-0.18, 'RF', color1), (0.18, 'ANN', color2)):
                y = sites.loc[sites['Model'] == model, feature]
                ax1.boxplot(y, positions=[i + offset], widths=0.3, showfliers=False,
                            medianprops=dict(color='k'), boxprops=dict(color=color))
                ax1.scatter(i + offset + rng.uniform(-0.06, 0.06, len(y)), y, s=10, color=color, alpha=0.8, zorder=3)
        ax1.set_xticks(range(len(panel)))
        ax1.set_xticklabels([cluster_label(row) for _, row in panel.iterrows()] if ticklabels else [], fontsize=7.5)
        ax1.set_xlim(-0.5, len(panel) - 0.5)
        if titles:
            ax1.set_title(title, fontsize=10)
        tidy(ax1)

    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel(f'Site {feature_labels[feature]} feat. import. (-)')
    if legend:
        ax[-1].legend([Line2D([], [], marker='o', ls='', color=color1),
                       Line2D([], [], marker='o', ls='', color=color2)],
                      ['RF', 'NN'], frameon=False, loc='upper right', fontsize=8)


for feature in my_features:
    fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharey=True, gridspec_kw={'width_ratios': panel_widths})
    plot_feature(ax, feature)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f'site_feature_importance_{feature}.png'), dpi=300)

# ---------------------------------------------------------------------------------------------- all four combined
# One row per input, a to d, on one grid so the size panels line up; every row shows the same sites, so the size
# titles are only on the top row and the cluster labels only on the bottom row, as in feature_importance_clusters.png
fig, ax = plt.subplots(len(my_features), 3, figsize=(11, 11), sharex='col', sharey=True,
                       gridspec_kw={'width_ratios': panel_widths})
for row, (feature, letter) in enumerate(zip(my_features, 'abcd')):
    plot_feature(ax[row], feature, titles=row == 0, ticklabels=row == len(my_features) - 1, legend=row == 0)
    ax[row, 0].text(-0.25, 1.05, letter, transform=ax[row, 0].transAxes, fontweight='bold', fontsize='large')
fig.tight_layout()
fig.align_ylabels(ax[:, 0])
fig.savefig(os.path.join(folder, 'site_feature_importance_all.png'), dpi=300, bbox_inches='tight')  # keeps the letters
