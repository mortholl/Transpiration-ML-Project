# -*- coding: utf-8 -*-
"""
Model performance figures for the revised paper, RF vs NN, from the re-trained results.

Replaces the performance figures in 'SAPFLUXNET transpiration ML project visualizations for paper.ipynb',
which is an archive of the original figures:
    cell 2, panel a (R2 violins, PFT vs biome)   ->  model_performance_cluster_r2.png
    new, site-level test R2                      ->  model_performance_site_r2.png
    cell 3 (numsites_r2.png)                     ->  model_performance_numlocations_r2.png
Clusters are only compared within panels of similar size (number of locations), per the review.
Feature importance figures are handled separately.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import uncertainties as unc
import uncertainties.unumpy as unp

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)

RF = pd.read_csv(os.path.join(project, 'RandomForest', 'rf_results.csv'), skipinitialspace=True)
NN = pd.read_csv(os.path.join(project, 'Neural_Networks', 'ann_results.csv'), skipinitialspace=True)

# The cluster name the model, test set and site files share, e.g. 'pft_deciduous' or 'biome_Temperate forest'
RF['key'] = RF['Data set'].str.replace('_rf$', '', regex=True)
NN['key'] = NN['Data set'].str.replace('_ann$', '', regex=True)

RF_PFT = RF[RF['key'].str.startswith('pft_')]
RF_biome = RF[RF['key'].str.startswith('biome_')]
NN_PFT = NN[NN['key'].str.startswith('pft_')]
NN_biome = NN[NN['key'].str.startswith('biome_')]

# One row per cluster with both methods side by side, largest cluster first
RF_trim = RF.merge(NN, on='key', suffixes=('_rf', '_nn')).sort_values('n locations_rf', ascending=False)

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

# Panels of similarly sized clusters, by number of locations
size_panels = [
    ('Large (14-21 locations)', lambda n: n >= 14),
    ('Medium (5-9 locations)', lambda n: (n >= 5) & (n < 14)),
    ('Small (1-2 locations)', lambda n: n < 5),
]
panel_widths = [4, 2, 2]

color1 = '#377eb8'  # RF
color2 = '#4daf4a'  # NN
hatch1 = '.'
hatch2 = '/'


def cluster_label(row):  # 'Temperate\nforest\n(21 loc., 25 sites)'
    return f"{cluster_names[row['key']]}\n({row['n locations_rf']} loc., {row['n sites_rf']} sites)"


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------------------------- cluster test R2
barWidth = 0.36

fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, gridspec_kw={'width_ratios': panel_widths})

for ax1, (title, in_panel) in zip(ax, size_panels):
    panel = RF_trim[in_panel(RF_trim['n locations_rf'])]
    r1 = np.arange(len(panel))
    ax1.bar(r1 - barWidth / 2, panel['R2 test_rf'], barWidth, color=color1, hatch=hatch1, edgecolor='k', label='RF')
    ax1.bar(r1 + barWidth / 2, panel['R2 test_nn'], barWidth, color=color2, hatch=hatch2, edgecolor='k', label='ANN')
    ax1.set_xticks(r1)
    ax1.set_xticklabels([cluster_label(row) for _, row in panel.iterrows()], fontsize=7.5)
    ax1.set_title(title, fontsize=10)
    tidy(ax1)

ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel('Test set R$^2$')
ax[0].legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.02))

fig.tight_layout()
fig.savefig(os.path.join(folder, 'model_performance_cluster_r2.png'), dpi=300)

# ---------------------------------------------------------------------------------------------- site test R2
# Every site's test rows are scored separately; sites belong to one PFT and one biome cluster
site_r2 = []
for key in RF_trim['key']:
    rf_sites = pd.read_csv(os.path.join(project, 'RandomForest', 'test_sets', f'{key}_rf_site_r2.csv'))
    nn_sites = pd.read_csv(os.path.join(project, 'Neural_Networks', 'test_sets', f'{key}_ann_site_r2.csv'))
    sites = rf_sites.merge(nn_sites, on='Site', suffixes=('_rf', '_nn'))
    sites['key'] = key
    site_r2.append(sites)
site_r2 = pd.concat(site_r2, ignore_index=True)

r2_floor = -1  # a few sites fall far below zero (down to about -40); they are drawn at the floor and counted
rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible

fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharey=True, gridspec_kw={'width_ratios': panel_widths})

for ax1, (title, in_panel) in zip(ax, size_panels):
    panel = RF_trim[in_panel(RF_trim['n locations_rf'])]
    for i, key in enumerate(panel['key']):
        sites = site_r2[site_r2['key'] == key]
        for offset, column, color in ((-0.18, 'r2_rf', color1), (0.18, 'r2_nn', color2)):
            y = sites[column].clip(lower=r2_floor)
            ax1.boxplot(y, positions=[i + offset], widths=0.3, showfliers=False,
                        medianprops=dict(color='k'), boxprops=dict(color=color))
            ax1.scatter(i + offset + rng.uniform(-0.06, 0.06, len(y)), y, s=10, color=color, alpha=0.8, zorder=3)
            n_clipped = (sites[column] < r2_floor).sum()
            if n_clipped:
                ax1.annotate(f'{n_clipped}↓', (i + offset, r2_floor - 0.08), ha='center', fontsize=7)
    ax1.set_xticks(range(len(panel)))
    ax1.set_xticklabels([cluster_label(row) for _, row in panel.iterrows()], fontsize=7.5)
    ax1.set_title(title, fontsize=10)
    ax1.axhline(0, color='grey', linewidth=0.5)
    tidy(ax1)

ax[0].set_ylim(r2_floor - 0.15, 1)
ax[0].set_ylabel('Site test R$^2$ (clipped at -1)')
ax[-1].legend([Line2D([], [], marker='o', ls='', color=color1), Line2D([], [], marker='o', ls='', color=color2)],
              ['RF', 'ANN'], frameon=False, loc='lower right')

fig.tight_layout()
fig.savefig(os.path.join(folder, 'model_performance_site_r2.png'), dpi=300)

# ---------------------------------------------------------------------------------------------- R2 vs number of locations
def f(x, a, b):
    return a * x + b


def conf_int(x_data, y_data, px):  # as in the notebook, also returning popt for the printed fit
    popt, pcov = curve_fit(f, x_data, y_data)
    # calculate parameter confidence interval
    a, b = unc.correlated_values(popt, pcov)
    # calculate regression confidence interval
    py = a * px + b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    return popt, nom, std


px = np.arange(0, 25, 1)

fig, pt = plt.subplots(figsize=(7.5, 4.5))

for label, PFT, biome, data, color in (('RF', RF_PFT, RF_biome, RF, color1), ('ANN', NN_PFT, NN_biome, NN, color2)):
    popt, nom, std = conf_int(data['n locations'], data['R2 test'], px)
    y_pred = f(data['n locations'], *popt)
    print(f'{label}: slope {popt[0]:.4f} per location, intercept {popt[1]:.3f}, '
          f'R2 = {r2_score(data["R2 test"], y_pred):.3f}')

    pt.plot(PFT['n locations'], PFT['R2 test'], 's', color=color, label=f'{label}, PFT')
    pt.plot(biome['n locations'], biome['R2 test'], '^', color=color, label=f'{label}, biome')
    pt.plot(px, nom, c=color)
    pt.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, color=color, alpha=0.3)  # 95% confidence interval

pt.set_ylim(0, 1)
pt.set_xlim(0, 24)
pt.set_xlabel('Number of locations')
pt.set_ylabel('Test set R$^2$')
tidy(pt)
pt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
pt.legend(bbox_to_anchor=(1, 0.5), loc='lower left', fontsize=10, bbox_transform=pt.transAxes, framealpha=1.0)

fig.tight_layout()
fig.savefig(os.path.join(folder, 'model_performance_numlocations_r2.png'), dpi=300)
