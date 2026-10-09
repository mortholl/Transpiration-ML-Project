# -*- coding: utf-8 -*-
"""
Feature importance figures for the revised paper, RF vs NN, from the permutation feature importances.

Replaces the feature importance figures in 'SAPFLUXNET transpiration ML project visualizations for paper.ipynb',
which is an archive of the original figures:
    cell 1 (feat_import.png)                     ->  feature_importance_clusters.png
    cell 2, panel b (importance violins)         ->  feature_importance_summary.png
    the two above stacked as a, b and c          ->  feature_importance_clusters_summary.png
    cell 4 (correlations_mat_r2.png)             ->  feature_importance_mat_r2.png
    cell 5 (correlations_swc_import.png)         ->  feature_importance_swc.png
    cell 7 (feat_climate_correlation.tiff)       ->  feature_importance_climate_correlation_clusters.png and
                                                     feature_importance_climate_correlation_sites.png and
                                                     feature_importance_climate_correlation_sites_pft_biome.png
Every figure but the site climate correlation has one value per cluster: each model is trained on a whole cluster,
so its importances describe the cluster. The site level importances also serve as replicates in
utilities/site_importance_tests.py.
Clusters are only compared within panels of similar size (number of locations), per the review.
Reads data/modeling_data/feature_importance.csv, written by utilities/feature_importance.py, and
cluster_statistics.csv, written by utilities/cluster_statistics.py; the site climate correlation also reads
site_feature_importance_r2_05.csv and site_statistics.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)
modeling_data = os.path.join(project, 'data', 'modeling_data')

my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
feature_labels = {'ta': 'T$_a$', 'vpd': 'VPD', 'ppfd_in': 'PPFD', 'swc_shallow': 'SWC'}

# One row per cluster and model, with the cluster's climate and data averages
importance = pd.read_csv(os.path.join(modeling_data, 'feature_importance.csv'))
cluster_statistics = pd.read_csv(os.path.join(modeling_data, 'cluster_statistics.csv'))
importance = importance.merge(cluster_statistics, on='Data set').rename(columns={'Data set': 'key'})

RF = importance[importance['Model'] == 'RF']
NN = importance[importance['Model'] == 'ANN']
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

# Panels of similarly sized clusters, by number of locations, as in the performance figures
size_panels = [
    ('Large (14-21 locations)', lambda n: n >= 14),
    ('Medium (5-9 locations)', lambda n: (n >= 5) & (n < 14)),
    ('Small (1-2 locations)', lambda n: n < 5),
]
panel_widths = [4, 2, 2]

color1 = '#0072B2'  # RF, Okabe-Ito blue (colour blind safe)
color2 = '#D55E00'  # NN, Okabe-Ito vermillion

# Feature colors and hatches of the stacked bars, as in the notebook
feature_colors = {'ta': '#fb8072', 'vpd': '#8dd3c7', 'ppfd_in': '#ffffb3', 'swc_shallow': '#80b1d3'}
feature_hatches = {'ta': '\\', 'vpd': '--', 'ppfd_in': '', 'swc_shallow': '.'}


def cluster_label(row):  # 'Temperate\nforest\n(21 loc., 25 sites)'
    return f"{cluster_names[row['key']]}\n({row['n locations_rf']} loc., {row['n sites_rf']} sites)"


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------------------------- stacked importances
barWidth = 0.5


def panel_letter(ax1, letter):
    ax1.text(-0.25, 1.05, letter, transform=ax1.transAxes, fontweight='bold', fontsize='large')


def plot_stacked(ax):  # ax is a 2 x 3 array, NN on top and RF below
    for row, (suffix, ylabel) in enumerate((('_nn', 'NN feat. import. (-)'), ('_rf', 'RF feat. import. (-)'))):
        for ax1, (title, in_panel) in zip(ax[row], size_panels):
            panel = RF_trim[in_panel(RF_trim['n locations_rf'])]
            r1 = np.arange(len(panel))
            bottom = np.zeros(len(panel))
            for feature in my_features:
                ax1.bar(r1, panel[feature + suffix], bottom=bottom, width=barWidth, color=feature_colors[feature],
                        hatch=feature_hatches[feature], edgecolor='k', label=feature_labels[feature])
                bottom += panel[feature + suffix].values
            ax1.set_xticks(r1)
            ax1.set_xticklabels([cluster_label(r) for _, r in panel.iterrows()] if row == 1 else [], fontsize=7.5)
            if row == 0:
                ax1.set_title(title, fontsize=10)
            tidy(ax1)
        ax[row, 0].set_ylabel(ylabel)

    ax[0, 0].set_ylim(0, 1)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[0, -1].legend(handles[::-1], labels[::-1], frameon=False, loc='upper left', bbox_to_anchor=(1, 1))


fig, ax = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True, gridspec_kw={'width_ratios': panel_widths})
plot_stacked(ax)
for ax1, letter in zip(ax[:, 0], 'ab'):
    panel_letter(ax1, letter)
fig.tight_layout()
fig.savefig(os.path.join(folder, 'feature_importance_clusters.png'), dpi=300)

# ---------------------------------------------------------------------------------------------- summary across clusters
# Every cluster is one point, shaped by its size panel, so clusters of very different size are shown side by side
# instead of pooled into one distribution; lines join the RF and NN values of the same cluster
size_markers = ['o', 's', '^']


def plot_summary(ax1, ylabel_size='large', legend_inside=False):
    rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible
    for i, feature in enumerate(my_features):
        x_rf, x_nn = 2 * i - 0.35, 2 * i + 0.35
        for (title, in_panel), marker in zip(size_panels, size_markers):
            panel = RF_trim[in_panel(RF_trim['n locations_rf'])]
            jitter = rng.uniform(-0.08, 0.08, len(panel))
            for xr, xn, yr, yn in zip(x_rf + jitter, x_nn + jitter, panel[feature + '_rf'], panel[feature + '_nn']):
                ax1.plot([xr, xn], [yr, yn], color='grey', linewidth=0.5, zorder=1)
            ax1.scatter(x_rf + jitter, panel[feature + '_rf'], marker=marker, color=color1, edgecolor='k', zorder=3)
            ax1.scatter(x_nn + jitter, panel[feature + '_nn'], marker=marker, color=color2, edgecolor='k', zorder=3)
        for x, column in ((x_rf, feature + '_rf'), (x_nn, feature + '_nn')):  # mean across clusters, as the violins showed
            ax1.plot([x - 0.2, x + 0.2], [RF_trim[column].mean()] * 2, color='k', linewidth=2, zorder=2)

    ax1.set_xticks([2 * i for i in range(len(my_features))])
    ax1.set_xticklabels([feature_labels[f] for f in my_features], fontsize='large')
    ax1.set_ylim(0, 0.7)
    ax1.set_ylabel('Feature importance (-)', fontsize=ylabel_size)
    tidy(ax1)
    handles = ([Patch(color=color1), Patch(color=color2)] +
               [Line2D([], [], marker=m, ls='', color='grey', markeredgecolor='k') for m in size_markers] +
               [Line2D([], [], color='k', linewidth=2)])
    labels = ['RF', 'NN'] + [title for title, _ in size_panels] + ['Mean across clusters']
    if legend_inside:  # two columns: models and the mean, then the size panels
        order = [0, 1, 5, 2, 3, 4]
        handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    ax1.legend(handles, labels, frameon=False, fontsize=8, ncol=2 if legend_inside else 1,
               **(dict(loc='upper right') if legend_inside else dict(loc='upper left', bbox_to_anchor=(1, 1))))


fig, ax1 = plt.subplots(figsize=(7.5, 4))
plot_summary(ax1)
fig.tight_layout()
fig.savefig(os.path.join(folder, 'feature_importance_summary.png'), dpi=300)

# ---------------------------------------------------------------------------------------------- stacked and summary combined
# (a) NN and (b) RF stacked importances above (c) the summary, on one grid so (c) spans the same width as the size
# panels above it and the y-labels line up; the legend of (c) moves into its empty top, keeping the right margin
# to the legend of (a)
fig = plt.figure(figsize=(11, 9.5))
grid = fig.add_gridspec(3, 3, width_ratios=panel_widths, height_ratios=[1, 1, 1.15])
ax = np.empty((2, 3), dtype=object)
for row in range(2):
    for col in range(3):
        ax[row, col] = fig.add_subplot(grid[row, col], sharey=ax[0, 0] if (row, col) != (0, 0) else None)
        if col:
            ax[row, col].tick_params(labelleft=False)
ax_summary = fig.add_subplot(grid[2, :])
plot_stacked(ax)
plot_summary(ax_summary, ylabel_size=None, legend_inside=True)
for ax1, letter in zip(ax[:, 0], 'ab'):
    panel_letter(ax1, letter)
fig.tight_layout()
fig.align_ylabels([ax[0, 0], ax[1, 0], ax_summary])
# (c) is placed after the layout, at the same distance left of its axes as a and b; (c) is wider, so its axes
# fraction is rescaled
scale = ax[0, 0].get_position().width / ax_summary.get_position().width
ax_summary.text(-0.25 * scale, 1.05, 'c', transform=ax_summary.transAxes, fontweight='bold', fontsize='large')
fig.savefig(os.path.join(folder, 'feature_importance_clusters_summary.png'), dpi=300)


# ---------------------------------------------------------------------------------------------- importance vs climate
def f(x, a, b):
    return a * x + b


def conf_int(x_data, y_data, px):  # as in the notebook
    popt, pcov = curve_fit(f, x_data, y_data)
    # calculate parameter confidence interval
    a, b = unc.correlated_values(popt, pcov)
    # calculate regression confidence interval
    py = a * px + b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    return py, nom, std


def importance_panel(ax1, rf, nn, x, y, point_size, alpha):
    # PFT clusters as squares and biome clusters as triangles, one fitted line and 95% confidence band per model
    px = np.linspace(min(rf[x].min(), nn[x].min()), max(rf[x].max(), nn[x].max()), 100)
    notes = []
    for label, data, color in (('RF', rf, color1), ('NN', nn, color2)):
        pft = data['key'].str.startswith('pft_')
        ax1.scatter(data.loc[pft, x], data.loc[pft, y], marker='s', s=point_size, color=color, alpha=alpha)
        ax1.scatter(data.loc[~pft, x], data.loc[~pft, y], marker='^', s=point_size, color=color, alpha=alpha)
        py, nom, std = conf_int(data[x], data[y], px)
        ax1.plot(px, nom, c=color)
        ax1.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, color=color, alpha=0.3)  # 95% confidence interval
        r, p = stats.pearsonr(data[x], data[y])
        notes.append(f'{label}: r = {r:.2f}, p = {p:.2g}, n = {len(data)}')
    ax1.text(0.02, 0.98, '\n'.join(notes), transform=ax1.transAxes, va='top', fontsize=6.5)
    ax1.set_ylim(bottom=0)
    tidy(ax1)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)


def importance_figure(panels, rf, nn, point_size, alpha, name):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5.5))
    for ax1, (x, y, xlabel, ylabel), letter in zip(ax.flat, panels, 'abcd'):
        importance_panel(ax1, rf, nn, x, y, point_size, alpha)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.text(-0.2, 1.02, letter, transform=ax1.transAxes, fontweight='bold', fontsize='large')
    ax[0, 1].legend([Line2D([], [], marker='s', ls='', color=color1), Line2D([], [], marker='^', ls='', color=color1),
                     Line2D([], [], marker='s', ls='', color=color2), Line2D([], [], marker='^', ls='', color=color2)],
                    ['RF, PFT', 'RF, biome', 'NN, PFT', 'NN, biome'], bbox_to_anchor=(1, 0.5), loc='lower left',
                    fontsize=8, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=300)


mat_r2_panels = [  # cell 4
    ('MAT', 'ta', 'MAT ($^\\circ$C)', 'Temp. feat. import. (-)'),
    ('R2 test', 'ppfd_in', 'Test set R$^2$', 'PPFD feat. import. (-)'),
    ('MAT', 'ppfd_in', 'MAT ($^\\circ$C)', 'PPFD feat. import. (-)'),
    ('R2 test', 'swc_shallow', 'Test set R$^2$', 'SWC feat. import. (-)'),
]
swc_panels = [  # cell 5
    ('MAP', 'swc_shallow', 'MAP (mm)', 'SWC feat. import. (-)'),
    ('swc_shallow_avg', 'swc_shallow', 'Mean SWC (-)', 'SWC feat. import. (-)'),
    ('Sap flux average', 'swc_shallow', 'Mean sap flux (cm h$^{-1}$)', 'SWC feat. import. (-)'),
    ('vpd_avg', 'swc_shallow', 'Mean VPD (kPa)', 'SWC feat. import. (-)'),
]

importance_figure(mat_r2_panels, RF, NN, 40, 1.0, 'feature_importance_mat_r2.png')
importance_figure(swc_panels, RF, NN, 40, 1.0, 'feature_importance_swc.png')

# ---------------------------------------------------------------------------------------------- climate correlation
climate = ['R2 test', 'MAP', 'MAT', 'ta_avg', 'vpd_avg', 'ppfd_in_avg', 'swc_shallow_avg', 'Sap flux average']
climate_labels = ['Test set R$^2$', 'MAP', 'MAT', 'Avg. T$_a$', 'Avg. VPD', 'Avg. PPFD', 'Avg. SWC', 'Avg. sap flux']


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def climate_correlation_figure(panels, name):  # panels is a grid (list of rows) of (title, data)
    rows, cols = len(panels), len(panels[0])
    fig, ax = plt.subplots(rows, cols, figsize=(8, 4.5 * rows), sharey=True, squeeze=False,
                           gridspec_kw=dict(hspace=0.35))
    for ax1, (title, data) in zip(ax.flat, [panel for row in panels for panel in row]):
        # Spearman correlation between each feature's importance and each cluster or site property
        rho = np.array([[stats.spearmanr(data[c], data[f])[0] for f in my_features] for c in climate])
        p = np.array([[stats.spearmanr(data[c], data[f])[1] for f in my_features] for c in climate])
        image = ax1.imshow(rho, cmap='bwr', vmin=-1, vmax=1)
        for (i, j), value in np.ndenumerate(rho):  # only the significant correlations are labelled
            if p[i, j] < 0.05:
                ax1.text(j, i, f'{value:.2f}{stars(p[i, j])}', ha='center', va='center', fontsize=6.5)
        ax1.set_xticks(range(len(my_features)))
        ax1.set_xticklabels([feature_labels[f] for f in my_features], rotation=45, ha='right')
        ax1.set_title(f'{title} (n = {len(data)})', fontsize=10)
    for ax1 in ax[:, 0]:
        ax1.set_yticks(range(len(climate)))
        ax1.set_yticklabels(climate_labels)
    fig.colorbar(image, ax=ax, shrink=0.8 if rows == 1 else 0.5,
                 label='Spearman correlation with feature importance\n(* p < 0.05, ** p < 0.01, *** p < 0.001)')
    fig.savefig(os.path.join(folder, name), dpi=300, bbox_inches='tight')


climate_correlation_figure([[('RF', RF), ('NN', NN)]], 'feature_importance_climate_correlation_clusters.png')

# Site level: one row per site and cluster model where that model has R2 test > 0.5 (above floor = True in
# site_feature_importance_r2_05.csv, written by utilities/site_feature_importance.py, as in the other site level
# figures and tests), joined to the site averages in site_statistics.csv, written by utilities/cluster_statistics.py
site_importance = pd.read_csv(os.path.join(modeling_data, 'site_feature_importance_r2_05.csv'))
site_statistics = pd.read_csv(os.path.join(modeling_data, 'site_statistics.csv'))
site_importance = site_importance[site_importance['above floor']].merge(site_statistics, on=['Site', 'Data set'])
site_RF = site_importance[site_importance['Model'] == 'RF']
site_NN = site_importance[site_importance['Model'] == 'ANN']
climate_correlation_figure([[('RF', site_RF), ('NN', site_NN)]], 'feature_importance_climate_correlation_sites.png')

# The same with the PFT and biome cluster models apart, so each site appears once per panel
climate_correlation_figure(
    [[(f'{model}, {grouping}', data[data['Data set'].str.startswith(prefix)])
      for model, data in (('RF', site_RF), ('NN', site_NN))]
     for grouping, prefix in (('PFT', 'pft_'), ('biome', 'biome_'))],
    'feature_importance_climate_correlation_sites_pft_biome.png')
