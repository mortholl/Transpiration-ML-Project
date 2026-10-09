# -*- coding: utf-8 -*-
"""
Mean relative feature importance by cluster with the significance tests, RF vs NN.

One row per model input, PFT clusters on the left and biome clusters on the right. Each point is a cluster's mean
site level relative importance with its 95% confidence interval, over the sites where that model has test R2 above
the floor, on one y scale for every feature. A model whose clusters differ significantly (Welch's ANOVA, p < 0.05) is drawn in full with its means
joined; otherwise it is faded. An asterisk marks a cluster where RF and NN differ significantly (paired t-test).
Only clusters with at least 5 locations are shown, the ones that were tested.
Reads data/modeling_data/site_feature_importance_r2_05.csv and the test results written by
utilities/site_importance_tests.py (importance_tests_clusters_r2_05.csv, importance_tests_rf_vs_ann_r2_05.csv).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)
modeling_data = os.path.join(project, 'data', 'modeling_data')

my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
feature_labels = {'ta': 'T$_a$', 'vpd': 'VPD', 'ppfd_in': 'PPFD', 'swc_shallow': 'SWC'}
cluster_names = {  # PFT then biome, largest first
    'pft_deciduous': 'Deciduous', 'pft_evergreen': 'Evergreen', 'pft_mixed': 'Mixed',
    'biome_WoodlandShrubland': 'Woodland/Shrubland', 'biome_Temperate forest': 'Temperate forest',
    'biome_Tropical rain forest': 'Tropical rainforest', 'biome_Tropical forest savanna': 'Tropical forest savanna',
    'biome_Temperate grassland desert': 'Grassland desert',
}
groupings = [('PFT', 'pft_'), ('Biome', 'biome_')]

color1 = '#0072B2'  # RF, Okabe-Ito blue (colour blind safe)
color2 = '#D55E00'  # NN, Okabe-Ito vermillion
marker1, marker2 = 'o', 's'  # RF, NN

sites = pd.read_csv(os.path.join(modeling_data, 'site_feature_importance_r2_05.csv'))
sites = sites[sites['above floor']].dropna(subset=my_features)  # per model R2 floor, as in the Welch's ANOVAs
anova = pd.read_csv(os.path.join(modeling_data, 'importance_tests_clusters_r2_05.csv'))
rf_vs_nn = pd.read_csv(os.path.join(modeling_data, 'importance_tests_rf_vs_ann_r2_05.csv'))
tested = [key for key in cluster_names if key in set(rf_vs_nn['Data set'])]


def anova_p(prefix, model, feature):
    row = anova[(anova['grouping'] == prefix.rstrip('_')) & (anova['Model'] == model) & (anova['feature'] == feature)]
    return row['p'].iloc[0]


def ci_half(values):  # 95% confidence interval half width of a mean
    return stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))


def p_text(p):
    return ('p < 0.001' if p < 0.001 else f'p = {p:.3f}') + (' *' if p < 0.05 else '')


fig, ax = plt.subplots(len(my_features), 2, figsize=(9, 11), sharey='row')
row_low, row_high = [np.inf] * len(my_features), [0] * len(my_features)
for i, feature in enumerate(my_features):
    for j, (title, prefix) in enumerate(groupings):
        ax1 = ax[i, j]
        keys = [k for k in tested if k.startswith(prefix)]
        p_model = {model: anova_p(prefix, model, feature) for model in ('RF', 'ANN')}
        cluster_top = {}
        for offset, model, color, marker in ((-0.12, 'RF', color1, marker1), (0.12, 'ANN', color2, marker2)):
            significant = p_model[model] < 0.05
            means, xs = [], []
            for x, key in enumerate(keys):
                v = sites.loc[(sites['Data set'] == key) & (sites['Model'] == model), feature]
                ax1.errorbar(x + offset, v.mean(), yerr=ci_half(v), fmt=marker, color=color, capsize=3,
                             mfc=color if significant else 'white', alpha=1 if significant else 0.35,
                             markersize=7 if significant else 5)
                means.append(v.mean())
                xs.append(x + offset)
                cluster_top[x] = max(cluster_top.get(x, 0), v.mean() + ci_half(v))
                row_low[i] = min(row_low[i], v.mean() - ci_half(v))
            if significant:
                ax1.plot(xs, means, color=color, linewidth=1.2, alpha=0.8)
        for x, key in enumerate(keys):  # RF vs NN within the cluster
            if rf_vs_nn[(rf_vs_nn['Data set'] == key) & (rf_vs_nn['feature'] == feature)]['p'].iloc[0] < 0.05:
                ax1.annotate('*', (x, cluster_top[x]), ha='center', va='bottom', fontsize=14, fontweight='bold')
        for k, (model, color, label) in enumerate((('RF', color1, 'RF'), ('ANN', color2, 'NN'))):
            p = p_model[model]
            ax1.text(0.02, 0.97 - 0.09 * k, f'{label}: {p_text(p)}', transform=ax1.transAxes, va='top',
                     fontsize=8, color=color if p < 0.05 else 'grey', fontweight='bold' if p < 0.05 else 'normal')
        row_high[i] = max(row_high[i], max(cluster_top.values()))
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels([cluster_names[k] for k in keys], fontsize=8)
        ax1.set_xlim(-0.5, len(keys) - 0.5)
        if i == 0:
            ax1.set_title(f'{title} clusters', fontsize=11)
        if j == 0:
            ax1.set_ylabel(f'{feature_labels[feature]} relative importance')
low, high = max(min(row_low), 0), max(row_high)  # one scale for every feature, so their magnitudes compare
for i in range(len(my_features)):  # room below the lowest interval and above the asterisks and p values
    ax[i, 0].set_ylim(low - 0.05 * (high - low), high + 0.2 * (high - low))
fig.legend([Line2D([], [], marker=marker1, color=color1), Line2D([], [], marker=marker2, color=color2),
            Line2D([], [], marker='o', ls='', mfc='white', color='grey', alpha=0.5),
            Line2D([], [], marker='$*$', ls='', color='k', markersize=9)],
           ["RF: significant difference among clusters (Welch's ANOVA)",
            "NN: significant difference among clusters (Welch's ANOVA)",
            "No significant difference among clusters (Welch's ANOVA)",
            'Significant RF vs NN difference (paired t-test)'],
           loc='lower center', ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(folder, 'feature_importance_tests.png'), dpi=300)
