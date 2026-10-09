# -*- coding: utf-8 -*-
"""
Night and dormancy figure for the revised paper, RF vs NN, answering the review comment that drivers differ
between day and night and between the growing and non-growing season.

Shows each cluster's pooled test R2 on the full test set and once night rows (PPFD < 10 umol m-2 s-1) or low flow
rows (sap flux < 0.5 cm3 cm-2 h-1, night or dormancy) are removed, as points joined by a line so the length of the
line is the drop:
    night_dormancy_r2_drop.png
Clusters are only compared within panels of similar size (number of locations), as in the performance figures.
Reads data/modeling_data/night_dormancy_splits/night_dormancy_performance.csv, written by
utilities/night_dormancy_performance.py, and the model results for the cluster sizes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

folder = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(folder)

performance = pd.read_csv(os.path.join(project, 'data', 'modeling_data', 'night_dormancy_splits',
                                       'night_dormancy_performance.csv'))
RF = pd.read_csv(os.path.join(project, 'RandomForest', 'rf_results.csv'), skipinitialspace=True)
RF['key'] = RF['Data set'].str.replace('_rf$', '', regex=True)

# One row per cluster, the R2 of each model on the full and the filtered test sets, largest cluster first
r2 = performance.pivot_table(index='Data set', columns=['Model', 'Condition'], values='R2 test')
r2.columns = [f'{condition} {model}' for model, condition in r2.columns]
r2 = r2.join(RF.set_index('key')[['n locations', 'n sites']]).reset_index(names='key')
r2 = r2.sort_values('n locations', ascending=False)

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

color1 = '#0072B2'  # RF, Okabe-Ito blue (colour blind safe)
color2 = '#D55E00'  # NN, Okabe-Ito vermillion

# One marker per test set, the same for both models
conditions = [('all', 'o', 'Full test set'), ('day', 'v', 'Night removed'), ('active flow', 's', 'Low flow removed')]


def cluster_label(row):  # 'Temperate\nforest\n(21 loc., 25 sites)'
    return f"{cluster_names[row['key']]}\n({row['n locations']} loc., {row['n sites']} sites)"


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------------------------- R2 with and without
offset = 0.18  # RF left of the cluster's tick, NN right of it
spread = [0, -0.07, 0.07]  # full test set in the middle, night removed left, low flow removed right

fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, gridspec_kw={'width_ratios': panel_widths})

for ax1, (title, in_panel) in zip(ax, size_panels):
    panel = r2[in_panel(r2['n locations'])]
    r1 = np.arange(len(panel))
    for model, color, x in (('RF', color1, r1 - offset), ('ANN', color2, r1 + offset)):
        full = panel[f'all {model}'].values
        for (condition, marker, _), shift in zip(conditions, spread):  # the two filtered sets sit either side of
            values = panel[f'{condition} {model}'].values                  # the full one, so close values don't hide
            ax1.plot([x, x + shift], [full, values], color=color, linewidth=1.5, zorder=1)  # each other
            ax1.scatter(x + shift, values, marker=marker, s=36, color=color, edgecolor='k', linewidth=0.6, zorder=2)
    ax1.set_xticks(r1)
    ax1.set_xticklabels([cluster_label(row) for _, row in panel.iterrows()], fontsize=7.5)
    ax1.set_xlim(-0.5, len(panel) - 0.5)
    ax1.set_title(title, fontsize=10)
    tidy(ax1)

ax[0].set_ylim(0.4, 1)
ax[0].set_ylabel('Test set R$^2$')
legend = [Line2D([], [], marker=marker, linestyle='', color='w', markeredgecolor='k', markersize=7, label=label)
          for _, marker, label in conditions]
legend += [Line2D([], [], color=color1, linewidth=6, label='RF'), Line2D([], [], color=color2, linewidth=6,
                                                                            label='NN')]
ax[-1].legend(handles=legend, frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

fig.tight_layout()
fig.savefig(os.path.join(folder, 'night_dormancy_r2_drop.png'), dpi=300)
