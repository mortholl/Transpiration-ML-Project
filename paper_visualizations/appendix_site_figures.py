# Regenerates the appendix figures fig:site_stats (input features) and fig:site_target (sap flux)
# from the same data the models train on: data_import() in utilities/data_sanitizer.py, i.e. the
# resampled sapwood-level files, the 60 sites that pass the load-time criteria, sensor readings above
# 200 cm/h masked. cap_quantile=1.0 turns the location cap off, so every usable row is shown.
# Usage: python appendix_site_figures.py <repo path> <output folder>
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo, out_dir = sys.argv[1], os.path.abspath(sys.argv[2])
os.chdir(repo)
sys.path.insert(0, repo)
try:
    import sklearn  # noqa: F401, only cluster_creator needs it, data_import does not
except ImportError:  # lets the figures build where scikit-learn is not installed
    import types
    for name in ['sklearn', 'sklearn.cluster', 'sklearn.preprocessing']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['sklearn.cluster'].KMeans = None
    sys.modules['sklearn.preprocessing'].StandardScaler = None
from utilities.data_sanitizer import data_import

features = ['ta', 'vpd', 'swc_shallow', 'ppfd_in']
target = 'Average Sap Flux'
x, y, info = data_import(features, [], verbose=False, cap_quantile=1.0, return_info=True)
df = pd.DataFrame(x, columns=features)
df[target] = y
df['Site'] = info['Site'].values

clusters = pd.read_csv('data/modeling_data/cluster_info.csv', index_col='Site')
df['Biome'] = df['Site'].map(clusters['Biome'])
df['FT'] = df['Site'].map(clusters['Functional Type'].astype(str)).map(
    {'0': 'Deciduous', '1': 'Evergreen', '2': 'Mixed'})

groups = [('FT', 'Deciduous'), ('FT', 'Evergreen'), ('FT', 'Mixed'),
          ('Biome', 'Woodland/Shrubland'), ('Biome', 'Temperate forest'), ('Biome', 'Tropical rain forest'),
          ('Biome', 'Tropical forest savanna'), ('Biome', 'Temperate grassland desert')]
subsets = [df[df[col] == name] for col, name in groups]
labels = [f'{name.replace("Temperate grassland desert", "Temperate grassland" + chr(10) + "desert")}\n'
          f'({s["Site"].nunique()} sites)' for (col, name), s in zip(groups, subsets)]
positions = np.arange(len(groups))
wp = 0.75

summary = pd.DataFrame({'group': [n for _, n in groups],
                        'sites': [s['Site'].nunique() for s in subsets],
                        'rows': [len(s) for s in subsets]})
print(f'{df["Site"].nunique()} sites, {len(df)} rows')
print(summary.to_string(index=False))


def violins(ax, column):
    ax.violinplot([s[column].values for s in subsets], positions, widths=wp, showmeans=True,
                  showextrema=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.axvline(2.5, color='grey', linewidth=0.75)  # functional types left, biomes right
    ax.set_xticks(positions)
    ax.set_xticklabels([])


# fig:site_stats
fig, ax = plt.subplots(4, 1, figsize=(8, 12))
ylabels = ['T$_a$ (C)', 'VPD (kPa)', 'SWC (-)', r'PPFD ($\mu$mol m$^{-2}$ s$^{-1}$)']
for a, column, ylabel, letter in zip(ax, features, ylabels, 'abcd'):
    violins(a, column)
    a.set_ylabel(ylabel)
    a.text(-0.08, 1.02, letter, transform=a.transAxes, fontweight='bold', fontsize='large')
ax[3].set_xticklabels(labels, rotation=45, ha='right')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'site_stats_AppendixC_column.png'), dpi=300)

# fig:site_target
fig, a = plt.subplots(1, 1, figsize=(7.5, 4.5))
violins(a, target)
a.set_ylabel('Sap flux density (cm h$^{-1}$)')
a.set_xticklabels(labels, rotation=45, ha='right')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'site_sapf_AppendixC_full_cropped.png'), dpi=300)
