import itertools
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

# Significance tests on the site level relative feature importances written by site_feature_importance.py,
# with sites as replicates, one test per feature:
#   RF vs ANN within a cluster: paired t-test on the sites where both models have test R2 above the floor
#   between clusters: Welch's ANOVA across the PFT clusters and across the biome clusters, per model, on the sites
#       where that model has test R2 above the floor
# Welch's ANOVA does not assume the clusters have the same number of sites or the same spread
# Run from the project folder after site_feature_importance.py

input_file = 'data/modeling_data/site_feature_importance_r2_05.csv'
methods_file = 'data/modeling_data/importance_tests_rf_vs_ann_r2_05.csv'
clusters_file = 'data/modeling_data/importance_tests_clusters_r2_05.csv'

# Must match the feature list in the optimization scripts
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']

min_locations = 5  # sites at one location are not independent, so a cluster needs this many locations among
alpha = 0.05       # the sites it keeps to be tested; the others are reported without tests

if __name__ == '__main__':
    all_sites = pd.read_csv(input_file)
    all_sites = all_sites[all_sites['above floor']]
    all_sites = all_sites.dropna(subset=my_features)  # no feature lowered R2 when shuffled, so there are no shares

    # ------------------------------------------------------------------------------------ RF vs ANN
    sites = all_sites[all_sites['replicate']]
    sites = sites[sites.groupby(['Data set', 'Site'])['Model'].transform('size') == 2]  # keep pairs whole
    n_locations = sites.groupby('Data set')['Location'].nunique()  # left after the R2 floor
    print(f'Clusters with fewer than {min_locations} locations, not tested RF vs ANN: '
          f'{n_locations[n_locations < min_locations].to_dict()}')
    sites = sites[sites['Data set'].map(n_locations) >= min_locations]
    rows = []
    for key, cluster in sites.groupby('Data set', sort=False):
        paired = cluster.pivot(index='Site', columns='Model', values=my_features)
        for feature in my_features:
            rf, ann = paired[(feature, 'RF')], paired[(feature, 'ANN')]
            t, p = stats.ttest_rel(rf, ann)
            rows.append({'Data set': key, 'feature': feature, 'n locations': n_locations[key], 'n sites': len(paired),
                         'mean RF': rf.mean(), 'mean ANN': ann.mean(), 'mean difference': (rf - ann).mean(),
                         't': t, 'p': p, 'significant': p < alpha})
    methods = pd.DataFrame(rows)
    methods.to_csv(methods_file, index=False)

    # ------------------------------------------------------------------------------------ between clusters
    rows = []
    for prefix, model in itertools.product(['pft_', 'biome_'], ['RF', 'ANN']):
        grouping = all_sites[all_sites['Data set'].str.startswith(prefix) & (all_sites['Model'] == model)]
        n_locations = grouping.groupby('Data set')['Location'].nunique()
        print(f"{prefix.rstrip('_')} {model} clusters with fewer than {min_locations} locations, not tested: "
              f'{n_locations[n_locations < min_locations].to_dict()}')
        grouping = grouping[grouping['Data set'].map(n_locations) >= min_locations]
        keys = list(grouping['Data set'].unique())
        for feature in my_features:
            groups = [grouping.loc[grouping['Data set'] == key, feature] for key in keys]
            result = anova_oneway(groups, use_var='unequal')  # Welch's ANOVA
            rows.append({'grouping': prefix.rstrip('_'), 'Model': model, 'feature': feature,
                         'clusters': '; '.join(f'{key} ({n_locations[key]} loc., {len(g)} sites, mean {g.mean():.3f})'
                                               for key, g in zip(keys, groups)),
                         'F': result.statistic, 'df between': result.df[0], 'df within': result.df[1],
                         'p': result.pvalue, 'significant': result.pvalue < alpha})
    clusters = pd.DataFrame(rows)
    clusters.to_csv(clusters_file, index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.precision', 3,
                           'display.max_colwidth', 200):
        print('\n', methods[methods['significant']].drop(columns=['t']), '\n')
        print(clusters[clusters['significant']])
