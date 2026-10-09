import os
import numpy as np
import pandas as pd
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key, load_test_mask
from utilities.feature_importance import my_features, load_model, r2_scorer, relative_importance

# Relative permutation feature importance for every site
# Same method as feature_importance.py, applied to one site's held out test rows at a time
# A model's importances are only measured at the sites where it has skill, test R2 above r2_floor ('above floor');
# the other sites are written with their R2 and empty importances, so the number dropped can be reported
# A site is a replicate for the paired RF vs ANN comparison where both models are above the floor ('replicate')

output_file = 'data/modeling_data/site_feature_importance_r2_05.csv'
r2_floor = 0.5


if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    # An interrupted run picks up at the first unfinished cluster, delete the output file to measure every site again
    rows = pd.read_csv(output_file).to_dict('records') if os.path.exists(output_file) else []
    finished = {row['Data set'] for row in rows}
    for identifier, cluster_group in groups:  # the same clusters the optimization scripts loop over
        for data_cluster in cluster_group:
            key = cluster_key(identifier, data_cluster)
            if key in finished:
                continue
            X, Y, info = data_import(my_features, cluster_group[data_cluster], return_info=True)
            is_test = load_test_mask(key, info)  # drawn once by test_set_builder.py
            X_test, Y_test, test_info = X[is_test], Y[is_test], info[is_test].reset_index(drop=True)
            models = {label: load_model(key + suffix) for suffix, label in (('_rf', 'RF'), ('_ann', 'ANN'))}
            for site, site_rows in test_info.groupby('Site').groups.items():
                site_rows = np.asarray(site_rows)
                X_site, Y_site = X_test[site_rows], Y_test[site_rows]
                site_r2 = {label: r2_scorer(model, scaler.transform(X_site), Y_site)
                           for label, (model, scaler) in models.items()}
                is_replicate = all(r2 > r2_floor for r2 in site_r2.values())
                for label, (model, scaler) in models.items():
                    shares = drops = [np.nan] * len(my_features)
                    if site_r2[label] > r2_floor:
                        _, shares, drops = relative_importance(model, scaler.transform(X_site), Y_site)
                    rows.append({'Data set': key, 'Site': site, 'Location': test_info.loc[site_rows[0], 'Location'],
                                 'Model': label, 'n test points': len(site_rows), 'R2 test': site_r2[label],
                                 'above floor': site_r2[label] > r2_floor, 'replicate': is_replicate,
                                 **dict(zip(my_features, shares)),
                                 **{f'{feature} R2 drop': drop for feature, drop in zip(my_features, drops)}})
            n_sites = test_info['Site'].nunique()
            for label in models:
                kept = sum(row['above floor'] for row in rows if row['Data set'] == key and row['Model'] == label)
                print(f'{key}: {kept} of {n_sites} sites have {label} test R2 above {r2_floor}', flush=True)
            pd.DataFrame(rows).to_csv(output_file, index=False)  # rewritten after every cluster so an
    print(f'Written to {output_file}')                             # interrupted run keeps what it finished
