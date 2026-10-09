import os
import numpy as np
import pandas as pd
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key, load_test_mask
from utilities.feature_importance import my_features, load_model, r2_scorer, relative_importance
from utilities.site_feature_importance import r2_floor
from utilities.night_dormancy_performance import ppfd_floor, sap_flux_floor, min_site_rows

# Relative permutation feature importance for every site, on daytime active flow only
# Same method and site floor as site_feature_importance.py, applied to the test rows left once night (PPFD below
# ppfd_floor) and low flow (sap flux below sap_flux_floor) rows are both removed, as in night_dormancy_performance.py
# The saved models are not retrained, so these are the drivers the same models rely on during active daytime flow
# A model's R2 and its floor are measured on the filtered rows; the output has the same columns as
# site_feature_importance.py so the two files can be paired by Data set, Site and Model

output_file = 'data/modeling_data/night_dormancy_splits/site_feature_importance_both_floors.csv'


if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
            is_active = (X[:, my_features.index('ppfd_in')] >= ppfd_floor) & (Y >= sap_flux_floor)
            keep = is_test & is_active
            X_test, Y_test, test_info = X[keep], Y[keep], info[keep].reset_index(drop=True)
            sites = {site: np.asarray(index) for site, index in test_info.groupby('Site').groups.items()}
            cluster_rows = []
            for suffix, label in (('_rf', 'RF'), ('_ann', 'ANN')):  # one model in memory at a time, the
                model, scaler = load_model(key + suffix)            # Temperate forest RF nearly fills 32 GB
                for site, site_index in sites.items():
                    X_site, Y_site = scaler.transform(X_test[site_index]), Y_test[site_index]
                    site_r2 = r2_scorer(model, X_site, Y_site) if len(site_index) >= min_site_rows else np.nan
                    shares = drops = [np.nan] * len(my_features)
                    if site_r2 > r2_floor:
                        _, shares, drops = relative_importance(model, X_site, Y_site)
                    cluster_rows.append({'Data set': key, 'Site': site,
                                         'Location': test_info.loc[site_index[0], 'Location'], 'Model': label,
                                         'n test points': len(site_index), 'R2 test': site_r2,
                                         'above floor': site_r2 > r2_floor, **dict(zip(my_features, shares)),
                                         **{f'{feature} R2 drop': drop for feature, drop in zip(my_features, drops)}})
                del model
                kept = sum(row['above floor'] for row in cluster_rows if row['Model'] == label)
                print(f'{key}: {kept} of {len(sites)} sites have {label} filtered test R2 above {r2_floor}', flush=True)
            passing = pd.DataFrame(cluster_rows).groupby('Site')['above floor'].all()  # both models above the floor
            for row in cluster_rows:
                row['replicate'] = bool(passing[row['Site']])
            rows += cluster_rows
            pd.DataFrame(rows).to_csv(output_file, index=False)  # rewritten after every cluster so an
    print(f'Written to {output_file}')                             # interrupted run keeps what it finished
