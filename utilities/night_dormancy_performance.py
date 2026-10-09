import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key, load_test_mask
from utilities.feature_importance import my_features, load_model

# Test R2 of the saved RF and ANN models once night and dormant rows are removed, to see whether
# performance depends on those periods. Each model predicts its cluster's held out test rows once, then is scored on
#   all: every test row
#   day: rows with PPFD below ppfd_floor removed (night)
#   active flow: rows with sap flux below sap_flux_floor removed (night or dormancy)
# The floors are absolute, so a cluster with little dormancy, such as tropical rain forest, loses few daytime rows
# Run from the project folder after rf_optimization.py, ann_optimization.py and test_set_builder.py

output_directory = 'data/modeling_data/night_dormancy_splits'
output_file = f'{output_directory}/night_dormancy_performance.csv'  # pooled over each cluster's test rows
site_output_file = f'{output_directory}/site_night_dormancy_performance.csv'  # the same scores site by site
ppfd_floor = 10  # umol m-2 s-1
sap_flux_floor = 0.5  # cm3 cm-2 h-1, just above the median night time sap flux of the test rows (0.38)
r2_floor = 0  # a site is a replicate where both models are above it
min_site_rows = 50  # a site with fewer rows left than this gets no score
target = 'Average Sap Flux'


def label_conditions(X, Y, info):  # Returns info with each row's PPFD, sap flux and which subsets keep it
    rows = info.copy()
    rows['ppfd_in'] = X[:, my_features.index('ppfd_in')]
    rows[target] = Y
    rows['all'] = True
    rows['day'] = rows['ppfd_in'] >= ppfd_floor
    rows['active flow'] = rows[target] >= sap_flux_floor
    return rows


def condition_scores(Y, Y_pred, conditions):  # Returns one row of scores for each subset of kept rows
    scores = []
    for condition in ('all', 'day', 'active flow'):
        keep = conditions[condition].values
        n = int(keep.sum())
        scores.append({'Condition': condition, 'n test points': n, 'share removed': 1 - n / len(Y),
                       'R2 test': r2_score(Y[keep], Y_pred[keep]) if n >= min_site_rows else np.nan})
    return scores


if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    os.makedirs(output_directory, exist_ok=True)
    # An interrupted run picks up at the first unfinished cluster, delete the output files to score every model again
    rows = pd.read_csv(output_file).to_dict('records') if os.path.exists(output_file) else []
    site_rows = pd.read_csv(site_output_file).to_dict('records') if os.path.exists(site_output_file) else []
    finished = {row['Data set'] for row in rows}
    site_rows = [row for row in site_rows if row['Data set'] in finished]  # drops a half written cluster
    for identifier, cluster_group in groups:  # the same clusters the optimization scripts loop over
        for data_cluster in cluster_group:
            key = cluster_key(identifier, data_cluster)
            if key in finished:
                continue
            X, Y, info = data_import(my_features, cluster_group[data_cluster], return_info=True)
            is_test = load_test_mask(key, info)  # drawn once by test_set_builder.py
            conditions = label_conditions(X[is_test], Y[is_test], info[is_test].reset_index(drop=True))
            conditions.to_csv(f'{output_directory}/{key}_test_conditions.csv', index=False)  # the filtered test sets
            X_test, Y_test = X[is_test], Y[is_test]
            predictions = {}
            for suffix, label in (('_rf', 'RF'), ('_ann', 'ANN')):  # one model in memory at a time, the
                model, scaler = load_model(key + suffix)            # Temperate forest RF nearly fills 32 GB
                if label == 'RF':
                    predictions[label] = model.predict(scaler.transform(X_test))
                else:
                    predictions[label] = np.ravel(model.predict(scaler.transform(X_test), batch_size=4096, verbose=0))
                del model
                for score in condition_scores(Y_test, predictions[label], conditions):
                    rows.append({'Data set': key, 'Model': label, **score})
                    print(f"{key + suffix} {score['Condition']}: R2 {score['R2 test']:.3f}, "
                          f"{score['share removed']:.0%} of test rows removed", flush=True)
            for site, site_index in conditions.groupby('Site').groups.items():
                site_index = np.asarray(site_index)
                site_r2 = {label: r2_score(Y_test[site_index], Y_pred[site_index])
                           for label, Y_pred in predictions.items()}
                is_replicate = all(r2 > r2_floor for r2 in site_r2.values())
                for label, Y_pred in predictions.items():
                    for score in condition_scores(Y_test[site_index], Y_pred[site_index],
                                                  conditions.loc[site_index]):
                        site_rows.append({'Data set': key, 'Site': site,
                                          'Location': conditions.loc[site_index[0], 'Location'],
                                          'Model': label, 'replicate': is_replicate, **score})
            pd.DataFrame(site_rows).to_csv(site_output_file, index=False)  # rewritten after every cluster so an
            pd.DataFrame(rows).to_csv(output_file, index=False)            # interrupted run keeps what it finished
    print(f'Written to {output_file} and {site_output_file}')
