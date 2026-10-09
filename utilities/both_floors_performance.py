import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key, load_test_mask
from utilities.feature_importance import my_features, load_model
from utilities.night_dormancy_performance import ppfd_floor, sap_flux_floor

# Pooled test R2 of the saved RF and ANN models once night and low flow rows are removed together, the companion to
# site_feature_importance_floors.py. Each model is scored on
#   all: every test row
#   both floors: rows with PPFD below ppfd_floor or sap flux below sap_flux_floor removed (night or dormancy)
# Written to its own file, night_dormancy_performance.csv keeps the two floors applied one at a time
# Run from the project folder after rf_optimization.py, ann_optimization.py and test_set_builder.py

output_file = 'data/modeling_data/night_dormancy_splits/both_floors_performance.csv'


if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # An interrupted run picks up at the first unfinished cluster, delete the output file to score every model again
    rows = pd.read_csv(output_file).to_dict('records') if os.path.exists(output_file) else []
    finished = {row['Data set'] for row in rows}
    for identifier, cluster_group in groups:  # the same clusters the optimization scripts loop over
        for data_cluster in cluster_group:
            key = cluster_key(identifier, data_cluster)
            if key in finished:
                continue
            X, Y, info = data_import(my_features, cluster_group[data_cluster], return_info=True)
            is_test = load_test_mask(key, info)  # drawn once by test_set_builder.py
            X_test, Y_test = X[is_test], Y[is_test]
            subsets = {'all': np.full(len(Y_test), True),
                       'both floors': (X_test[:, my_features.index('ppfd_in')] >= ppfd_floor) &
                                      (Y_test >= sap_flux_floor)}
            for suffix, label in (('_rf', 'RF'), ('_ann', 'ANN')):  # one model in memory at a time, the
                model, scaler = load_model(key + suffix)            # Temperate forest RF nearly fills 32 GB
                if label == 'RF':
                    Y_pred = model.predict(scaler.transform(X_test))
                else:
                    Y_pred = np.ravel(model.predict(scaler.transform(X_test), batch_size=4096, verbose=0))
                del model
                for condition, keep in subsets.items():
                    n = int(keep.sum())
                    rows.append({'Data set': key, 'Model': label, 'Condition': condition, 'n test points': n,
                                 'share removed': 1 - n / len(Y_test), 'R2 test': r2_score(Y_test[keep], Y_pred[keep])})
                    print(f"{key + suffix} {condition}: R2 {rows[-1]['R2 test']:.3f}, "
                          f"{rows[-1]['share removed']:.0%} of test rows removed", flush=True)
            pd.DataFrame(rows).to_csv(output_file, index=False)  # rewritten after every cluster so an
    print(f'Written to {output_file}')                             # interrupted run keeps what it finished
