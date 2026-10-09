import os
import keras
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from utilities.test_set_builder import cluster_key, load_test_mask

# Relative permutation feature importance for the saved RF and ANN models, measured the same way for both:
# the shared held out test rows, R2 as the score, 10 shuffles per feature, each feature's mean drop in R2
# (negative drops set to 0) divided by the sum over all features so the importances of one model add up to 1
# Run from the project folder after rf_optimization.py and ann_optimization.py

output_file = 'data/modeling_data/feature_importance.csv'
n_repeats = 10
seed = 51

# Must match the feature list in the optimization scripts
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']


def r2_scorer(model, X, Y):  # one score for both model types, the keras model returns an (n, 1) array
    if isinstance(model, keras.Model):
        Y_pred = model.predict(X, batch_size=4096, verbose=0)
    else:
        Y_pred = model.predict(X)
    return r2_score(Y, np.ravel(Y_pred))


def load_model(model_name):
    if model_name.endswith('_rf'):
        model = joblib.load('RandomForest/models/' + model_name + '.joblib')
        model.n_jobs = -1  # predicting with 1200 deep trees on one core is the slow step, use every core
        scaler = joblib.load('RandomForest/models/' + model_name + '_scaler.joblib')
    else:
        model = keras.models.load_model('Neural_Networks/models/' + model_name + '.keras')
        scaler = joblib.load('Neural_Networks/models/' + model_name + '_scaler.joblib')
    return model, scaler


def relative_importance(model, X_test, Y_test):  # Returns the test R2, the importance share of each feature
    result = permutation_importance(model, X_test, Y_test, scoring=r2_scorer, n_repeats=n_repeats,  # and the
                                    random_state=seed)                                              # raw R2 drops
    drops = result.importances_mean
    if (drops < 0).any():
        print(f'  negative R2 drop for {[f for f, d in zip(my_features, drops) if d < 0]}')
    # Shuffling a feature the model does not rely on can raise R2 slightly by chance; that feature gets no
    # importance, otherwise a negative drop pushes the other shares above 1 and a near zero sum makes them explode
    kept = np.clip(drops, 0, None)
    shares = kept / kept.sum() if kept.sum() > 0 else np.full(len(drops), np.nan)
    return r2_scorer(model, X_test, Y_test), shares, drops


if __name__ == '__main__':
    cluster_creator = ClusterCreator.build_clusters()
    groups = zip(['pft_', 'biome_'], [cluster_creator.func_cluster_dict, cluster_creator.biome_cluster_dict])
    # An interrupted run picks up where it stopped, delete the output file to measure every model again
    rows = pd.read_csv(output_file).to_dict('records') if os.path.exists(output_file) else []
    finished = {(row['Data set'], row['Model']) for row in rows}
    for identifier, cluster_group in groups:  # the same clusters the optimization scripts loop over
        for data_cluster in cluster_group:
            key = cluster_key(identifier, data_cluster)
            if {(key, 'RF'), (key, 'ANN')} <= finished:
                continue
            X, Y, info = data_import(my_features, cluster_group[data_cluster], return_info=True)
            is_test = load_test_mask(key, info)  # drawn once by test_set_builder.py
            X_test, Y_test = X[is_test], Y[is_test]
            for suffix, label in (('_rf', 'RF'), ('_ann', 'ANN')):
                if (key, label) in finished:
                    continue
                model, scaler = load_model(key + suffix)
                r2, shares, _ = relative_importance(model, scaler.transform(X_test), Y_test)
                print(f'{key + suffix}: test R2 {r2:.3f}, ' +
                      ', '.join(f'{f} {s:.3f}' for f, s in zip(my_features, shares)), flush=True)
                rows.append({'Data set': key, 'Model': label, 'n locations': info['Location'].nunique(),
                             'n test points': int(is_test.sum()), 'R2 test': r2,
                             **dict(zip(my_features, shares))})
                pd.DataFrame(rows).to_csv(output_file, index=False)  # rewritten after every model so an
    print(f'Written to {output_file}')                                     # interrupted run keeps what it finished
