import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.preprocessing import StandardScaler
from utilities.test_set_builder import cluster_key, load_test_mask
import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import pickle

begin_time = datetime.datetime.now()

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict


# Pick relevant features
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']


# Define model creation in function
def create_model(meta, n_hidden=2, n_neuron=20, regul_weight=0.01, lr=0.001):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(meta['n_features_in_'],)))  # scikeras passes the feature count in
    model.add(keras.layers.Dense(n_neuron, activation='relu',       # meta, the shape is no longer read
                                 kernel_regularizer=keras.regularizers.l2(regul_weight)))  # off X_train
    for i in range(1, n_hidden):
        model.add(keras.layers.Dense(n_neuron, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(regul_weight)))
    model.add(keras.layers.Dense(1, name='output', activation=None))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


# Wrap model in scikit learn estimator, define parameters to test
sk_estimator = KerasRegressor(model=create_model, batch_size=32, verbose=0)
param_grid = {'model__n_hidden': [8, 10],   # scikeras routes the model__ parameters to create_model and
              'model__n_neuron': [24, 32],  # keeps the rest for fit(), so epochs stays unprefixed
              'epochs': [120, 150],
              # 'model__regul_weight': [1e-1, 1e-2, 1e-3],
              # 'model__lr': [1e-2, 1e-3, 1e-4],
              }


with open('Neural_Networks/ann_results.csv', 'w', newline='') as csvfile:
    csvfile.write(f'Data set, n sites, n locations, n data points, R2 test, R2 train, MAE, {",".join(my_features)}, Best parameters \n')

    # Loop over all clusters
    for identifier, cluster_group in zip(['func_', 'biome_'], [func_clusters, biome_clusters]):  # add 'k_means_' and k_clusters to include the k-means groups
        for data_cluster in cluster_group:
            # Get data
            my_files = cluster_group[data_cluster]
            n_files = len(my_files)
            model_name = f'{identifier}{data_cluster}_ann'
            model_name = model_name.replace('/', '')
            X, Y, info = data_import(my_features, my_files, return_info=True)
            n_points = len(X)
            n_locations = info['Location'].nunique()
            is_test = load_test_mask(cluster_key(identifier, data_cluster), info)  # drawn once by
            X_train, X_test = X[~is_test], X[is_test]        # test_set_builder.py, so this model and the
            Y_train, Y_test = Y[~is_test], Y[is_test]        # other one are scored on the same rows
            order = np.random.default_rng(51).permutation(len(X_train))  # GridSearchCV folds by position,
            X_train, Y_train = X_train[order], Y_train[order]            # so the training rows are shuffled
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  # transform, never fit, on held out data
            outfile = 'Neural_Networks/models/'+model_name+'_scaler.sav'
            pickle.dump(scaler, open(outfile, 'wb'))

            # Set random seeds for reproducibility
            keras.backend.clear_session()
            np.random.seed(51)
            tf.random.set_seed(51)

            ann_grid = GridSearchCV(sk_estimator, param_grid, cv=5, scoring='r2', verbose=3, n_jobs=1, return_train_score=True)
            ann_grid.fit(X_train, Y_train)

            # Get metrics
            model = ann_grid.best_estimator_
            Y_pred = model.predict(X_test)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)
            r2_train = ann_grid.best_score_
            feature_importances = permutation_importance(model, X_train, Y_train)
            feature_importances = feature_importances.importances_mean
            feature_importances = feature_importances / np.sum(feature_importances)
            feature_importances = f'{[feature for feature in feature_importances]}'.replace('[', '').replace(']', '')
            plt.scatter(Y_test, Y_pred)
            plt.xlabel('True values [$cm^3/s$]')
            plt.ylabel('Predicted values [$cm^3/s$]')
            plt.title(model_name)
            r2_label = '$R^2$ = ' + str(round(r2, 3))
            mae_label = 'MAE = ' + str(int(round(mae, 0)))
            plt.annotate(r2_label, (0.8*max(Y_test), 0.1*max(Y_pred)))
            plt.annotate(mae_label, (0.8*max(Y_test), 0.2*max(Y_pred)))
            plt.savefig('Neural_Networks/plots/'+model_name+'.png')
            plt.clf()
            outfile = 'Neural_Networks/models/'+model_name+'.keras'
            model.model_.save(outfile)  # the fitted keras model behind the scikeras wrapper
            test_set = info[is_test].copy()  # kept for inspection, a rerun regenerates it
            test_set['observed'] = Y_test
            test_set['predicted'] = Y_pred
            test_set.to_csv('Neural_Networks/test_sets/' + model_name + '_test.csv', index=False)
            site_r2 = test_set.groupby('Site')[['observed', 'predicted']].apply(  # columns named so the
                lambda group: pd.Series({'n': len(group),  # grouping column is not passed to the lambda
                                         'r2': r2_score(group['observed'], group['predicted'])}))
            site_r2.to_csv('Neural_Networks/test_sets/' + model_name + '_site_r2.csv')
            csvfile.write(f'{model_name}, {n_files}, {n_locations}, {n_points}, {r2}, {r2_train}, {mae}, {feature_importances}, {ann_grid.best_params_} \n')
            print(f'{model_name} complete')

end_time = datetime.datetime.now()
print(f'The run time was {end_time-begin_time}')
