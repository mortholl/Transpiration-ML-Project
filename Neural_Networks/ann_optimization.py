import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import pickle

begin_time = datetime.datetime.now()

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict


# Pick relevant features
my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow']


# Define model creation in function
def create_model(n_hidden=2, n_neuron=20, regul_weight=0.01, lr=0.001):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_neuron, activation='relu', input_shape=(X_train.shape[1:]),
                                 kernel_regularizer=keras.regularizers.l2(regul_weight)))
    for i in range(1, n_hidden):
        model.add(keras.layers.Dense(n_neuron, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(regul_weight)))
    keras.layers.BatchNormalization()
    model.add(keras.layers.Dense(1, name='output', activation=None))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


# Wrap model in scikit learn estimator, define parameters to test
sk_estimator = KerasRegressor(build_fn=create_model, batch_size=32, verbose=0)
param_grid = {'n_hidden': [8, 10],
              'n_neuron': [24, 32],
              'epochs': [120, 150],
              # 'regul_weight': [1e-1, 1e-2, 1e-3],
              # 'lr': [1e-2, 1e-3, 1e-4],
              }


with open('Neural_Networks/ann_results.csv', 'w', newline='') as csvfile:
    csvfile.write(f'Data set, n locations, n data points, R2 test, R2 train, MAE, {",".join(my_features)}, Best parameters \n')

    # Loop over all clusters
    for identifier, cluster_group in zip(['k_means_', 'func_', 'biome_'], [k_clusters, func_clusters, biome_clusters]):
        for data_cluster in cluster_group:
            # Get data
            my_files = cluster_group[data_cluster]
            n_files = len(my_files)
            model_name = f'{identifier}{data_cluster}_ann'
            model_name = model_name.replace('/', '')
            X, Y = data_import(my_features, my_files)
            n_points = len(X)
            scaler = StandardScaler()
            outfile = 'Neural_Networks/models/'+model_name+'_scaler.sav'
            pickle.dump(scaler, open(outfile, 'wb'))
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

            # Set random seeds for reproducibility
            keras.backend.clear_session()
            np.random.seed(42)
            tf.random.set_seed(42)

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
            outfile = 'Neural_Networks/models/'+model_name+'.h5'
            model.model.save(outfile)
            csvfile.write(f'{model_name}, {n_files}, {n_points}, {r2}, {r2_train}, {mae}, {feature_importances}, {ann_grid.best_params_} \n')
            print(f'{model_name} complete')

end_time = datetime.datetime.now()
print(f'The run time was {end_time-begin_time}')
