import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import datetime
import numpy as np

begin_time = datetime.datetime.now()

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict

my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']
with open('Neural_Networks/ann_results.csv', 'w', newline='') as csvfile:
    csvfile.write(f'Data set, n locations, n data points, R2 test, R2 train, MAE, {",".join(my_features)} \n')

    # Loop over all clusters to create random forest models
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
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
            X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

            # Set random seeds for reproducibility
            keras.backend.clear_session()
            np.random.seed(42)
            tf.random.set_seed(42)

            # Create loop for hyperparameter testing here
            model = keras.models.Sequential([
                keras.layers.Dense(30, activation='relu', name='hidden1', input_shape=X_train.shape[1:]),
                keras.layers.Dense(20, activation='relu', name='hidden2'),
                keras.layers.Dense(20, activation='relu', name='hidden3'),
                keras.layers.Dense(20, activation='relu', name='hidden4'),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(1, name='output', activation=None)
            ])

            model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.0001), metrics=['mae'])
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
            history = model.fit(X_train, Y_train, epochs=100, batch_size=30, validation_data=(X_val, Y_val),
                                callbacks=early_stopping_cb)


            # Get metrics
            mse, mae = model.evaluate(X_test, Y_test)
            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, Y_pred)
            Y_pred_train = model.predict(X_train)
            r2_train = r2_score(Y_train, Y_pred_train)
            feature_importances = permutation_importance(model, X_train, Y_pred_train, scoring='r2', random_state=42)

            plt.scatter(Y_test, Y_pred)
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.savefig('Neural_Networks/plots/'+model_name+'.png')
            plt.clf()
            outfile = 'Neural_Networks/models/'+model_name+'.h5'
            model.save(outfile)
            csvfile.write(f'{model_name}, {n_files}, {n_points}, {r2}, {r2_train}, {mae}, {feature_importances.importances_mean} \n')
            print(f'{model_name} complete')

end_time = datetime.datetime.now()
print(f'The run time was {end_time-begin_time}')
