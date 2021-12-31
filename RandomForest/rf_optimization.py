from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
from sklearn.model_selection import GridSearchCV
import pickle

begin_time = datetime.datetime.now()

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict

my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']

param_grid = {'n_estimators': [200, 600, 800],
              'max_depth': [8, 10, 12],
              }

rf = RandomForestRegressor(n_estimators=500, max_depth=9, random_state=42)

with open('RandomForest/rf_results.csv', 'w', newline='') as csvfile:
    csvfile.write(f'Data set, n locations, n data points, R2 test, R2 train, MAE, {",".join(my_features)}, Best parameters \n')

    # Loop over all clusters
    for identifier, cluster_group in zip(['k_means_', 'func_', 'biome_'], [k_clusters, func_clusters, biome_clusters]):
        for data_cluster in cluster_group:
            # Get data
            my_files = cluster_group[data_cluster]
            n_files = len(my_files)
            model_name = f'{identifier}{data_cluster}_rf'
            model_name = model_name.replace('/', '')
            X, Y = data_import(my_features, my_files)
            n_points = len(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
            X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

            # Grid search to find optimal hyperparameters
            rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', verbose=3, n_jobs=5)
            rf_grid.fit(X_train, Y_train)

            # Get metrics
            model = rf_grid.best_estimator_
            feature_importances = [str(round(n, 4)) for n in model.feature_importances_]
            Y_pred = model.predict(X_test)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)
            r2_train = rf_grid.best_score_
            plt.scatter(Y_test, Y_pred)
            plt.xlabel('True values')
            plt.ylabel('Predicted values')
            plt.savefig('RandomForest/plots/'+model_name+'.png')
            plt.clf()
            outfile = 'RandomForest/models/'+model_name+'.sav'
            pickle.dump(rf, open(outfile, 'wb'))
            csvfile.write(f'{n_files}, {n_points}, {model_name}, {r2}, {r2_train}, {mae}, {",".join(feature_importances)}, {rf_grid.best_params_} \n')
            print(f'{model_name} complete')

end_time = datetime.datetime.now()
print(f'The run time was {end_time-begin_time}')
