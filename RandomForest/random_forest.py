from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import pickle
from sklearn.model_selection import GridSearchCV


begin_time = datetime.datetime.now()

# Create clusters of files to select from
cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict

# Select desired features and files
my_features = ['ta', 'vpd', 'ppfd_in', 'swc_shallow']
my_files = biome_clusters['Woodland/Shrubland']  # can select using the cluster dictionaries or use [] for all

param_grid = {'n_estimators': [600, 800, 1200],
              'max_depth': [20, 25],
              }

# Import data and scale X inputs
X, Y = data_import(my_features, my_files)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Split to training/validation sets: 90% training, 10% test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=500, max_depth=9, random_state=42)
rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', verbose=3, n_jobs=5, return_train_score=True)
rf_grid.fit(X_train, Y_train)

# feature_importances = [str(round(n, 4)) for n in rf_grid.feature_importances_]
# print(feature_importances)
Y_pred = rf_grid.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f'R2 score was {r2}')
Y_pred_train = rf_grid.predict(X_train)
r2_train = r2_score(Y_train, Y_pred_train)
if r2_train > r2:
    print('Model may be overfitting')

end_time = datetime.datetime.now()
print(f'The runtime was {end_time - begin_time}.')

plt.scatter(Y_test, Y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('RandomForest/rf_plot.png')
plt.show()

# Filing results: model, metrics, plot
# outfile = 'RandomForest/rf.sav'
# pickle.dump(rf, open(outfile, 'wb'))
# with open('RandomForest/rf_results.csv', 'w', newline='') as csvfile:
#     csvfile.write(f'Data set ')
#     csvfile.write(f'R2 test, R2 train, MAE, {",".join(my_features)} \n')
#     csvfile.write(f'{r2}, {r2_train}, {mae}, {",".join(feature_importances)} \n')
