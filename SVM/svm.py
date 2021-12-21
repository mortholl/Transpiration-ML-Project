from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from utilities.data_sanitizer import data_import
from utilities.cluster_creator import ClusterCreator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

creator = ClusterCreator()
creator.k_means_clusters()
creator.func_type_clusters()
creator.biome_clusters()
k_clusters = creator.k_cluster_dict
func_clusters = creator.func_cluster_dict
biome_clusters = creator.biome_cluster_dict

my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']
my_files = biome_clusters['Tropical rain forest']  # can select using the cluster dictionaries or use [] for all

X, Y = data_import(my_features, my_files)
param_grid = {'C': [0.1, 1, 100],
              'gamma': [1, 0.1, 0.01],
              'kernel': ['rbf']}


def svm_search(x, y, params):
    # Grid search on different parameters to find the best support vector machine
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    #  Split to training/validation sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    svr = SVR()
    svr_cv = GridSearchCV(svr, params, verbose=1, scoring='r2')
    svr_cv.fit(x_train, y_train)

    # Model validation
    best_params = svr_cv.best_params_
    print(f"Best parameters from the grid search were {best_params}")
    r2_train = svr_cv.best_score_
    print(f"R2 of training set is {r2_train}")

    y_pred = svr_cv.predict(x_test)
    r2_test = r2_score(y_test, y_pred)
    print(f"R2 of test set is {r2_test}")

    mse_test = mean_squared_error(y_test, y_pred)
    print(f'MSE of test set is {mse_test}')

    res = permutation_importance(svr_cv, x_train, y_train, scoring='r2', n_repeats=5, random_state=42)
    p_importances = res['importances_mean']/res['importances_mean'].sum()
    print(f"The permutation-based feature importance is {p_importances}")
    return svr_cv


best_svm = svm_search(X, Y, param_grid)
