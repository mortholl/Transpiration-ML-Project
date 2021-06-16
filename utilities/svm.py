# Functions to build SVMs and visualize their performance


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


def svm_search(x_train, x_test, y_train, y_test, params):
    # Grid search on different parameters to find the best support vector machine
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

# Data visualization


def svm_visualize(x, y, svr, times, name):
    plt.figure(figsize=(20, 5))
    y_pred = svr.predict(x)
    plt.plot(times, y_pred, linestyle='dotted', label='Model prediction')
    plt.plot(times, y, linestyle='solid', label='Measured')
    plt.xticks(np.arange(1, 200, 25))
    plt.legend()
    plt.title(name)
    plt.xlabel('Time')
    plt.ylabel('Average transpiration (mm/d)')
