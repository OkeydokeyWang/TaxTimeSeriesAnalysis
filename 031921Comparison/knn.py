from sklearn.model_selection import KFold
from sklearn import neighbors
import time
from utils import rmse, mae
import sys
import numpy as np

def knn_model_selection(train_x, train_y, nn_params):
    K=5

    # Vars to store results
    cval_errs = {}
    train_time = {}
    best_af_nn_regressor = None
    best_err = sys.maxsize

    for nn in nn_params:
        kf = KFold(n_splits=K, random_state=None, shuffle=False)
        y_err = []

        start = time.time()

        # Cross Validaiton
        for train_index, val_index in kf.split(train_x):
        #     print("TRAIN:", train_index, "VAL:", val_index)
            X_train, X_val = train_x[train_index], train_x[val_index]
            y_train, y_val = train_y[train_index], train_y[val_index]

            NNRegressor = knn_model(nn)
            y_hat = NNRegressor.fit(X_train, y_train).predict(X_val)
        #     print(y_hat)
            y_err.append(rmse(y_hat, y_val))

        end = time.time()

        print(str(nn), "mean val RMSE:", np.mean(y_err))
        print("Time lapsed", str((end - start)*1000))

        # add to dict
        cval_errs[str(nn)] = np.mean(y_err)
        train_time[str(nn)] = (end - start)*1000
        if np.mean(y_err) < best_err:
            best_af_nn_regressor = knn_model(nn)
            best_af_nn_regressor = best_af_nn_regressor.fit(train_x, train_y)
            best_err = np.mean(y_err)

    return best_af_nn_regressor, best_err

def knn_model(number_of_neighbors):
    return neighbors.KNeighborsRegressor(n_neighbors=number_of_neighbors)