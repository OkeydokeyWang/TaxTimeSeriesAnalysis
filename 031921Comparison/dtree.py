from sklearn import tree
from sklearn.model_selection import KFold
import time
from utils import rmse, mae
import sys
import numpy as np

def dicision_tree_model_selection(train_x, train_y, criteria_params, max_depth_params):
    K = 5

    # Vars to store results
    cval_errs = {}
    train_time = {}
    best_af_nn_regressor = None
    best_err = sys.maxsize

    for criterion in criteria_params:
        for max_dep in max_depth_params:
            kf = KFold(n_splits=K, random_state=None, shuffle=False)
            y_err = []

            start = time.time()

            # Cross Validaiton
            for train_index, val_index in kf.split(train_x):
            #     print("TRAIN:", train_index, "VAL:", val_index)
                X_train, X_val = train_x[train_index], train_x[val_index]
                y_train, y_val = train_y[train_index], train_y[val_index]

                DTRegressor = dicision_tree_model(criterion, max_dep)
                y_hat = DTRegressor.fit(X_train, y_train).predict(X_val)
            #     print(y_hat)
                y_err.append(rmse(y_hat, y_val))

            end = time.time()

            print(str(criterion), str(max_dep), "mean val MSE:", np.mean(y_err))
#             print("Time lapsed", str((end - start)*1000))

            # add to dict
            cval_errs[str(criterion)+str(max_dep)] = np.mean(y_err)
            train_time[str(criterion)+str(max_dep)] = (end - start)*1000
            if np.mean(y_err) < best_err:
                best_af_dt_regressor = dicision_tree_model(criterion, max_dep)
                best_af_dt_regressor = best_af_dt_regressor.fit(train_x, train_y)
                best_err = np.mean(y_err)
                best_errs = y_err

    return best_af_dt_regressor, best_err

def dicision_tree_model(criterion, max_dep):
    return tree.DecisionTreeRegressor(criterion=criterion, max_depth=max_dep)