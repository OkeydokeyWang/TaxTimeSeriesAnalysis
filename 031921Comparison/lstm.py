import sys
import time
import pandas as pd
import numpy as np
import datetime
import gc
from utils import rmse, mae

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#importing required libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, TimeDistributed


def lstm_model_selection(train_x, train_y, unit_params):
    # How many features do we want to pass as hidden unit
    K = 5
    model = None
    gc.collect()

    best_unit_count = None
    # Best Validation Error
    best_err = sys.maxsize

    for unit_count in unit_params:
        kf = KFold(n_splits=K, random_state=None, shuffle=False)
        y_err = []
        
        # Cross Validaiton
        for train_index, val_index in kf.split(train_x):
            X_train, X_val = train_x[train_index], train_x[val_index]
            y_train, y_val = train_y[train_index], train_y[val_index]
        
            model = lstm_model(train_x, unit_count)
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
            y_hat = model.predict(X_val).transpose()[0]
            del model
            gc.collect()
            
            y_err.append(rmse(y_hat, y_val))
            
        print(str(unit_count), "mean val RMSE:", np.mean(y_err))

        if np.mean(y_err) < best_err:
            best_err = np.mean(y_err)
            best_unit_count = unit_count

    return best_unit_count, best_err
    
def lstm_model(train_x, unit_count):
    model = Sequential()
    model.add(LSTM(units=unit_count, return_sequences=True, input_shape=(train_x.shape[1],train_x.shape[2])))
    model.add(Flatten())
    model.add(Dense(unit_count))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    