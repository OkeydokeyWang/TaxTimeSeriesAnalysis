import pandas as pd
import dataloader
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# Compute MAE
def mae(y_hat, y):
    # mean absolute error
    return np.abs(y_hat - y).mean()

def rmse(y_hat, y):
    # root mean squared error
    return np.sqrt(np.mean(np.power((y-y_hat),2)))

def get_train_and_test_set(path, date, count, history_size, training_count, column="Close"):
    """
        load dataset from {path}, with end date to be {date}. 
        Load {count} number of entries with {training_count} number of the entries being training set with each of the training data consist of previous .
        return training set array and test set array.
    """
    df = dataloader.load_data_up_to_date(path, date, count)
    amzn_closing_all = df[column].to_numpy()
    # reshape to count x 1 matrix
    amzn_closing_all = np.reshape(amzn_closing_all, (amzn_closing_all.shape[0], -1))
    amzn_closing_all.shape
    
    #converting prices to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(amzn_closing_all)
    train = scaled_data[:training_count,:]
    test = scaled_data[training_count:, :]
    
    train_x, train_y = [], []
    test_x, test_y = [], []
    for i in range(history_size,len(train)):
        train_x.append(scaled_data[i-history_size:i,0])
        train_y.append(scaled_data[i,0])
    train_x, train_y = np.array(train_x), np.array(train_y)

    train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1], 1))
    print(train_x.shape)

    for i in range(len(train),len(scaled_data)):
        test_x.append(scaled_data[i-history_size:i,0])
        test_y.append(scaled_data[i,0])
    test_x, test_y = np.array(test_x), np.array(test_y)

    test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1], 1))
    print(test_x.shape)
    
    return (train_x, train_y, test_x, test_y)