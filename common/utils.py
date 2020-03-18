import numpy as np
import pandas as pd
import os
from collections import UserDict

from pandas.core.indexes.datetimes import DatetimeIndex
import datetime as dt

from sklearn.preprocessing.data import MinMaxScaler
from sklearn.preprocessing.data import StandardScaler

from common.TimeseriesTensor import TimeSeriesTensor
from sklearn.utils import check_array
import csv



def split_train_validation_test(multi_time_series_df, valid_start_time, test_start_time, features,
                                time_step_lag=1, horizon=1, target='target', time_format='%Y-%m-%d %H:%M:%S', freq='H'):

    if not isinstance(features, list) or len(features) < 1:
        raise Exception("Bad input for features. It must be an array of dataframe colummns used")

    train = multi_time_series_df.copy()[multi_time_series_df.index < valid_start_time]
    train_features = train[features]
    train_targets = train[target]

    # X_scaler = MinMaxScaler()
    # target_scaler = MinMaxScaler()
    # y_scaler = MinMaxScaler()

    X_scaler = None
    target_scaler = None
    y_scaler = None

    # X_scaler = StandardScaler()
    # target_scaler = StandardScaler()
    # y_scaler = StandardScaler()


    # 'load' is our key target. If it is in features, then we scale it.
    # if it not 'load', then we scale the first column
    if 'load' in features:
        tg = train[['load']]
        if y_scaler is not None:
            y_scaler.fit(tg)
    else:

        tg = train[target]
        ## scale the first column
        if y_scaler is not None:
            y_scaler.fit(tg.values.reshape(-1, 1))

    if target_scaler is not None:
        train[target] = target_scaler.fit_transform(train_targets)

    if X_scaler is not None:
        X_scaler.fit(train_features)
        train[features] = X_scaler.transform(train_features)

    tensor_structure = {'X': (range(-time_step_lag + 1, 1), features)}
    train_inputs = TimeSeriesTensor(train, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print(train_inputs.dataframe.head())


    look_back_dt = dt.datetime.strptime(valid_start_time, time_format) - dt.timedelta(hours=time_step_lag - 1)
    valid = multi_time_series_df.copy()[(multi_time_series_df.index >= look_back_dt) & (multi_time_series_df.index < test_start_time)]
    valid_features = valid[features]

    if X_scaler is not None:
        valid[features] = X_scaler.transform(valid_features)
    tensor_structure = {'X': (range(-time_step_lag + 1, 1), features)}
    valid_inputs = TimeSeriesTensor(valid, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print(valid_inputs.dataframe.head())

    # test set
    # look_back_dt = dt.datetime.strptime(test_start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=time_step_lag - 1)
    test = multi_time_series_df.copy()[test_start_time:]
    test_features = test[features]

    if X_scaler is not None:
        test[features] = X_scaler.transform(test_features)
    test_inputs = TimeSeriesTensor(test, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print("time lag:", time_step_lag, "original_feature:", len(features))

    return train_inputs, valid_inputs, test_inputs, y_scaler


def mape(predictions, actuals):
    predictions = check_array(predictions)
    actuals = check_array(actuals)

    """Mean absolute percentage error"""
    return ( np.mean(np.abs(predictions - actuals) / actuals))


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df



def flatten_test_predict(y_tests, y_predicts):
    if len(y_tests) == 1:
        y_tests = y_tests[0]

    if len(y_predicts) == 1:
        y_predicts = y_predicts[0]

    if not isinstance(y_tests, np.ndarray) or not isinstance(y_predicts, np.ndarray):
        raise Exception('bad input data for y_tests or y_predicts')

    y_tests = y_tests.ravel()
    y_predicts = y_predicts.ravel()

    return y_tests, y_predicts



def store_predict_points(y_tests, y_predicts, filepath):
    n = len(y_tests)
    if n != len(y_predicts):
        raise Exception('bad testing samples and predictions')

    ri = filepath.rfind('/')
    folder = filepath[0:ri]
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath, 'w', newline='') as writer:
        csv_writer = csv.writer(writer, delimiter=',')
        csv_writer.writerow(["y_test", "y_pred"])

        for i in range(len(y_tests)):
            csv_writer.writerow([y_tests[i], y_predicts[i]])

        print("Done writing prediction result to", filepath)