import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.engine.topology import Input
from keras.layers.core import RepeatVector, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from pandas.core.frame import DataFrame
from pandas import concat
from common.utils import split_train_validation_test, flatten_test_predict, store_predict_points
from keras.models import Model as KerasModel
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn import metrics
from common.utils import mape


target = pd.read_csv('data/target_confirmed_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
target['t+1'] = target['confirmed_cases'].shift(1 * -1, freq='d')
target.dropna(how='any', inplace=True)

X = pd.DataFrame(target['confirmed_cases'], index=target.index)
X.columns = ['corona']
y = pd.DataFrame(target['t+1'], index=target.index)
y.columns = ['confirmed_cases']

# X = features
# y = target['confirmed_cases']

feature_count = 1
print('feature count:', feature_count)
X = X.values.reshape(-1, 1, feature_count)
Y_entire = y['confirmed_cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

HORIZON = 1
LATENT_DIM = 16
BATCH_SIZE = 32
EPOCHS = 100
KERNEL_SIZE = 2

model = Sequential()
# model.add(LSTM(LATENT_DIM, input_shape=(1, feature_count)))
# model.add(Dense(32, activation='tanh'))
# model.add(Dense(1))

model.add(
    Conv1D(filters=16, kernel_size=KERNEL_SIZE, padding='causal', strides=3, activation='relu', dilation_rate=1,
           input_shape=(1, feature_count)))
model.add(
    Conv1D(filters=16, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
model.add(
    Conv1D(filters=16, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
model.add(Flatten())
model.add(Dropout(rate=0.05))
model.add(Dense(HORIZON, activation='linear'))

model.compile(optimizer='adam', loss='mse')
print(model.summary())
earlystop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[earlystop],
                    verbose=1)



y_pred = model.predict(X_test)

predicted_Y_entire = model.predict(X)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mape_v = mape(y_pred.reshape(-1, 1), y_test.values.reshape(-1, 1))
print('mape:', mape_v)
r2 = metrics.r2_score(y_test, y_pred)
print("R2:",  r2)
store_predict_points(Y_entire, predicted_Y_entire, 'output/test_cnn_without_prediction_epochs_r2_' + str(r2) + '.csv')