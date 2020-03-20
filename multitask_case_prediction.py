import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.engine.topology import Input
from keras.layers.core import RepeatVector
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from pandas.core.frame import DataFrame
from pandas import concat
from common.utils import split_train_validation_test, flatten_test_predict, store_predict_points
from keras.models import Model as KerasModel


filepath = 'data/gtrends.csv'
df_gtrends = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
df_gtrends = df_gtrends[37:]
df_gtrends['Worldwide'] = df_gtrends.sum(axis=1)

df_new_cases = pd.read_csv('data/new_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_cases = df_new_cases[:-1]
df_new_cases.fillna(value=0, inplace=True)

# countries = ['Italy', 'Spain', 'France', 'Germany']
countries = ['United Kingdom', 'South Korea', 'United States', 'China', 'Worldwide']
selected_index = 4
selected_country = countries[selected_index]

selected_country_cases_df = DataFrame(df_new_cases[selected_country].values, index=df_new_cases.index)
selected_country_gtrends_df = DataFrame(df_gtrends[selected_country].values, index=df_gtrends.index)

concatenated_dfs = [
    selected_country_cases_df,
    selected_country_gtrends_df
]


combined_data = concat(concatenated_dfs, axis=1)

HORIZON = 1
time_step_lag = 1

multi_time_series = DataFrame(combined_data)
multi_time_series.columns = ['load', 'trends']
features = ["load", "trends"]
# features = ["load"]

# targets = ["load", "imf3", "imf2"]
targets = ["load", "trends"]

valid_start_dt = '2020-02-29'
test_start_dt = '2020-03-07'

train_inputs, valid_inputs, test_inputs, y_scaler, entire_inputs = split_train_validation_test(multi_time_series,
                                                                                valid_start_time=valid_start_dt,
                                                                                test_start_time=test_start_dt,
                                                                                time_step_lag=time_step_lag,
                                                                                horizon=HORIZON,
                                                                                features=features,
                                                                                target=targets,
                                                                                time_format='%Y-%m-%d',
                                                                                freq='d'
                                                                                )

print('done')

## build CNN
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

from common.utils import split_train_validation_test, mape

LATENT_DIM = 16
BATCH_SIZE = 32
EPOCHS = 200
KERNEL_SIZE = 2

X_train = train_inputs['X']
y_train = train_inputs['target_load']
y2_train = train_inputs['target_trends']
final_y_train = [y_train, y2_train]


X_valid = valid_inputs['X']
y_valid = valid_inputs['target_load']
y2_valid = valid_inputs['target_trends']
final_y_valid = [y_valid, y2_valid]

# input_x = train_inputs['X']
print("train_X shape", X_train.shape)
print("valid_X shape", X_valid.shape)
# print("target shape", y_train.shape)
# print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
# print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))
model = Sequential()
# model.add(LSTM(LATENT_DIM, input_shape=(time_step_lag, len(features))))

# model.add(Dense(16, activation='tanh', input_shape=(time_step_lag, len(features)))),
# model.add(Dense(32, activation='tanh'))
# model.add(Flatten())
# model.add(Dense(1))

# model.add(
#     Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=3, activation='relu', dilation_rate=1,
#            input_shape=(time_step_lag, len(features))))
# model.add(
#     Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
# model.add(
#     Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
# model.add(Flatten())
# model.add(Dense(HORIZON, activation='linear'))


x = Input(shape=(time_step_lag, len(features)), name="input_layer")
conv = Conv1D(kernel_size=1, filters=5, activation='relu')(x)
conv2 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
lstm = GRU(16, return_sequences=True)(conv2)
shared_dense = Dense(32, name="shared_layer")(lstm)
## sub1 is main task; units = reshape dimension multiplication
sub1 = GRU(units=64, name="task1")(shared_dense)
sub2 = GRU(units=16, name="task2")(shared_dense)

out1 = Dense(8, name="spec_out1")(sub1)
out1 = Dense(1, name="out1")(out1)

out2 = Dense(8, name="spec_out2")(sub2)
out2 = Dense(1, name="out2")(out2)

outputs = [out1, out2]

model = KerasModel(inputs=x, outputs=outputs)
# model.compile(optimizer='adam', loss='mse')


model.compile(optimizer='RMSprop', loss='mse')
model.summary()
earlystop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train,
                    final_y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_valid, final_y_valid),
                    callbacks=[earlystop],
                    verbose=1)

# Test the model
X_test = test_inputs['X']
y1_test = test_inputs['target_load']
y1_preds, _ = model.predict(X_test)

X_entire = entire_inputs['X']
Y_entire = entire_inputs['target_load']
predicted_Y_entire, predicted_trends_entire = model.predict(X_entire)

if y_scaler is not None:
    y1_test = y_scaler.inverse_transform(y1_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)
    predicted_Y_entire = y_scaler.inverse_transform(predicted_Y_entire)




y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)
Y_entire, predicted_Y_entire = flatten_test_predict(Y_entire, predicted_Y_entire)

mse = mean_squared_error(y1_test, y1_preds)

rmse_predict = sqrt(mse)
evs = explained_variance_score(y1_test, y1_preds)
mae = mean_absolute_error(y1_test, y1_preds)
mse = mean_squared_error(y1_test, y1_preds)

meae = median_absolute_error(y1_test, y1_preds)
r_square = r2_score(y1_test, y1_preds)

print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
      "mse:", mse, "meae:", meae, "r2:", r_square)

# output_actual_y = np.concatenate((y_train, y_valid, y1_test), axis=0)
# output_predicted_y = np.concatenate((y_predicted_train, y_predicted_valid, y1_preds), axis=0)

store_predict_points(Y_entire, predicted_Y_entire, 'output/test_lstm_prediction_epochs_' + str(EPOCHS) + '.csv')