import pandas as pd
from sklearn import linear_model
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from common.utils import split_train_validation_test, flatten_test_predict, store_predict_points
from common.utils import mape

target = pd.read_csv('data/target_confirmed_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
target['t+1'] = target['confirmed_cases'].shift(1 * -1, freq='d')

target.dropna(how='any', inplace=True)

print(target.head())

X = pd.DataFrame(target['confirmed_cases'], index=target.index)
y = target['t+1']
Y_entire = y
lm = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
X_test = X_test.sort_index()
y_test = y_test.sort_index()
X_train = X_train.sort_index()
y_train = y_train.sort_index()

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print("R2:",  metrics.r2_score(y_test, y_pred))

predicted_Y_entire = regressor.predict(X)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mape_v = mape(y_pred.reshape(-1, 1), y_test.values.reshape(-1, 1))
print('mape:', mape_v)
r2 = metrics.r2_score(y_test, y_pred)
print("R2:",  r2)
store_predict_points(Y_entire, predicted_Y_entire, 'output/test_linear_without_prediction_r2_' + str(r2) + '.csv')


# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# df.plot(kind='bar',figsize=(10, 8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()