import pandas as pd
from sklearn import linear_model
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

target = pd.read_csv('data/target_confirmed_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
features = pd.read_csv('data/features.csv', sep=',', header=0, index_col=0, parse_dates=True)

X = features
y = target['confirmed_cases']
lm = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2:",  metrics.r2_score(y_test, y_pred))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.plot(kind='bar',figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()