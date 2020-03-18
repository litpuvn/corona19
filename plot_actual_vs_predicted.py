import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8, 5))

filepath = 'output/test_lstm_prediction_epochs_200.csv'
df_results = pd.read_csv(filepath, sep=',', header=0)
df_new_cases = pd.read_csv('data/new_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)

# drop 1 because of the t+1 in the tensor X input;
# drop another 1 because the date of new cases is 1 day later compared to google trends
# so the total drop is 2
df_new_cases = df_new_cases[:-2]
df_new_cases.fillna(value=0, inplace=True)

df_new_cases = df_new_cases['United States']
plt.plot(df_new_cases.index, df_results['y_actual'], marker='', color='blue', linewidth=1)
plt.plot(df_new_cases.index, df_results['y_pred'], marker='', color='red', linewidth=1,  linestyle='dashed')


plt.tick_params(labelsize=12)
# plt.xticks(x_values, x_labels)
plt.xlabel('Date', size=12)
plt.ylabel('New confirmed cases', size=12)
# plt.ylim(bottom=0)

plt.legend(['Actual', 'Prediction'])
plt.show()