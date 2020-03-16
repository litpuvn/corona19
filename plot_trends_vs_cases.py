import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8, 5))

filepath = 'data/gtrends.csv'
df_gtrends = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
df_gtrends = df_gtrends[37:]
df_gtrends['world'] = df_gtrends.sum(axis=1)


df_new_deaths = pd.read_csv('data/new_deaths.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_cases = pd.read_csv('data/new_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_total_cases = pd.read_csv('data/total_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_total_deaths = pd.read_csv('data/total_deaths.csv', sep=',', header=0, index_col=0, parse_dates=True)



plt.fill_between(df_gtrends.index, df_gtrends['world'], color="blue", alpha=0.4)
plt.plot(df_gtrends.index, df_gtrends['world'], marker='', color='blue', linewidth=1)
plt.plot(df_new_deaths.index, df_new_deaths['Worldwide'], marker='', color='red', linewidth=1, linestyle='dashed')
plt.plot(df_new_cases.index, df_new_cases['Worldwide'], marker='', color='olive', linewidth=1)
plt.plot(df_total_cases.index, df_total_cases['Worldwide'] / 5, marker='', color='skyblue', linewidth=1)
plt.plot(df_total_deaths.index, df_total_deaths['Worldwide'], marker='', color='#8ebad9', linewidth=1)


# y_values = [2, 7, 14, 17, 20, 27, 30, 38, 25, 18, 6, 1]
# x_values = np.arange(12) # 0 to 11
# x_labels = np.arange(13) # 1 to 12
#
#
# plt.fill_between(x_values, y_values, color="skyblue", alpha=0.4)
# plt.plot(x_values, y_values, color="Slateblue", alpha=0.6, linewidth=2)
# plt.plot(x_values, [3*y for y in y_values], color="Slateblue", alpha=0.6, linewidth=2)

plt.tick_params(labelsize=12)
# plt.xticks(x_values, x_labels)
plt.xlabel('Date', size=12)
plt.ylabel('Good Trends Search', size=12)
# plt.ylim(bottom=0)

plt.legend(['trends', 'new deaths', 'new cases', 'total deaths', 'total cases'])
plt.show()