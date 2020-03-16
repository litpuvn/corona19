import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt
# from pylab import *
# import numpy as np
#
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
#
# subplots_adjust(hspace=0.000)
# number_of_subplots=3
#
# col = 1
# for i,v in enumerate(range(number_of_subplots)):
#     v = v+1
#     ax1 = plt.subplot(number_of_subplots, col, v)
#     ax1.plot(x,y)
#
# plt.show()


# countries = ['Italy', 'Spain', 'France', 'Germany']
countries = ['United Kingdom', 'South Korea', 'United States', 'China']

filepath = 'data/gtrends.csv'
df_gtrends = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
df_gtrends = df_gtrends[37:]

df_new_deaths = pd.read_csv('data/new_deaths.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_deaths = df_new_deaths[:-1]
df_new_cases = pd.read_csv('data/new_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_cases = df_new_cases[:-1]

# fig, (ax1, ax2, ax3) = plt.subplots(len(countries))

for i, country in enumerate(countries):

    if country not in df_gtrends.columns or country not in df_new_deaths.columns or country not in df_new_cases.columns:
        print('ignoring country:', country)
        continue

    ax_i = plt.subplot(len(countries), 1,  i+1)
    ax_i.fill_between(df_gtrends.index, df_gtrends[country], color="blue", alpha=0.4)
    ax_i.plot(df_gtrends.index, df_gtrends[country], marker='', color='blue', linewidth=1)
    ax_i.plot(df_new_deaths.index, df_new_deaths[country], marker='', color='red', linewidth=1, linestyle='dashed')
    ax_i.plot(df_new_cases.index, df_new_cases[country], marker='', color='olive', linewidth=1)
    ax_i.legend(['trends', 'new deaths', 'new cases'])
    ax_i.set_ylabel(country, size=12)
    ax_i.set_xlabel('Date', size=12)


# ax1.fill_between(df_gtrends.index, df_gtrends['world'], color="blue", alpha=0.4)
# ax1.plot(df_gtrends.index, df_gtrends['world'], marker='', color='blue', linewidth=1)
# ax1.plot(df_new_deaths.index, df_new_deaths['Worldwide'], marker='', color='red', linewidth=1, linestyle='dashed')
# ax1.plot(df_new_cases.index, df_new_cases['Worldwide'], marker='', color='olive', linewidth=1)
# ax1.legend(['trends', 'new deaths', 'new cases'])
# ax1.set_ylabel('Good Trends Search', size=12)
# ax1.set_xlabel('Date', size=12)
#
#
# ax2.fill_between(df_gtrends.index, df_gtrends['world'], color="blue", alpha=0.4)
# ax2.plot(df_gtrends.index, df_gtrends['world'], marker='', color='blue', linewidth=1)
# ax2.plot(df_new_deaths.index, df_new_deaths['Worldwide'], marker='', color='red', linewidth=1, linestyle='dashed')
# ax2.plot(df_new_cases.index, df_new_cases['Worldwide'], marker='', color='olive', linewidth=1)
# ax2.legend(['trends', 'new deaths', 'new cases'])
# ax2.set_xlabel('Date', size=12)
# ax2.set_ylabel('Good Trends Search', size=12)

# plt.tick_params(labelsize=12)
# plt.xticks(x_values, x_labels)
# plt.ylim(bottom=0)

plt.show()