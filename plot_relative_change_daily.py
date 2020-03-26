import matplotlib.pyplot as plt
import numpy as np
import os
# Fixing random state for reproducibility
from datetime import timedelta, date
import pandas as pd

np.random.seed(19680801)
plt.rcdefaults()

fig, ax = plt.subplots(figsize=(1, 9))

start_year = 2020
start_mon = 1
start_day = 21

start_date = date(start_year, start_mon, start_day)
day_count = 61

days = []
colors = []

keyword = 'covid19 cases'
folder = os.path.join('data/daily-complete', keyword.replace(' ', '-', 5))

def extract_data(filepath):
    global keyword

    try:
        df = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
        if len(df) < 1:
            return 0

        val1 = df[keyword].values[0]
        val2 = df[keyword].values[1]
        print("val1:", val1, ";val2:", val2, ";change:", (val2-val1))
        return (val2 - val1)
    except:
        print('Error reading file:', filepath)

    return 0


def get_color(index, change):
    if i % 2 == 0:
        if change > 0:
            return '#344e9d'
        return '#fa2e30'
    else:
        if change > 0:
            return '#00a6dd'

    return '#fc987d'


performance = []
for i in range(day_count):

    day_string = start_date.strftime('%Y-%m-%d')
    filename = os.path.join(folder, day_string + '-interest.csv')

    if not os.path.isfile(filename):
        print('File', filename, 'does not exist')
        continue

    print('processing for file: ', filename)

    days = days + [day_string]
    start_date = start_date + timedelta(1)

    val = extract_data(filepath=filename)
    performance = performance + [val]

    my_color = get_color(index=i, change=val)
    colors = colors + [my_color]
# Example data

y_pos = np.arange(len(days))
ax.barh(y_pos, performance, color=colors, xerr=None, align='center')
# ax.set_yticks(y_pos)
ax.set_yticklabels([])
ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Relative Change (%)')
ax.set_title(keyword)

# plt.ylim(-2, 2)

plt.show()