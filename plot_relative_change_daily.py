#Short Title of the Article
#Figure 4:Daily interest change of each keyword

import matplotlib.pyplot as plt
import numpy as np
import os
# Fixing random state for reproducibility
from datetime import timedelta, date
import pandas as pd

def extract_data(filepath, keyword):
    # global keyword

    try:
        df = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
        if len(df) < 1:
            return 0

        val1 = df[keyword].values[0]
        val2 = df[keyword].values[1]
        # print("val1:", val1, ";val2:", val2, ";change:", (val2-val1))
        return (val2 - val1)
    except:
        print('Error reading file:', filepath)

    return 0


def get_color(index, change):
    i = index
    if i % 2 == 0:
        if change > 0:
            return '#344e9d'
        return '#fa2e30'
    else:
        if change > 0:
            return '#00a6dd'

    return '#fc987d'


def plot_graph(keyword, is_last=False):
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

    folder = os.path.join('data/daily-complete', keyword.replace(' ', '-', 5))

    performance = []
    for i in range(day_count):

        day_string = start_date.strftime('%Y-%m-%d')
        filename = os.path.join(folder, day_string + '-interest.csv')
        # filename = os.path.abspath(filename)
        # print('os.path.abspath(filename)', os.path.abspath(filename))
        start_date = start_date + timedelta(1)
        if not os.path.isfile(filename):
            print('File', filename, 'does not exist')
            continue

        # print('processing for file: ', filename)

        days = days + [day_string]
        # start_date = start_date + timedelta(1)

        val = extract_data(filename, keyword)
        performance = performance + [val]

        my_color = get_color(index=i, change=val)
        colors = colors + [my_color]
    # Example data

    y_pos = np.arange(len(days))
    # print('days length:', len(days),', values:', days)
    ax.barh(y_pos, performance, color=colors, xerr=None, align='center')
    # ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    #4/15
    if is_last:
        ax.set_yticklabels(days)
    ax.set_xticklabels([])

    ax.invert_yaxis()  # labels read top-to-bottom

    # ax.set_xlabel('Relative Change (%)')
    # ax.set_title(keyword)

    plt.xlim(-100, 100)
    # plt.xticks(np.arange(-25, 100, step=100))

    # plt.show()
    out_path = 'figures/'
    out_filename = out_path + 'daily-change-of-' + keyword + '.png'
    out_filename = out_filename.replace(" ", "-")
    print('The figure save into:', out_filename)
    plt.savefig(out_filename, dpi=200, bbox_inches='tight') #save figure as ward_clusters
    # plt.show()
    # plt.close()


#################################################
#############  MAIN  ############################
#################################################
keywords = ['cases of covid19', 'corona', 'coronavirus', 'coronavirus cases', 'coronavirus covid19',
            'coronavirus news', 'coronavirus symptoms', 'coronavirus update', 'covid', 'covid 19',
            'covid 19 cases', 'covid19', 'covid19 cases'
            ]

# keyword = 'covid19 cases'
for i in range(len(keywords)):
    keyword = keywords[i]
    # keyword = 'cases of covid19'
    # keyword = 'covid19 cases'
    is_last = i==(len(keywords)-1)
    # is_last = False
    plot_graph(keyword, is_last=is_last)
    # break
