import pandas as pd
import  os
from os.path import isfile, join
from datetime import date, timedelta
import os
import time
import seaborn as sns
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data'

keywords = ['case of covid19', 'corona', 'coronavirus', 'coronavirus cases', 'coronavirus covid19',
            'coronavirus news', 'coronavirus symptoms', 'coronavirus update', 'covid', 'covid 19',
            'covid 19 cases', 'covid19', 'covid19 cases'
            ]


day_count = 63

start_year = 2020
start_mon = 1
start_day = 20

all_frames = []
topic_types = {}
for k in keywords:
    start_date = date(start_year, start_mon, start_day)
    current_dir_path = os.path.join(dir_path, 'daily-complete', k.replace(" ", '-', 5))
    for i in range(day_count):
        filepath = os.path.join(current_dir_path,  start_date.strftime('%Y-%m-%d') + '-top-related-topic.csv')
        if not os.path.exists(filepath):
            print('file', filepath, 'does not exist')
            continue
        # print("--- data day:", start_date.strftime('%Y-%m-%d'), "; counter i=", i)
        df = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=False)
        if len(df) < 1:
            continue

        for index, row in df.iterrows():
            topic_type = row['topic_type']
            if topic_type not in topic_types:
                topic_types[topic_type] = 1
            else:
                topic_types[topic_type] += 1
        # all_frames = all_frames + [df]
        start_date = start_date + timedelta(1)


sorted_types = {k: v for k, v in sorted(topic_types.items(), key=lambda item: item[1], reverse=True)}
# print(sorted_types)

topics = []
counter = 0
# ignore_topics = ['Topic', 'People', 'Broadcast genre']
ignore_topics = ['Topic']
for k, v in sorted_types.items():
    if k in ignore_topics:
        continue
    if v > 10:
        topics.append(k)
        print('topic:', k, "; val:", v)
        counter += 1

start_date = date(start_year, start_mon, start_day)
days = []
for i in range(day_count):
    days = days + [start_date.strftime('%Y-%m-%d')]
    start_date = start_date + timedelta(1)

df = DataFrame(index=topics, columns=days)
df.fillna(0, inplace=True)
for k in keywords:
    start_date = date(start_year, start_mon, start_day)
    current_dir_path = os.path.join(dir_path, 'daily-complete', k.replace(" ", '-', 5))
    for i in range(day_count):
        filepath = os.path.join(current_dir_path,  start_date.strftime('%Y-%m-%d') + '-top-related-topic.csv')
        if not os.path.exists(filepath):
            print('file', filepath, 'does not exist')
            continue
        # print("--- data day:", start_date.strftime('%Y-%m-%d'), "; counter i=", i)
        df2 = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=False)
        if len(df) < 1:
            continue

        for index, row in df2.iterrows():
            topic_type = row['topic_type']
            value = row['value']
            if topic_type in topics:
                r = topic_type
                c = start_date.strftime('%Y-%m-%d')
                df.at[r, c] = value
                t = row['topic_title']
                if topic_type in ['Website']:
                    print('type:', r, '; title: ', t)
        # all_frames = all_frames + [df]
        start_date = start_date + timedelta(1)


print("topic types:", len(topics))
print("days:", len(days))
ax = sns.heatmap(df, yticklabels=True, linewidths=0, cmap="YlGnBu")
# ax.figure.tight_layout()
ax.figure.subplots_adjust(left = 0.3) # change 0.3 to suit your needs.

plt.show()


# print("total topic types:", counter)
        # df = pd.concat(all_frames, axis=0, sort=False)
# print(df.shape)
#
# #//df.groupby(by=['class_energy'])['ACT_TIME_AERATEUR_1_F1', 'ACT_TIME_AERATEUR_1_F3','ACT_TIME_AERATEUR_1_F5'].sum()
# grouped_types = df.groupby('topic_type').count()
# grouped_types.reset_index(level=0, inplace=True)