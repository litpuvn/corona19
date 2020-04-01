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
from scipy.stats import pearsonr
import os

filepath = 'data/gtrends_coronavirus.csv'
df_gtrends = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
df_gtrends = df_gtrends[37:]
df_gtrends['Worldwide'] = df_gtrends.sum(axis=1)

df_new_cases = pd.read_csv('data/new_cases_april_01.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_cases = df_new_cases[21:-10]
df_new_cases.fillna(value=0, inplace=True)

# countries = ['Italy', 'Spain', 'France', 'Germany']
# countries = ['United Kingdom', 'South Korea', 'United States', 'China', 'Worldwide']
# selected_index = 4
# selected_country = countries[selected_index]
#

folder = 'data/daily-complete'

def get_gtrend_data_by_keyword(keyword):
    global folder
    file_folder = os.path.join(folder, keyword.replace(' ', '-', 5))
    gtrends_interest = pd.read_csv(os.path.join(file_folder, '2020-01-20-2020-03-23-interest.csv'), sep=',', header=0, index_col=0, parse_dates=True)
    # gtrends_related_queries = pd.read_csv(os.path.join(file_folder, '2020-01-20-2020-03-23-top-related-queries.csv'), sep=',', header=0, index_col=0, parse_dates=True)
    # gtrends_related_topics = pd.read_csv(os.path.join(file_folder, '2020-01-20-2020-03-23-top-related-topic.csv'), sep=',', header=0, index_col=0, parse_dates=True)

    # return gtrends_interest, gtrends_related_queries, gtrends_related_topics
    return gtrends_interest


countries = ['World']
keywords = ['cases of covid19', 'corona', 'coronavirus', 'coronavirus cases', 'coronavirus covid19',
            'coronavirus news', 'coronavirus symptoms', 'coronavirus update', 'covid', 'covid 19',
            'covid 19 cases', 'covid19', 'covid19 cases'
            ]

all_frames = []
for country in countries:
    if country not in df_new_cases.columns:
        continue
    country_new_cases_df = df_new_cases[country]
    frame = {'confirmed_cases': country_new_cases_df}
    target_frame = DataFrame(frame)
    target_frame.to_csv("data/target_confirmed_cases.csv", index=True)

    for kw in keywords:
        gtrends_df = get_gtrend_data_by_keyword(kw)
        remove_rows = 62-len(gtrends_df)
        if remove_rows < 0:
            gtrends_df = gtrends_df[:remove_rows]
        corr, pval = pearsonr(country_new_cases_df.values, gtrends_df[kw].values)

        frame = {kw.replace(' ', '_', 5): gtrends_df[kw]}
        all_frames = all_frames + [DataFrame(frame)]
        print("shape:")
        print('correlation:', kw, ':new case vs gtrends:', country,"; corr:", corr, "p-val:", pval)

frame = pd.concat(all_frames, axis=1)
frame.to_csv("data/features.csv", index=True)
