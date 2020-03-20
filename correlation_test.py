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

filepath = 'data/gtrends.csv'
df_gtrends = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=True)
df_gtrends = df_gtrends[37:]
df_gtrends['Worldwide'] = df_gtrends.sum(axis=1)

df_new_cases = pd.read_csv('data/new_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
df_new_cases = df_new_cases[:-1]
df_new_cases.fillna(value=0, inplace=True)

# countries = ['Italy', 'Spain', 'France', 'Germany']
# countries = ['United Kingdom', 'South Korea', 'United States', 'China', 'Worldwide']
# selected_index = 4
# selected_country = countries[selected_index]
#
# selected_country_cases_df = DataFrame(df_new_cases[selected_country].values, index=df_new_cases.index)
# selected_country_gtrends_df = DataFrame(df_gtrends[selected_country].values, index=df_gtrends.index)

countries = df_gtrends.columns
for country in countries:
    if country not in df_new_cases.columns:
        continue
    corr, pval = pearsonr(df_new_cases[country].values, df_gtrends[country].values)
    print('correlation: new case vs gtrends', country,"; corr:", corr, "p-val:", pval)