#Figure 5: Overall terms in related queries.
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from pandas.core.frame import DataFrame
from wordcloud import WordCloud, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from datetime import timedelta, date
import pandas as pd
import os

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0, 16.0),
                   title=None, title_size=40, image_color=False):

    global stopwords
    stopwords2 = set({})
    more_stopwords = {'covid19', 'corona'}
    stopwords3 = stopwords2.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords3,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          mask=mask)
    # wordcloud.generate(text)
    wordcloud.generate_from_frequencies(text, max_font_size=max_font_size)

    if title is None:
        title = ''
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'green',
                                   'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()

    plt.show()


def extract_top_words_for_keyword(keyword):
    start_year = 2020
    start_mon = 1
    start_day = 21

    start_date = date(start_year, start_mon, start_day)
    day_count = 61

    folder = os.path.join('data/daily-complete', keyword.replace(' ', '-', 5))

    daily_frames = []
    for i in range(day_count):

        day_string = start_date.strftime('%Y-%m-%d')
        start_date = start_date + timedelta(1)

        filename = os.path.join(folder, day_string + '-top-related-queries.csv')
        if not os.path.isfile(filename):
            print('File', filename, 'does not exist')
            continue

        df = pd.read_csv(filename, sep=',', header=0, index_col=0, parse_dates=True)
        if len(df) < 1:
            continue

        daily_frames = daily_frames + [df]

    return pd.concat(daily_frames, axis=0)


keywords = ['covid19', 'corona']

sum_of_words = 0

frames = []
for kw in keywords:

    keyword_df = extract_top_words_for_keyword(kw)
    frames = frames + [keyword_df]
    # print("Keyword:", kw, "has", len(words), 'words')
    # sum_of_words = sum_of_words + len(words)
    # text = text + ' ' + ' '.join(words)

all_words = pd.concat(frames, axis=0)

# grouped_words = all_words.groupby('query')['value'].apply(sum_values).to_frame()
grouped_words = all_words.groupby('query').sum()
grouped_words.reset_index(level=0, inplace=True)

min_frequency = grouped_words['value'].min()
print(grouped_words.head())

text_dict = dict()
stop_words = {'corona', 'covid19'}
for index, row in grouped_words.iterrows():
    query_words = row['query'].split()
    for w in query_words:
        if w in stop_words:
            continue
        if w not in text_dict:
            text_dict[w] = 1
        else:
            text_dict[w] += 1



# text = ' '.join(grouped_words['query'].tolist())

print(text_dict)

comments_mask = np.array(Image.open('data/comment.png'))
comments_mask = None
plot_wordcloud(text_dict, comments_mask,
               max_words=1400,
               max_font_size=120,
               figure_size=(8, 6)
               )

