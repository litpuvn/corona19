import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import tweepy as tw
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import networkx as nx

# Create list of lists containing bigrams in tweets

# keywords = ['corona', 'cases of covid19', 'coronavirus cases', 'coronavirus covid19']
# keywords = ['covid19']
# keywords = ['coronavirus']
keywords = ['covid19 cases']
folder = 'data/daily-complete'
number_of_pairs = 50


queries = []
for kw in keywords:
    filepath = os.path.join(folder, kw.replace(' ', '-', 5), '2020-01-20-2020-03-23-top-related-queries.csv')
    if not os.path.exists(filepath):
        print('bad file path')
        continue
    df = pd.read_csv(filepath, sep=',', header=0, index_col=0, parse_dates=False)
    for index, row in df.iterrows():
        q = row['query']
        repeat = int(row['value'])
        q_data = q.split()
        for i in range(repeat):
            queries.append(q_data)
        # print(index, row)

terms_bigram = [list(bigrams(query)) for query in queries]
# Flatten list of bigrams in clean tweets
bigrams_terms = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams_terms)
print("size:", len(bigrams_terms))
print(bigram_counts.most_common(20))



bigram_df = pd.DataFrame(bigram_counts.most_common(number_of_pairs),
                             columns=['bigram', 'count'])

# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')
# Create network plot
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v/10))

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]


# G.add_node("china", weight=100)
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, k=1)
# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=weights,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0] + .135, value[1] + .045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)

plt.show()