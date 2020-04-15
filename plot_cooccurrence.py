import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

# import tweepy as tw
import nltk
from nltk import bigrams
from nltk.corpus import stopwords
import re
import networkx as nx

# Create list of lists containing bigrams in tweets

# keywords = ['corona', 'cases of covid19', 'coronavirus cases', 'coronavirus covid19']

#used in Figure: Categories of related topic
# keywords = ['covid19']
keywords = ['coronavirus']
# keywords = ['covid19 cases']
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
# fig, ax = plt.subplots(figsize=(10, 8))
fig, ax = plt.subplots(figsize=(8, 7))

pos = nx.spring_layout(G, k=1)
# Plot networks
nx.draw_networkx(G, pos,
                #  font_size=16,
                 font_size=10,
                 width=weights,
                #  edge_color='grey',
                 edge_color='gray',
                 node_color='purple',#'purple',#'mediumpurple',#'plum',#'purple',
                 with_labels=False,
                 #node_shape='h', #4/14: one of ‘so^>v<dph8’ (default=’o’)
                 ax=ax)

# Create offset labels
x_arr = []
y_arr = []
for key, value in pos.items():
    # x, y = value[0] + .135, value[1] + .045
    x_delta = 0
    y_delta = .06
    x = value[0]
    y = value[1]
    if x > 0:
        x_delta = 0.01
    if y < 0:
        y_delta = -.135
    x, y = x + x_delta, y + y_delta
    x_arr.append(x)
    y_arr.append(y)
    ax.text(x, y,
            s=key,
            # bbox=dict(facecolor='red', alpha=0.25),
            bbox=dict(facecolor='black', alpha=0.01),
            horizontalalignment='center', fontsize=20)

print('x:', x_arr)
print('y:', y_arr)
#remove frame
ax.axis("off")

#save fig
out_path = 'figures/'
out_filename = out_path + 'cooccurrence ' + keywords[0] + '.pdf'
out_filename = out_filename.replace(" ", "-")
print('The figure save into:', out_filename)
plt.savefig(out_filename, dpi=200) #save figure as ward_clusters
# plt.show()