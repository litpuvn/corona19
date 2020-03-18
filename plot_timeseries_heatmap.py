from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
import os
from pandas import concat

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = 'data/gtrends.csv'
gtrends = read_csv(os.path.join(dir_path, filename), header=0, index_col=0, parse_dates=True)
gtrends = gtrends[37:]


# gtrends = gtrends.T
pyplot.matshow(gtrends, interpolation=None, aspect='auto')
heatmap = pyplot.pcolor(gtrends)
pyplot.colorbar(heatmap)
pyplot.show()

# one_year = series['1990']
# groups = one_year.groupby(Grouper(freq='M'))
# months = concat([DataFrame(x[1].values) for x in groups], axis=1)
# months = DataFrame(months)
# months.columns = range(1,13)
# pyplot.matshow(months, interpolation=None, aspect='auto')
# pyplot.show()