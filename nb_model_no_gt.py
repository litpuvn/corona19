import pandas as pd
from patsy import dmatrices, dmatrix
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from common.utils import mape
from common.utils import split_train_validation_test, flatten_test_predict, store_predict_points


#create a pandas DataFrame for the counts data set
target = pd.read_csv('data/target_confirmed_cases.csv', sep=',', header=0, index_col=0, parse_dates=True)
target['t+1'] = target['confirmed_cases'].shift(1 * -1, freq='d')
target.dropna(how='any', inplace=True)

X = pd.DataFrame(target['confirmed_cases'], index=target.index)
X.columns = ['corona']
y = pd.DataFrame(target['t+1'], index=target.index)
y.columns = ['confirmed_cases']


# X.index = y.index
Y_entire = y['confirmed_cases']

## remove temporal feature, to randomly select rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
X_test = X_test.sort_index()
y_test = y_test.sort_index()
X_train = X_train.sort_index()
y_train = y_train.sort_index()

# X_test = X.loc[X_test.index, :]

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

expr = "confirmed_cases ~ corona"
y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')


#Using the statsmodels GLM class, train the Poisson regression model on the training data set
poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

#print out the training summary
print(poisson_training_results.summary())

#print out the fitted rate vector
print(poisson_training_results.mu)

#Add the λ vector as a new column called 'BB_LAMBDA' to the Data Frame of the training data set
df_train['BB_LAMBDA'] = poisson_training_results.mu

#add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. This new column will store the values of the dependent variable of the OLS regression
df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['confirmed_cases'] - x['BB_LAMBDA'])**2 - x['confirmed_cases']) / x['BB_LAMBDA'], axis=1)

#use patsy to form the model specification for the OLSR
ols_expr = "AUX_OLS_DEP ~ BB_LAMBDA - 1"

#Configure and fit the OLSR model
aux_olsr_results = smf.ols(ols_expr, df_train).fit()

#Print the regression params
print(aux_olsr_results.params)

#train the NB2 model on the training data set
nb2_training_results = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()

#print the training summary
print(nb2_training_results.summary())

#make some predictions using our trained NB2 model
nb2_predictions = nb2_training_results.get_prediction(X_test)

#print out the predictions
predictions_summary_frame = nb2_predictions.summary_frame()
print(predictions_summary_frame)

#plot the predicted counts versus the actual counts for the test data
predicted_counts = predictions_summary_frame['mean']
actual_counts = y_test['confirmed_cases']

y_pred = predicted_counts

nb2_train_predictions = nb2_training_results.get_prediction(X_train)
nb2_train_predictions_summary_frame = nb2_train_predictions.summary_frame()

predicted_Y_train = nb2_train_predictions_summary_frame['mean']

predicted_Y_entire = pd.concat([predicted_Y_train, y_pred], axis=0)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mape_v = mape(y_pred.values.reshape(-1, 1), y_test.values.reshape(-1, 1))
print('mape:', mape_v)
r2 = metrics.r2_score(y_test, y_pred)
print("R2:",  r2)

store_predict_points(Y_entire, predicted_Y_entire, 'output/test_nb2_without_prediction_r2_' + str(r2) + '.csv')

# fig = plt.figure()
# fig.suptitle('Predicted versus actual bicyclist counts on the Brooklyn bridge')
# predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted counts')
# actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
# plt.legend(handles=[predicted, actual])
# plt.show()
