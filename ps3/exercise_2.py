'''
Exercise 2: Ebay Auction Empirical Exercise
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats 
from scipy.stats import norm, truncnorm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import math

'''
Part 2: Summary Statistics
'''
# Read in Data
bids = pd.read_csv('data/bids.csv')
items = pd.read_csv('data/items.csv')
attributes = pd.read_csv('data/sparse_attributes.csv')

# Reindex the item_nums because seems like something was dropped in a prefiltering step
items['new_item_num'] = range(1, len(items) + 1)
# Merge the new item_num back to bids and attributes
bids = bids.merge(items[['item_num', 'new_item_num']], on='item_num', how='left')
bids['item_num'] = bids['new_item_num']
bids.drop(columns=['new_item_num'], inplace=True)


## Convert attributes to sparse matrix
wide_attributes = (attributes.pivot_table(index="i", columns="j", values="fill",
                     aggfunc="max", fill_value=0).sort_index())

wide_attributes = wide_attributes.reindex(range(1, 6984), fill_value=0)
bids['log_bid_value'] = np.log(bids['bid_value'])




# # Plot Histograms of Mean Log Bid and Num Bids by Item
# plt.hist(bids.groupby('item_num')['log_bid_value'].mean(),bins=50,color='skyblue',edgecolor='black')
# plt.xlabel('Log Bid')
# plt.ylabel('Frequency')
# plt.title('Mean of Log Bids by Item')
# plt.show()

# counts = bids.groupby("item_num")["item_num"].count()
# plt.hist(counts,bins=20,color='forestgreen',edgecolor='black')
# plt.xlabel('Number of Bids')
# plt.ylabel('Frequency')
# plt.title('Distribution of Number of Bids')
# plt.xticks(range(int(counts.min()), int(counts.max()) + 2, 2))  # step=1
# plt.show()
# plt.show()



# Create regression dataset
bids['ln_max_bid'] = bids.groupby('item_num')['log_bid_value'].transform('max')


regression_df = bids.copy(deep=True)
regression_df = regression_df.merge(items, on='item_num', how='left')
regression_df = regression_df.dropna()

wide_attributes = wide_attributes.reset_index().rename(columns={'i': 'item_num'})
regression_df = regression_df.merge(wide_attributes, on='item_num', how='left')

# Drop rows with na values in the regression dataset

# Run OLS regression on regression_df
reg_df = regression_df.drop(columns=['item_num','bid_value','log_bid_value','new_item_num'])

import statsmodels.api as sm
X = reg_df.drop(columns=['ln_max_bid'])
# Multiply each column in X by X.pred_n_participants
X = X.multiply(X['pred_n_participant'], axis=0)
X = X.drop(columns=['pred_n_participant'])
X = X.drop(columns=[1])
y = reg_df['ln_max_bid']
X = sm.add_constant(X)  # Add constant term for intercept

model = sm.OLS(y, X).fit()
print(model.summary())

# Predict the max log bid using the regression model
regression_df['predicted_ln_max_bid'] = model.predict(X)
# Calculate MSE
mse = np.mean((regression_df['ln_max_bid'] - regression_df['predicted_ln_max_bid']) ** 2)
print(f'Mean Squared Error: {mse}')

# Now try lasso regression. Use OLS.fit_regularized with L1 penalty
lasso_model = sm.OLS(y, X).fit_regularized(method='elastic_net', L1_wt=1.0, alpha=0.1)
# Show results (fit_regularized returns RegularizedResultsWrapper, which has no .summary())
print("Lasso non-zero coefficients:")
lasso_params = lasso_model.params
print(lasso_params[lasso_params != 0])

# predict the max log bid using the lasso regression model
regression_df['predicted_ln_max_bid_lasso'] = lasso_model.predict(X)
# Calculate MSE for lasso model
mse_lasso = np.mean((regression_df['ln_max_bid'] - regression_df['predicted_ln_max_bid_lasso']) ** 2)
print(f'Mean Squared Error for Lasso: {mse_lasso}')

# Neural network regression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Drop the constant column added by statsmodels (sklearn handles intercept internally)
X_nn = X.drop(columns=['const'])

# Standardize features for neural network
scaler = StandardScaler()
X_nn_scaled = scaler.fit_transform(X_nn)

nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
nn_model.fit(X_nn_scaled, y)

regression_df['predicted_ln_max_bid_nn'] = nn_model.predict(X_nn_scaled)
mse_nn = np.mean((regression_df['ln_max_bid'] - regression_df['predicted_ln_max_bid_nn']) ** 2)
print(f'Mean Squared Error for Neural Network: {mse_nn}')
