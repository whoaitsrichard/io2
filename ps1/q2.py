import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

df = pd.read_csv('data/ps1_ex2.csv')

# Create outside option. For each market, share of outside option is 1 - sum of inside shares
market_share_sums = df.groupby('market')['Shares'].transform('sum')
df['oo'] = 1 - market_share_sums
# Create unique market to outside option
oo_df = df[['market']].drop_duplicates().copy()
oo_df['Product'] = 0
oo_df['Shares'] = df.groupby('market')['oo'].first().values
oo_df['Prices'] = 0
oo_df['x'] = 0
oo_df['z'] = 0

# Insert outside option rows into the original dataframe
df = pd.concat([df, oo_df], ignore_index=True)
df = df.sort_values(by=['market', 'Product']).reset_index(drop=True)

# Sum shares by market to verify they sum to 1
share_sums = df.groupby('market')['Shares'].sum()
print(share_sums.describe())

# Just need price parameter to estimate own and cross price elasticities.
# Regression equation ln(s_jt/s_0t) = -alpha * p_jt + beta * x_jt
df['y'] = np.log(df['Shares'] / df['oo'])

reg_df = df.copy()
# Drop outside option rows for regression
reg_df = reg_df[reg_df['Product'] != 0]

# Use z as an instrument for Prices and run IV regression
controls = ['x']

# Regressors: const + endogenous + controls
exog = sm.add_constant(reg_df[['Prices'] + controls])

# Instruments: const + excluded instrument(s) + controls
instr = sm.add_constant(reg_df[['z'] + controls])
iv_model = IV2SLS(reg_df['y'], exog, instr).fit()
print(iv_model.summary())

alpha = -iv_model.params['Prices']

# Calculate own and cross price elasticities
# epsilon_jj = -alpha * p_j * (1 - s_j)
# epsilon_jk = alpha * p_k * s_k
mkt_elasticities = []
for mkt in reg_df['market'].unique():
    mkt_data = reg_df[reg_df['market'] == mkt]
    prices = mkt_data['Prices'].values
    shares = mkt_data['Shares'].values
    n_products = len(mkt_data)
    elasticity_matrix = np.zeros((n_products, n_products))
    for j in range(n_products):
        for k in range(n_products):
            if j == k:
                elasticity_matrix[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                elasticity_matrix[j, k] = alpha * prices[k] * shares[k]
    mkt_elasticities.append(elasticity_matrix.flatten())

# Average and reshape
avg_e_matrix = np.mean([el for el in mkt_elasticities], axis=0).reshape((6,6))
print(avg_e_matrix)

# Calculate marginal cost for each product in each market
mc_list = []
for mkt in reg_df['market'].unique():
    mkt_data = reg_df[reg_df['market'] == mkt]
    p = mkt_data['Prices']
    e_matrix = mkt_elasticities[mkt-1].reshape((6,6))
    e_jj = np.diag(e_matrix)
    mc = p*(1+1/e_jj)
    mc_list.append(mc)

avg_mc = np.mean([el for el in mc_list], axis=0)
print(avg_mc)

# Show average price of products
avg_prices = reg_df.groupby('Product')['Prices'].mean()
print(avg_prices)

# No it cannot be differences in marginal costs. The 

# Now let's do a counterfactual exercise where product j exits the market. This is the same as 
# setting the price of product j equal to infinity and recalculating the new optimal prices and market shares of remaining products
