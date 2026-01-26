import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy.optimize import fixed_point

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
beta = iv_model.params['x']
print(alpha)
'''
0.46747709693613215
'''
print(beta)
'''
0.3046723858040141
'''

# Use estimated parameters to compute xi
df['xi'] = df['y'].values + alpha * df['Prices'].values - beta * df['x'].values
 
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
'''
[[-1.24962402  0.32300246  0.12873807  0.12715379  0.12551045  0.12946544]
 [ 0.321076   -1.25124251  0.12873807  0.12715379  0.12551045  0.12946544]
 [ 0.321076    0.32300246 -1.28914608  0.12715379  0.12551045  0.12946544]
 [ 0.321076    0.32300246  0.12873807 -1.2938695   0.12551045  0.12946544]
 [ 0.321076    0.32300246  0.12873807  0.12715379 -1.29142552  0.12946544]
 [ 0.321076    0.32300246  0.12873807  0.12715379  0.12551045 -1.29079878]]
'''

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
'''
[0.6671259  0.67189737 0.67868414 0.6889061  0.68263232 0.68268899]
'''

# Show average price of products
avg_prices = reg_df.groupby('Product')['Prices'].mean()
print(avg_prices)
'''
Product
1    3.359951
2    3.367534
3    3.033056
4    3.039771
5    3.031028
6    3.038147
'''

# Code for Part 4,5 #--

# Keep only X's,Xi's, MCs and Relevant products
df2 = df[df['Product'] != 0].copy()
df2['mc'] = np.asarray(mc_list, dtype=float).reshape(-1)
df2 = df2[df2['Product']!= 1].copy()
df2 = df2.loc[:,['Prices','market', 'Product','x','mc','Shares', 'xi']]

# Define Product Market Share Function in order to compute fixed point
def share_update(s_guess, df, eps=1e-8):
    s = np.asarray(s_guess, dtype=float)
    s = np.clip(s, eps, 1 - eps)  # keep in (0,1)
    price = df['mc'].values + (1.0 / (alpha * (1.0 - s)))

    v = (-alpha * price) + (beta * df['x'].values) + df['xi'].values
    exp_v = np.exp(v)

    denom = 1.0 + df.assign(exp_v=exp_v).groupby('market')['exp_v'].transform('sum').values
    return exp_v / denom


# compute new shares, prices using fixed point algorithm
df2['New_Shares']= fixed_point(share_update, df2['Shares'].values, args=(df2,), xtol=1e-10, maxiter=1000)
df2['New_Prices'] = df2['mc'].values + (1.0 / (alpha * (1.0 - df2['New_Shares'].values)))


new_p_shares = (
    df2.groupby("Product")
      .agg(
          avg_new_price=('New_Prices', 'mean'),
          avg_old_price=('Prices', 'mean'),
          avg_new_share=('New_Shares', 'mean'),
          avg_old_share=('Shares', 'mean'),
      )
)
print(new_p_shares)
'''
         avg_new_price  avg_old_price  avg_new_share  avg_old_share
Product                                                            
2             3.517996       3.367534       0.244264       0.203076
3             3.095221       3.033056       0.113266       0.090349
4             3.100464       3.039771       0.111395       0.088941
5             3.091850       3.031028       0.110741       0.088172
6             3.100436       3.038147       0.113597       0.090688


'''

# Compute Change in Firm Profits
df2['delta_profit'] = (df2['New_Shares'].values*(df2['New_Prices'].values-df2['mc'].values))-(df2['Shares'].values*(df2['Prices'].values-df2['mc'].values))

delta_profit = df2.groupby("Product")['delta_profit'].mean()

print(delta_profit)
'''
Product
2    0.150462
3    0.062165
4    0.060693
5    0.060823
6    0.062289
'''

# Compute change in welfare
## calculate new outside shares
df2['oo_new'] = 1 - df2.groupby('market')['New_Shares'].transform('sum')
new_share = df2.groupby('market')['oo_new'].first()
old_share = df.groupby('market')['oo'].first()

delta_welfare = np.mean(np.log((1-new_share)/new_share)-np.log((1-old_share)/old_share))
print(delta_welfare)
'''
-0.358627133111499
'''

########################################################################