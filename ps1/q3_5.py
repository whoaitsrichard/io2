import numpy as np
import pandas as pd
import random as rd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

# Steps:
## Initialize dataframe with 10 nests and random number of products in each market 
## Draw x_{j}, \xi_{j} randomly, compute delta_{j} using \beta = 1
## compute exp(\rho^{-1}\delta_{j}), use to compute s_j, s_0 s_{j|G}
## Regress \ln(s_j)-\ln(s_0) on x_j, s_{j|G} instrumented w/ num. products
np.random.seed(2987)

# Create Simulated Data 
groups = np.array(list(range(1,11)))
num_prod = np.random.randint(3, 11, size=10)
group_prod = np.repeat(groups,num_prod)
df = pd.DataFrame(data=group_prod, columns=['group'])
df['product'] = range(1, len(df) + 1)
df['x'] = np.random.normal(0, 2, size=len(df))
df['xi'] = np.random.normal(0, 1, size=len(df))
df['delta'] = df['x'] + df['xi'] 
df['exp_xi'] = np.exp(2 * df['delta'].values) # set \rh0 = .5

# Compute s_j, s_{j|G}, 
##compute (sum exp)^\rho by group, ln(s_j,s_0)
df['group_exp'] = df.groupby('group')['exp_xi'].transform('sum')

df['s_j'] = df['exp_xi'].values*np.power(df['group_exp'],-.5)/(1+sum(np.power(df.groupby('group')['group_exp'].first(),.5)))

s_0 = 1-sum(df['s_j'].values)

df['ln_s_j_g'] = np.log(df['exp_xi'].values/df['group_exp'].values)

df['ln_s_j_ln_s_0'] = np.log(df['s_j'].values)- np.log(s_0)

df['num_prod'] = df.groupby('group').transform('size')

## Regress \ln(s_j)-\ln(s_0) on x_j, s_{j|G} instrumented w/ num. products
formula = 'ln_s_j_ln_s_0 ~ 1 + x + [ln_s_j_g ~ num_prod]'


# Fit the IV model
model = IV2SLS.from_formula(formula, df)
results_iv = model.fit()
print(results_iv)

# No IV OLS
results_noiv = smf.ols(formula='ln_s_j_ln_s_0 ~ 1 + x + ln_s_j_g', data=df).fit()
print(results_noiv.summary())