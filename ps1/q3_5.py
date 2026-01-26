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
'''
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
Intercept     -1.0469     0.6550    -1.5983     0.1100     -2.3306      0.2369
x              1.1905     0.1662     7.1629     0.0000      0.8647      1.5163
ln_s_j_g       0.3435     0.0922     3.7254     0.0002      0.1628      0.5242
==============================================================================
'''

first_stage = smf.ols(formula='ln_s_j_g~ 1 + num_prod', data=df).fit()
print(first_stage.summary())
'''
  OLS Regression Results                            
==============================================================================
Dep. Variable:               ln_s_j_g   R-squared:                       0.096
Model:                            OLS   Adj. R-squared:                  0.083
Method:                 Least Squares   F-statistic:                     7.527
Date:                Mon, 26 Jan 2026   Prob (F-statistic):            0.00769
Time:                        15:46:15   Log-Likelihood:                -203.41
No. Observations:                  73   AIC:                             410.8
Df Residuals:                      71   BIC:                             415.4
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.2709      2.205     -0.123      0.903      -4.668       4.126
num_prod      -0.7560      0.276     -2.744      0.008      -1.305      -0.207
==============================================================================
'''

# No IV OLS
results_noiv = smf.ols(formula='ln_s_j_ln_s_0 ~ 1 + x + ln_s_j_g', data=df).fit()
print(results_noiv.summary())
'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      1.4496      0.274      5.292      0.000       0.903       1.996
x              0.5871      0.083      7.087      0.000       0.422       0.752
ln_s_j_g       0.7044      0.037     19.253      0.000       0.631       0.777
==============================================================================
'''

