'''
Try Implementing PyBLP Package
'''

import pyblp
import numpy as np
import pandas as pd
pyblp.options.digits = 5
pyblp.options.verbose = False

# Read in product data
product_data = pd.read_csv('data/ps1_ex4.csv')
product_data = product_data.rename(columns={'p': 'prices', 
                                            'market': 'market_ids',
                                            'choice':'product_ids',
                                            'z1':'demand_instruments0',
                                            'z2':'demand_instruments1',
                                            'z3':'demand_instruments2',
                                            'z4':'demand_instruments3',
                                            'z5':'demand_instruments4',
                                            'z6':'demand_instruments5',
                                            'shares':'shares'})

product_data['firm_ids'] = product_data['product_ids']
# Set product formulations
X1_formulation = pyblp.Formulation('1 + prices + x')
X2_formulation = pyblp.Formulation(' 0 + prices + x')
product_formulations = (X1_formulation, X2_formulation)
product_formulations

# Set integretation configuration for market share estimation
mc_integration = pyblp.Integration('monte_carlo', size=1000, specification_options={'seed': 44259})
mc_integration

# Define problem to solve
mc_problem = pyblp.Problem(product_formulations, product_data, integration=mc_integration)
mc_problem

# configure optimization method
bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-6})
bfgs

# Test results 
results1 = mc_problem.solve(sigma=np.ones((2, 2)), optimization=bfgs)
results1

'''
Problem Results Summary:
============================================================================================================
GMM    Objective    Gradient        Hessian         Hessian     Clipped  Weighting Matrix  Covariance Matrix
Step     Value        Norm      Min Eigenvalue  Max Eigenvalue  Shares   Condition Number  Condition Number 
----  -----------  -----------  --------------  --------------  -------  ----------------  -----------------
 2    +2.1907E+00  +7.9034E-07   +2.5711E-01     +7.9322E+00       0       +4.9560E+01        +2.7874E+04   
============================================================================================================

Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
=====================================================================================
Sigma:     prices            x        |  Sigma Squared:     prices            x      
------  -------------  -------------  |  --------------  -------------  -------------
prices   +4.4445E+00                  |      prices       +1.9754E+01    +2.1955E+01 
        (+2.1477E+00)                 |                  (+1.9091E+01)  (+2.1899E+01)
                                      |                                              
  x      +4.9397E+00    -1.7683E+00   |        x          +2.1955E+01    +2.7527E+01 
        (+2.7486E+00)  (+2.0827E+00)  |                  (+2.1899E+01)  (+3.3076E+01)
=====================================================================================

Beta Estimates (Robust SEs in Parentheses):
===========================================
      1           prices            x      
-------------  -------------  -------------
 -2.7269E+00    -1.9494E+00    +8.4585E-01 
(+7.7456E-01)  (+1.0894E+00)  (+6.0217E-01)
===========================================
'''

'''
[-1.405113    0.88031527]
'''

'''
[[11.81344936 -0.42772071]
 [-0.42772071  0.01584006]]
'''
