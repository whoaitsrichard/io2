import numpy as np
import pandas as pd

df = pd.read_csv('data/ps1_ex4.csv')

# u_{ijt} = x'_{jt} * beta + xi_{jt} + x_{jt} * gamma * nu_{i} + e_{ijt}
# x_jt is vector of observed product characteristics with price

num_i = 1000 # 1000 simulated consumers
outer_tol = 1e-6
# normal distribution. Keep these fixed for all iterations. Although maybe this should go in the outer loop structure?
v_dist = np.random.normal(0,1,num_i)


''''
Returns MSE of market share residuals
    obs_shares: observed market shares
    sim_shares: simulated market shares
'''
def mkt_share_res(obs_shares, sim_shares):
    return np.sum((obs_shares - sim_shares)**2)

'''
Gives the next value for the mean utility levels for the contraction mapping
Input:
    d_k: kth guess of mean utility levels
    obs_shares: observed market shares
    k_shares: simulated market shares given d_k
Output:
    d_k1: (k+1)th guess of mean utility levels
'''
def c_map(d_k, obs_shares, k_shares):
    d_k1 = d_k + np.log(obs_shares) - np.log(k_shares)
    return d_k1

'''
Calculate predicted market shares given various parameters


'''
def calc_shares(d, ):

    return 
# Start with pseudo code

'''
Regresses mean utility levels on characteristics with instruments to estimate linear parameters

'''
def linear_iv():

    return
    



# Start with initial guesses of the non-linear parameters
gamma_init = [0.1, 0.1, 0.1] #gamma_{11}, gamma_{21}, gamma_{22} respectively
beta_init = [0.5, 0.5] # price coefficient and x coefficient respectively

# Start with initial guesses, avg_delta, for the mean utility levels of the products

# While Moment condition xi_res is greater than some tolerance

moment_res = 1
while moment_res > outer_tol:
    
    
    
    
    
    
    
    


    # Calculate moment res which is something with the instruments and the 
    # moment_res = 



# Calculate avg_delta such that (predicted market shares - observed market shares) < some tolerance

# Regress avg_delta on characteristics with instruments to estimate linear parameters

# Calculate xi_delta 

# Go back to loop condition to see if moment condition is satisfied

