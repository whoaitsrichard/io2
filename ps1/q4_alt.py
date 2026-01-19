'''
Things that could cause issues in the future
1. We divide by 600 twice in the gmm obj

'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy.optimize import minimize

df = pd.read_csv('data/ps1_ex4.csv')

num_i = 10000 # 1000 simulated consumers
outer_tol = 1e-6
inner_tol = 1e-7

# normal distribution. Keep these fixed for all iterations. Although maybe this should go in the outer loop structure?
v_i = np.random.normal(0.0, 1.0, size=(num_i, 2))

# store instruments, characteristics, obs. shares
z_inst = np.array(df[['z1','z2','z3','z4','z5','z6']].values)
z_inst_t = z_inst.T
obs_shares = np.array(df.shares.values)
x_jt = df[['p','x']].values

# initial W for GMM
W = np.linalg.inv(z_inst_t @ z_inst)

# Compute initial guess of beta using hom. logit
#outside_market_share = 1-df.groupby('market')['shares'].transform('sum')
#ln_st_s0= obs_shares/outside_market_share
#logit_df = df.copy()
#logit_df['ln_st_s0'] = ln_st_s0
#controls = ['x']
#exog = sm.add_constant(logit_df[['p'] + controls])
#instr = sm.add_constant(logit_df[['z1', 'z2','z3','z4','z5','z6'] + controls])
#iv_model = IV2SLS(logit_df['ln_st_s0'], exog, instr).fit()

'''
Calculates the GMM objective given a list of parameters
'''
def calc_gmm_obj(temp_xi, temp_W):
    sample_moment = (1/600) * np.sum(z_inst * temp_xi.reshape(-1,1),axis=0) # This should be sample analog to E[xi_{jt} * z_{jt}]. Shape should be 6x1
    gmm_temp = sample_moment.T @ temp_W @ sample_moment
    return gmm_temp

'''
Calculates the predicted market shares
'''
def calc_s_pred(temp_delta, temp_gamma):
    temp_gamma = np.insert(temp_gamma,1,0)
    temp_gamma = np.reshape(temp_gamma,(2,2))

    # temp_delta is 600x1, 
    temp_delta_proj = np.broadcast_to(temp_delta, (num_i,) + temp_delta.shape)
    x_jt_proj = np.broadcast_to(x_jt, (num_i,) + x_jt.shape)

    w = (temp_gamma @ v_i.T).T          # (num_i, 2)
    numerator = np.einsum('ijk,ik->ij', x_jt_proj, w)   # (num_i, 600)
    numerator = temp_delta_proj + numerator
    numerator = numerator.reshape(numerator.shape[0],100,6)
    numerator = np.exp(numerator)
    denom = numerator.sum(axis=2) + 1 # (10000, 100)

    ratio = numerator / denom[..., None]  # denom -> (10000,100,1), broadcasts to (10000,100,6)

    s_jk = ratio.mean(axis=0)
    s_jk = s_jk.reshape(600,)
    return s_jk

##################
# INIT VARIABLES #
##################

# Start with initial guesses of the non-linear parameters
gamma = np.array([0.1, 0.1, 0.1]) #gamma_{11}, gamma_{21}, gamma_{22} respectively
#beta = [iv_model.params['p'],iv_model.params['x']] # price coefficient and x coefficient respectively
delta = np.random.rand(600)


'''
'''
def calc_opt_W(temp_gamma, delta0):
    delta=delta0.copy()
    cm_obj = 1
    # Contraction mapping for delta - the mean utility levels for products
    while cm_obj > inner_tol:
        s_pred = calc_s_pred(delta, temp_gamma)
        delta_k1 = delta + (np.log(obs_shares) - np.log(s_pred))
        cm_obj = np.sum(np.abs(delta_k1 - delta))
        delta = delta_k1

    # Now we have the "correct" delta. 
    reg_df = df.copy()
    reg_df['y'] = delta
    # Run 2SLS to estimate beta
    # Use z as an instrument for Prices and run IV regression
    controls = ['x']
    # Regressors: const + endogenous + controls
    exog = sm.add_constant(reg_df[['p'] + controls])

    # Instruments: const + excluded instrument(s) + controls
    instr = sm.add_constant(reg_df[['z1','z2','z3','z4','z5','z6'] + controls])
    iv_model = IV2SLS(reg_df['y'], exog, instr).fit()
    beta = iv_model.params.values[1:]

    # Estimate xi, unobserved product heterogeneity
    # xi = delta - np.mult(reg_df[['p','x']].values, beta)
    xi = delta - reg_df[['p','x']].values @ beta

    # Calculate moment res which is something with the instruments and the 
    gmm_obj = calc_gmm_obj(xi, W)
    print(gmm_obj)
    return gmm_obj



# Runs one loop of BLP
def blp_obj(temp_gamma):

    cm_obj = 1
    # Contraction mapping for delta - the mean utility levels for products
    while cm_obj > inner_tol:
        s_pred = calc_s_pred(delta, temp_gamma)
        delta_k1 = delta + (np.log(obs_shares) - np.log(s_pred))
        cm_obj = np.sum(np.abs(delta_k1 - delta))
        delta = delta_k1

    # Now we have the "correct" delta. 
    reg_df = df.copy()
    reg_df['y'] = delta
    # Run 2SLS to estimate beta
    # Use z as an instrument for Prices and run IV regression
    controls = ['x']

    # Regressors: const + endogenous + controls
    exog = sm.add_constant(reg_df[['p'] + controls])

    # Instruments: const + excluded instrument(s) + controls
    instr = sm.add_constant(reg_df[['z1','z2','z3','z4','z5','z6'] + controls])
    iv_model = IV2SLS(reg_df['y'], exog, instr).fit()
    beta = iv_model.params.values

    # Estimate xi, unobserved product heterogeneity
    xi = delta - np.mult(reg_df[['p','x']].values, beta)

    # Calculate moment res which is something with the instruments and the 
    gmm_obj = calc_gmm_obj(xi, W_opt)

    return gmm_obj



# Calculate init optimal IV matrix
# Use Nelder-Mead

res = minimize(
    calc_opt_W,
    x0=[0.1,0.1,0.1],
    args=(delta,),                 # or args=(data1, data2, ...)
    method="L-BFGS-B",
    options={
        "maxiter": 10_000,
        "xatol": 1e-6,        # tolerance on x
        "fatol": 1e-6,        # tolerance on f
        "disp": True
    }
)





temp_W = opt_iv_mx_init
temp_xi = np.random.rand(600)
sample_moment = (1/600) * np.sum(z_inst * temp_xi.reshape(-1,1),axis=0) # This should be sample analog to E[xi_{jt} * z_{jt}]. Shape should be 6x1