'''
Things that could cause issues in the future

'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from scipy.optimize import minimize
from scipy.optimize import basinhopping


df = pd.read_csv('data/ps1_ex4.csv')

num_i = 200 # 100 simulated consumers
outer_tol = 1e-6
inner_tol = 1e-9


# normal distribution. Keep these fixed for all iterations. Although maybe this should go in the outer loop structure?
np.random.seed(44259)
v_i = np.random.normal(0.0, 1.0, size=(num_i, 2))

# store instruments, characteristics, obs. shares
z_inst = np.array(df[['z1','z2','z3','z4','z5','z6']].values)
z_inst_t = z_inst.T
obs_shares = np.array(df.shares.values)
x_jt = df[['p','x']].values
p_over_s_jt = df['p'].values/df['shares'].values

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
    denom = numerator.sum(axis=2) + 1 # (num_i, 100)

    ratio = numerator / denom[..., None]  # denom -> (num_i,100,1), broadcasts to (num_i,100,6) 

    s_jk = ratio.mean(axis=0)
    s_jk = s_jk.reshape(600,)
    return s_jk

'''
Calculates own-price elasticities using numerical integretation
'''
# def calc_own_price(mu, var,N):
#     np.random.seed(44259)
#     alpha_draws = np.random.normal(mu, np.sqrt(var), size=(N, 1))
#     integrand = alpha_draws * obs_shares.T * (1-obs_shares.T) 
#     own_price_est = integrand.mean(axis=0) * p_over_s_jt
#     return own_price_est

'''
Calculates own price elasticity using numerical integration
Inputs: 
    lin_p - linear parameters
    nlin_p - nonlinear parameters
    N - number of consumers to simulate
Outputs:
    length 6 vector of own price elasticities averaged over all markets
'''
def calc_jj_elas(lin_p, nlin_p, N):
    np.random.seed(44259)
    temp_df = df[['p','x']].copy()
    # Add in unobserved product characteristic
    temp_df['xi'] = xi_global
    temp_arr = temp_df.to_numpy()
    consumer_draws = np.random.normal(0, 1, size=(N, 2))

    # Calculate utility of each good for each consumer
    # temp_arr has shape (600, 3). We want an array of shape (N, 600, 5).
    # Expand the base product array across consumers: (N, 600, 3)
    base_expanded = np.broadcast_to(temp_arr, (N,) + temp_arr.shape)

    # consumer_draws is (N, 2). Repeat each consumer's draws across 600 products -> (N, 600, 2)
    draw_cols = np.repeat(consumer_draws[:, None, :], temp_arr.shape[0], axis=1)

# Concatenate along last axis to get (N, 600, 5)
    temp_proj = np.concatenate([base_expanded, draw_cols], axis=2)

    # Compute utility as:
    # utility = (price,x)·lin_p + xi + (price,x)·(nlin_p @ draws)
    # Ensure nlin_p is a (2,2) matrix and lin_p is length-2
    price_x = temp_proj[:, :, 0:2]  # (N, 600, 2)
    xi_vals = temp_proj[:, :, 2]    # (N, 600)
    draws = temp_proj[:, :, 3:5]    # (N, 600, 2)

    # Linear part: (price,x) · lin_p -> (N,600)
    # sum over the last axis (price,x), so use 'j' for that axis
    lin_part = np.einsum('nkj,j->nk', price_x, lin_p)

    # Nonlinear interaction: first compute (nlin_p @ draws) -> (N,600,2)
    nlin_times_draws = np.einsum('ab,njb->nja', nlin_p, draws)
    # Then dot with (price,x): (price,x) · (nlin_p @ draws) -> (N,600)
    interaction = np.einsum('nkj,nkj->nk', price_x, nlin_times_draws)
    utility = lin_part + xi_vals + interaction  # (N,600)

    # Now calculate probability of each good for each consumer
    # First calculate exp(utility)
    exp_utility = np.exp(utility)  # (N,600)

    # Reshape into (N, 100, 6) grouping products by market
    exp_u_by_market = exp_utility.reshape((N, 100, 6))

    # Denominator per (consumer, market) including outside option
    denom = exp_u_by_market.sum(axis=2) + 1  # (N,100)

    # Individual choice probabilities: (N,100,6)
    probs = exp_u_by_market / denom[:, :, None]

    # Consumer-specific price coefficients
    w = (nlin_p @ consumer_draws.T).T  # (N,2)
    alpha_i = lin_p[0] + w[:, 0]      # (N,)

    # Now calculate own price elasticities, which is s_ij * (1-s_ij)
    integrand = probs * (1 - probs)  # (N,100,6)
    # Drop the negative sign because sign convention is slightly different from the slides
    # We keep the alpha coefficient as negative.
    integrand = integrand * p_over_s_jt.reshape((1, 100, 6))  # Broadcasting p_over_s_jt
    integrand = alpha_i[:, None, None] * integrand  # Multiply by price coefficient (broadcast over markets/products)
    elas = integrand.mean(axis=0)  # Average over consumers -> (100,6)
    # Now average over markets
    elas = elas.mean(axis=0)  # (6,)
    return elas
'''
Calculates cross-price elasticities using numerical integration
'''
def calc_jk_elas(lin_p, nlin_p, N=1000):
    """
    Cross-price elasticity matrix (6x6) averaged across markets.
    Follows the same conventions as `calc_jj_elas`.
    """
    lin_p = l_input
    nlin_p = nl_input
    np.random.seed(44259)

    # Build product array and consumer draws
    temp_df = df[['p', 'x']].copy()
    temp_df['xi'] = xi_global
    temp_arr = temp_df.to_numpy()  # (600,3)
    consumer_draws = np.random.normal(0, 1, size=(N, 2))  # (N,2)

    # Expand to (N,600,5)
    base_expanded = np.broadcast_to(temp_arr, (N,) + temp_arr.shape)
    draw_cols = np.repeat(consumer_draws[:, None, :], temp_arr.shape[0], axis=1)
    temp_proj = np.concatenate([base_expanded, draw_cols], axis=2)

    # Compute utilities
    price_x = temp_proj[:, :, 0:2]
    xi_vals = temp_proj[:, :, 2]
    draws = temp_proj[:, :, 3:5]

    lin_part = np.einsum('nkj,j->nk', price_x, lin_p)
    nlin_times_draws = np.einsum('ab,njb->nja', nlin_p, draws)
    interaction = np.einsum('nkj,nkj->nk', price_x, nlin_times_draws)
    utility = lin_part + xi_vals + interaction  # (N,600)

    # Individual probabilities (N,100,6)
    exp_u_by_market = np.exp(utility).reshape((N, 100, 6))
    denom = exp_u_by_market.sum(axis=2) + 1
    probs = exp_u_by_market / denom[:, :, None]

    # Consumer-specific price coefficients
    w = (nlin_p @ consumer_draws.T).T  # (N,2)
    alpha_i = lin_p[0] + w[:, 0]      # (N,)

    # E[alpha * s_j * s_k] and E[alpha * s_j]
    s_jk = probs[:, :, :, None] * probs[:, :, None, :]  # (N,100,6,6)
    E_alpha_sjsk = (alpha_i[:, None, None, None] * s_jk).mean(axis=0)  # (100,6,6)
    E_alpha_sj = (alpha_i[:, None, None] * probs).mean(axis=0)  # (100,6)

    # Prices and observed shares by market
    prices_by_market = df['p'].values.reshape(100, 6)
    shares_by_market = obs_shares.reshape(100, 6)

    # p_k / s_j
    p_over_s = prices_by_market[:, None, :] / shares_by_market[:, :, None]

    # Cross-price: negative sign convention
    cross_price_matrices = -E_alpha_sjsk * p_over_s

    # Own-price diagonal adjustment
    for j in range(6):
        own = (E_alpha_sj[:, j] - E_alpha_sjsk[:, j, j])
        cross_price_matrices[:, j, j] = (prices_by_market[:, j] / shares_by_market[:, j]) * own

    # Average across markets -> (6,6)
    mean_cross = cross_price_matrices.mean(axis=0)
    return mean_cross


##################
# INIT VARIABLES #
##################

# Start with initial guesses of the non-linear parameters
gamma = np.array([0.1, 0.1, 0.1]) #gamma_{11}, gamma_{21}, gamma_{22} respectively
#beta = [iv_model.params['p'],iv_model.params['x']] # price coefficient and x coefficient respectively
delta = np.random.rand(600)
xi_opt_global = None
delta_opt_global = None
beta_opt_global = None

'''
'''
def calc_opt_W(temp_gamma, delta0):
    global xi_opt_global, delta_opt_global
    delta=delta0.copy()
    cm_obj = 1
    # Contraction mapping for delta - the mean utility levels for products
    while cm_obj > inner_tol:
        s_pred = calc_s_pred(delta, temp_gamma)
        delta_k1 = delta + (np.log(obs_shares) - np.log(s_pred))
        cm_obj = np.sum(np.abs(delta_k1 - delta))
        delta = delta_k1

    delta_opt_global = delta  # Store optimized delta
    # Now we have the "correct" delta. 
    reg_df = df.copy()
    reg_df['y'] = delta
    # Run 2SLS to estimate beta
    # Use z as an instrument for Prices and run IV regression
    controls = ['x']
    # Regressors: const + endogenous + controls
    exog = sm.add_constant(reg_df[['p'] + controls])
    # exog = reg_df[['p'] + controls]

    # Instruments: const + excluded instrument(s) + controls
    instr = sm.add_constant(reg_df[['z1','z2','z3','z4','z5','z6'] + controls])
    # instr = reg_df[['z1','z2','z3','z4','z5','z6'] + controls]
    iv_model = IV2SLS(reg_df['y'], exog, instr).fit()
    beta = iv_model.params.values[1:]
    # beta = iv_model.params.values
    # Estimate xi, unobserved product heterogeneity
    # xi = delta - np.mult(reg_df[['p','x']].values, beta)
    xi = delta - reg_df[['p','x']].values @ beta - iv_model.params.values[0]
    xi_opt_global = xi  # Store xi

    # Calculate moment res which is something with the instruments and the 
    gmm_obj = calc_gmm_obj(xi, W)
    print(gmm_obj)
    return gmm_obj  # return both objective and xi, xi



# Runs one loop of BLP
def blp_obj(temp_gamma, delta0):
    global beta_opt_global
    global xi_global
    cm_obj = 1
    delta = delta0.copy()
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
    # exog = reg_df[['p'] + controls]

    # Instruments: const + excluded instrument(s) + controls
    instr = sm.add_constant(reg_df[['z1','z2','z3','z4','z5','z6'] + controls])
    # Include version with no constant
    # instr = reg_df[['z1','z2','z3','z4','z5','z6'] + controls]
    iv_model = IV2SLS(reg_df['y'], exog, instr).fit()
    beta = iv_model.params.values[1:]
    # beta = iv_model.params.values
    beta_opt_global = beta

    # Estimate xi, unobserved product heterogeneity
    xi = delta - reg_df[['p','x']].values @ beta - iv_model.params.values[0]
    xi_global = xi
    # Calculate moment res which is something with the instruments and the 
    gmm_obj = calc_gmm_obj(xi, W_opt)
    print(gmm_obj)
    return gmm_obj



# Calculate init optimal IV matrix
# Use Nelder-Mead

res = minimize(
    calc_opt_W,
    x0=[4.19,4.911,-1.820],
    args=(delta,),                 # or args=(data1, data2, ...)
    method="Nelder-Mead",
    options={
        "maxiter": 1000,
        "xatol": 1e-6,        # tolerance on x
        "fatol": 1e-8,        # tolerance on f
        "disp": True
    }
)
# message: Optimization terminated successfully.
         #Current function value: 0.001625
         #Iterations: 118
        # Function evaluations: 210

 #Compute new W matrix: W = (1/J * Z' * diag(xi^2) * Z)^-1
J = 6
W_opt = np.linalg.inv((1/J) * (z_inst.T @ np.diag((xi_opt_global**2).flatten()) @ z_inst))

# Second Step using optimal W (use delta from step 1)
gamma_opt1 = res.x
res2 = minimize(
    blp_obj,
    x0=[4.19,4.911,-1.820],
    args=(delta_opt_global,),  # Use optimized delta from step 1
    method="Nelder-Mead",
    options={
        "maxiter": 1000,
        "xatol": 1e-6,        # tolerance on x
        "fatol": 1e-8,        # tolerance on f
        "disp": True
    })

print(beta_opt_global)
gamma_opt= np.insert(res2.x,1,0)
gamma_opt = np.reshape(gamma_opt,(2,2))
Omega_opt = gamma_opt @ gamma_opt.T

print(beta_opt_global)
'''
[-1.54976786  1.34234004]
'''
print(gamma_opt)
'''
[[ 4.1920799   0.        ]
 [ 4.91116479 -1.82063504]]
'''
print(Omega_opt)
'''
[[17.57353385 20.5879952 ]
 [20.5879952  27.43425161]]
'''

l_input = beta_opt_global
nl_input = gamma_opt

# Calculate Own Price Elasticities
jj_elas = calc_jj_elas(l_input, nl_input, N=10000)
print("Own Price Elasticities:")
print(jj_elas)

# Calculate Cross Price Elasticities
cross_price_matrices = calc_jk_elas(l_input, nl_input, N=10000)
print(np.diag(cross_price_matrices))

# Replace cross_price matrix diagonal with own-price elasticities
np.fill_diagonal(cross_price_matrices, jj_elas)
print("Elasticity Matrix:")
print(cross_price_matrices)
# Print it in a nice format
elas_df = pd.DataFrame(cross_price_matrices, 
                       columns=[f'Product {i+1}' for i in range(6)],
                       index=[f'Product {i+1}' for i in range(6)])
print(elas_df)

'''
           Product 1  Product 2  Product 3  Product 4  Product 5  Product 6
Product 1  -0.004520   0.003928   0.179544   0.252988   0.190164   0.131694
Product 2   0.006047  -0.004679   0.211048   0.152887   0.116066   0.202868
Product 3   0.003350   0.001622  -3.038214   0.109961   0.045566  -0.029653
Product 4   0.003395   0.001728   0.130416  -3.061436   0.041685  -0.095581
Product 5   0.000502   0.001003  -0.245244  -0.023026  -1.964587  -0.001636
Product 6   0.000618   0.000313  -0.135376  -0.061512  -0.034640  -1.498974
'''