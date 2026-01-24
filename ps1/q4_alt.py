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
def calc_own_price(mu, var,N):
    np.random.seed(44259)
    alpha_draws = np.random.normal(mu, np.sqrt(var), size=(N, 1))
    integrand = alpha_draws * obs_shares.T * (1-obs_shares.T) 
    own_price_est = integrand.mean(axis=0) * p_over_s_jt
    return own_price_est

'''
Calculates cross-price elasticities using numerical integration
'''
def calc_cross_price_rc(delta_opt, gamma_opt, N=1000):                                                                                       
      """                                                                                                                                      
      Cross-price elasticities using individual-level shares from RC model.                                                                    
      Returns (100, 6, 6) array.                                                                                                               
      """                                                                                                                                      
      np.random.seed(44259)                                                                                                                    
      v_i = np.random.normal(0.0, 1.0, size=(N, 2))                                                                                            
                                                                                                                                               
      L = gamma_opt  # (2, 2) Cholesky factor                                                                                                  
      w = (L @ v_i.T).T  # (N, 2) - random coefficients for each consumer                                                                      
                                                                                                                                               
      # Price coefficient for each consumer: α_i = β_p + σ_p * v_i1                                                                            
      alpha_i = beta_opt_global[0] + w[:, 0]  # (N,)                                                                                           
                                                                                                                                               
      # Reshape data by market                                                                                                                 
      delta_by_market = delta_opt.reshape(100, 6)                                                                                              
      x_by_market = x_jt.reshape(100, 6, 2)                                                                                                    
      prices_by_market = df['p'].values.reshape(100, 6)                                                                                        
      shares_by_market = obs_shares.reshape(100, 6)                                                                                            
                                                                                                                                               
      # Compute μ_{ijm} = x_jm · w_i for all markets, products, consumers                                                                      
      mu = np.einsum('mjk,ik->mji', x_by_market, w)  # (100, 6, N)                                                                             
                                                                                                                                               
      # Utility: u_{ijm} = δ_{jm} + μ_{ijm}                                                                                                    
      u = delta_by_market[:, :, None] + mu  # (100, 6, N)                                                                                      
                                                                                                                                               
      # Individual choice probabilities (within each market)                                                                                   
      exp_u = np.exp(u)                                                                                                                        
      denom = exp_u.sum(axis=1) + 1  # (100, N)                                                                                                
      s_ij = exp_u / denom[:, None, :]  # (100, 6, N)                                                                                          
                                                                                                                                               
      # E[α_i · s_{ij} · s_{ik}] for all (j,k) pairs                                                                                           
      s_jk = s_ij[:, :, None, :] * s_ij[:, None, :, :]  # (100, 6, 6, N)                                                                       
      integrand = alpha_i[None, None, None, :] * s_jk                                                                                          
      E_alpha_sjsk = integrand.mean(axis=3)  # (100, 6, 6)                                                                                     
                                                                                                                                               
      # E[α_i · s_{ij}] for own-price elasticity                                                                                               
      E_alpha_sj = (alpha_i[None, None, :] * s_ij).mean(axis=2)  # (100, 6)                                                                    
                                                                                                                                               
      # p_k / s_j                                                                                                                              
      p_over_s = prices_by_market[:, None, :] / shares_by_market[:, :, None]                                                                   
                                                                                                                                               
      # Cross-price: η_{jk} = -(p_k / s_j) · E[α_i · s_{ij} · s_{ik}]                                                                          
      cross_price_matrices = -E_alpha_sjsk * p_over_s                                                                                          
                                                                                                                                               
      # Own-price (diagonal): η_{jj} = (p_j / s_j) · E[α_i · s_{ij} · (1 - s_{ij})]                                                            
      for j in range(6):                                                                                                                       
          own_price = (E_alpha_sj[:, j] - E_alpha_sjsk[:, j, j])                                                                               
          cross_price_matrices[:, j, j] = (                                                                                                    
              prices_by_market[:, j] / shares_by_market[:, j]                                                                                  
          ) * own_price                                                                                                                        
                                                                                                                                               
      return cross_price_matrices    


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

    # Calculate moment res which is something with the instruments and the 
    gmm_obj = calc_gmm_obj(xi, W_opt)
    print(gmm_obj)
    return gmm_obj



# Calculate init optimal IV matrix
# Use Nelder-Mead

res = minimize(
    calc_opt_W,
    x0=[0.1,0.1,0.1],
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
    x0=[0.1,0.1,0.1],
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
[-1.60267124  1.12718513]

'''
print(gamma_opt)

print(Omega_opt)
'''
[[17.74390595 22.27206761]
 [22.27206761 30.42147995]]
'''

np.linalg.det(Omega_opt)

# Calculate Own Price Elasticities
df['own_price_e'] = calc_own_price(beta_opt_global[0], Omega_opt[0,0],1000)
mean_own_price_e = df.groupby('choice')['own_price_e'].mean()
print(mean_own_price_e)

# Calculate Cross Price Elasticities using RC model
cross_price_matrices = calc_cross_price_rc(delta_opt_global, gamma_opt, N=1000)

# Mean elasticity matrix across all 100 markets
mean_cross_price_e = cross_price_matrices.mean(axis=0)

# Print mean cross-price elasticity matrix
pd.set_option('display.precision', 3)
cross_price_df = pd.DataFrame(
    mean_cross_price_e,
    columns=[f'Prod {i+1}' for i in range(6)],
    index=[f'Prod {i+1}' for i in range(6)]
)
print("\nMean Cross-Price Elasticity Matrix (averaged across markets):")
print("Rows = product j whose share changes, Columns = product k whose price changes")
print(cross_price_df)

# Show elasticity matrix for a single market as example
print("\nCross-Price Elasticity Matrix for Market 1:")
market1_df = pd.DataFrame(
    cross_price_matrices[0],
    columns=[f'Prod {i+1}' for i in range(6)],
    index=[f'Prod {i+1}' for i in range(6)]
)
print(market1_df)