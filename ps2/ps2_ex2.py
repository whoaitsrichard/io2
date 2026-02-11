import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

df = pd.read_csv('data/ps2_ex2.csv')

dx = 10
J = 2
beta = 0.999

df['diff'] = df.milage.diff().shift(-1)
df['action'] = (df['diff'] < 0).astype(int)
# Get rid of diff column
df.drop(columns=['diff'], inplace=True)


# Discretize milage into bins
bins = np.linspace(0, df["milage"].max(), dx + 1)
df["milage_bin"] = pd.cut(df["milage"], bins=bins, labels=False, include_lowest=True)

# Subtract 1 from milage_bin to make it 0-indexed
df["milage_bin"] = df["milage_bin"].astype(int)

# Create column that indicates next state
df['next_milage_bin'] = df.milage_bin.shift(-1)

# Drop last row because of nan
df.dropna(inplace=True)

# Convert all bins to ints
df['next_milage_bin'] = df['next_milage_bin'].astype(int)


replace_df = df[df.action == 1].copy()
cont_df = df[df.action == 0].copy()



# Construct transition matrix
replace_trans_mx = np.zeros((dx, dx))
cont_trans_mx = np.zeros((dx, dx)) # the first element is the transition probabilties from state 0 to all states.
milage_bins = np.arange(dx)
for i in range(dx):
    # Continue transition matrix
    temp_df = cont_df[cont_df.milage_bin == i]
    temp_value_counts = temp_df.next_milage_bin.value_counts().reset_index()
    
    # Ensure all states are represented with 10 rows
    all_states = pd.DataFrame({'next_milage_bin': milage_bins})
    temp_value_counts = all_states.merge(temp_value_counts, on='next_milage_bin', how='left').fillna(0)
    temp_value_counts['count'] = temp_value_counts['count'].astype(int)
    
    tot_sum = temp_value_counts['count'].sum()
    temp_value_counts['prob'] = temp_value_counts['count'] / tot_sum if tot_sum > 0 else 0
    cont_trans_mx[i] = temp_value_counts['prob'].values
    
    # Replace transition matrix
    temp_df = replace_df[replace_df.milage_bin == i]
    temp_value_counts = temp_df.next_milage_bin.value_counts().reset_index()
    
    # Ensure all states are represented with 10 rows
    all_states = pd.DataFrame({'next_milage_bin': milage_bins})
    temp_value_counts = all_states.merge(temp_value_counts, on='next_milage_bin', how='left').fillna(0)
    temp_value_counts['count'] = temp_value_counts['count'].astype(int)
    
    tot_sum = temp_value_counts['count'].sum()
    temp_value_counts['prob'] = temp_value_counts['count'] / tot_sum if tot_sum > 0 else 0
    replace_trans_mx[i] = temp_value_counts['prob'].values


cont_trans_df = pd.DataFrame(cont_trans_mx)
replace_trans_df = pd.DataFrame(replace_trans_mx)

full_trans_mx = [cont_trans_mx, replace_trans_mx]

data_df = df[['milage_bin','action']].copy()
data_df.rename(columns={'milage_bin': 'state'}, inplace=True)

def utility(
    x: int, 
    a: int,
    theta: list):
    """Calculates the utility at a given action and state.
        
        Parameters
        ----------
            x : `int`
                State.
            a : `int`
                Action.
            theta : `list`
                Utility parameters.
    """
    
    if a == 0:
        u = -theta[0]*x - theta[1]*(x/100)**2
    elif a == 1:
        u = -theta[2]
        
    return u

def utility_matrix(
    temp_dx: int,
    temp_J: int,
    theta: list):
    """Returns the matrix of utilites over all states and choices.
        
        Each row corresponds to a state.
        
        Each column corresponds to an action.
        
        Parameters
        ----------
            dx : `int`
                Number of states.
            J : `int`
                Number of actions.
            theta: `list`
                Utility parameters.
    """
    
    umat = np.zeros((temp_dx, temp_J))
    
    for a in range(temp_J):
        for x in range(temp_dx):
            umat[x, a] = utility(x, a, theta)
            
    return umat

# umat = utility_matrix(dx, J, theta)





def csvf_state(u_matrix, V, x):
    """Choice-specific value function at a particular state.
        
        Parameters
        ----------
            u_matrix : `dgp`
                matrix of utilities based on some theta
            V : `ndarray`
                A vector of value function.
            x : `int`
                State.
    """
    
    w = [u_matrix[x, a] + beta * np.matmul(full_trans_mx[a][x, :], V) for a in range(J)]
    
    return w

def csvf(u_matrix, V):
    """Choice-specific value functions across all states.
    
        The function stacks the output of `csvf_state` over all states.
        
        Parameters
        ----------
            u_matrix : `ndarray`
                matrix of utilities based on some theta.
            V : `ndarray`
                A vector of value function.
    """
    
    wmat = np.zeros((dx, J))
    
    for x in range(dx):
        wmat[x, :] = csvf_state(u_matrix, V, x)    
    
    return wmat

def vf(w):
    """Computes the integrated value function.
    
        This function computes the integrated value function based on the 
        choice-specific value functions.
        
        Parameters
        ----------
            w : `ndarray`
                A vector of value functions.
    """
    
    # Extract the maximum value for numerical stability (log-sum-exp trick)
    # This prevents overflow when exponentiating large values
    wmax = np.amax(w)
    
    # Compute log(exp(w_0 - max) + exp(w_1 - max))
    # This is numerically stable because the exponentials are now of small values
    v = np.log(np.exp(w[:, 0] - wmax) + np.exp(w[:, 1] - wmax))
    
    # Add back the maximum value to recover the true log-sum-exp:
    # log(exp(w_0) + exp(w_1)) = log(exp(w_0 - max) + exp(w_1 - max)) + max
    v += wmax
    
    return v


def vfi(u_matrix, ϵ = 1e-8):
    """Value function iteration.
        
        Tadxes the DGP object and returns the fixed-point of the value function equation.
        
        This function implements the fixed-point iteration algorithm described in the 
        "Fixed point representation" section. It solves the Bellman equation by iteratively
        updating the value function until convergence.
        
        Algorithm steps:
        1. Start with an initial guess V0 for the integrated value function w(x)
        2. Compute choice-specific values: w = csvf(dgp, V0) gives v_j(x) for all choices j
        3. Compute integrated value: V1 = vf(w) gives w(x) = ln(sum_j exp{v_j(x)})
        4. Checdx convergence: if ||V0 - V1|| < ϵ, we've found the fixed point
        5. Otherwise, update V0 = V1 and repeat
        
        Parameters
        ----------
            dgp : `dgp`
                The primitives of the DDC model.
            ϵ : `float`
                The tolerance level.
    """
    
    # Initialize - construct utility, transition matrices, and value function
    V0 = np.zeros(10) # number of states is 10

    # Iterate until convergence
    gap = 10000.
    iter = 1
    while gap > ϵ:
        w = csvf(u_matrix, V0)
        V1 = vf(w)
        gap = norm(V0 - V1)
        V0 = V1
        iter += 1
        
    return V0

def prob_from_dgp(u_matrix, V):
    """Conditional choice probability of maintaining the engine.
    
        Compute the conditional choice probability for maintaining the engine.
        
        Parameters
        ----------
            dgp : `dgp`
                The primitives of the DDC model.
            V : `ndarray`
                A vector of value functions.
    """
    
    # Compute the choice probability and take the difference
    w = csvf(u_matrix, V)
    
    # Extract the maximum value for numerical stability (softmax trick)
    # This prevents overflow when exponentiating
    wmax = np.amax(w)
    wdifexp = np.exp(w - wmax)
    
    # Compute the conditional choice probability
    prob = np.zeros((dx, J))
    wsum = np.sum(wdifexp, axis = 1)
    for a in range(J):
        prob[:, a] = wdifexp[:, a]/wsum
        
    return prob

def likelihood_nfxp(theta, data):
    """The likelihood function for NFXP.
        
        Parameters
        ----------
        theta : `list`
            The utility parameters.
        data : `DataFrame`
            The data that contains the actions and states.
        dgp : `dgp`
            The primitives of the DDC model.
    """
    
    # Get value functions from parameters
    temp_u_matrix = utility_matrix(dx, J, theta)
    V = vfi(temp_u_matrix)
    
    # Get the choice probabilities
    pr0 = prob_from_dgp(temp_u_matrix, V)
        
    # Compute likelihood of data
    l = likelihood(data, pr0)
    
    return l


def likelihood(data, prob):
    """The likelihood function with the given choice probability
    
        Parameters
        ----------
        data : `DataFrame`
            The data that contains the actions and states.
        u_matrix : `ndarray`
            The utility matrix.
        prob : `ndarray`
            A matrix of choice probabilities.
    """
    
    # Compute likelihood of data
    data['likelihood'] = data.apply(lambda row: np.log(prob[int(row.state), int(row.action)]), axis=1)
    l = data['likelihood'].mean()
    return -l



def nested_fixed_point(init_theta, data):
    """The nested fixed-point algorithm.

        Parameters
        ----------
        data : `DataFrame`
            The data that contains the actions and states.
        dgp : `dgp`
            The primitives of the DDC model.
    """
    def ll(x):
        temp_ll = likelihood_nfxp(x, data)
        print(temp_ll)
        return temp_ll
    theta = minimize(ll, init_theta)
    return theta


f = nested_fixed_point([1, 0.1, 4], data_df)

# Predicted probabilities from estimated theta
est_theta = f.x
est_u_matrix = utility_matrix(dx, J, est_theta)
est_V = vfi(est_u_matrix)
pred_prob = prob_from_dgp(est_u_matrix, est_V)

# Empirical ("true") probability of maintaining (action=0) at each state
empirical_prob = data_df.groupby('state')['action'].apply(lambda a: 1 - a.mean()).values

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(milage_bins, pred_prob[:, 0], 'o-', label='Predicted (estimated model)')
ax.plot(milage_bins, empirical_prob, 's--', label='Empirical (data)')
ax.set_xlabel('Mileage bin')
ax.set_ylabel('Probability of maintaining engine')
ax.set_title('Predicted vs. Empirical Maintenance Probability')
ax.set_xticks(milage_bins)
ax.legend()
plt.tight_layout()
plt.savefig('outputs/ex2_maintenance_prob.png', dpi=150)
plt.show()