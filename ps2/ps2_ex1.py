import os
import copy
import random
import array
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from numpy.linalg import norm
from numpy import ndarray
from cyipopt import minimize_ipopt

# =============================================================================
# Exercise 1.1 — Code from ps2_ex1 notebook
# =============================================================================

class dgp:
    """Creates the primitives of the dynamic discrete choice model."""
    def __init__(
        self,
        β: float,
        u: ndarray,
        F: ndarray,
        dx: int,
        J: int):

        self.β = β
        self.u = u
        self.F = F
        self.dx = dx
        self.J = J

β = 0.95
θ = [-0.2, 5]
dx = 10
J = 2

def utility(
    x: int,
    a: int,
    θ: list):
    """Calculates the utility at a given action and state."""
    if a == 0:
        u = x * θ[0] + θ[1]
    elif a == 1:
        u = 0
    return u

def utility_matrix(
    dx: int,
    J: int,
    θ: list):
    """Returns the matrix of utilites over all states and choices."""
    umat = np.zeros((dx, J))
    umat[:, 0] = np.arange(dx) * θ[0] + θ[1]
    umat[:, 1] = 0.0
    return umat

umat = utility_matrix(dx, J, θ)

def transition(
    f: list,
    dx: int):
    """Creates the transition matrices."""
    F0 = np.zeros((dx, dx))
    F1 = copy.copy(F0)

    for i in range(dx):
        if i < dx - 2:
            F0[i, i] = f[0]
            F0[i, i + 1] = f[1]
            F0[i, i + 2] = 1 - f[0] - f[1]
        elif i == dx - 2:
            F0[i, i] = f[0]
            F0[i, i + 1] = 1 - f[0]
        else:
            F0[i, i] = 1

    F1[:, 0] = 1
    return F0, F1

f = [0.3, 0.6]
F = transition(f, dx)

primitive = dgp(β, umat, F, dx, J)

def csvf_state(dgp, V, x):
    """Choice-specific value function at a particular state."""
    w = [dgp.u[x, a] + dgp.β * np.matmul(dgp.F[a][x, :], V) for a in range(dgp.J)]
    return w

def csvf(dgp, V):
    """Choice-specific value functions across all states."""
    wmat = np.column_stack([
        dgp.u[:, a] + dgp.β * dgp.F[a] @ V for a in range(dgp.J)
    ])
    return wmat

def vf(w):
    """Computes the integrated value function."""
    wmax = np.amax(w)
    v = np.log(np.exp(w[:, 0] - wmax) + np.exp(w[:, 1] - wmax))
    v += wmax
    return v

def vfi(dgp, ϵ = 1e-8):
    """Value function iteration."""
    V0 = np.zeros(dgp.dx)
    gap = 10000.
    iter = 1
    while gap > ϵ:
        w = csvf(dgp, V0)
        V1 = vf(w)
        gap = norm(V0 - V1)
        V0 = V1
        iter += 1
    return V0

v_true = vfi(primitive)

def prob_from_dgp(dgp, V):
    """Conditional choice probability of maintaining the engine."""
    w = csvf(dgp, V)
    wmax = np.amax(w, axis=1, keepdims=True)
    wdifexp = np.exp(w - wmax)
    prob = wdifexp / np.sum(wdifexp, axis=1, keepdims=True)
    return prob

prob_true = prob_from_dgp(primitive, v_true)
prob_maintain_true = prob_true[:, 0]

# --- Simulate data ---

def draw(dgp, V, N, T):
    """Simulate Rust data."""
    action = []
    state = []
    ids = []
    ts = []

    for i in range(N):
        ai = np.zeros(T)
        si = np.zeros(T, dtype=int)
        si[0] = 0
        ai[0] = None

        for t in range(T - 1):
            uf = np.random.uniform()

            if (t > 0) & (ai[t] == 0) | (t == 0):
                st = si[t]
                f0 = dgp.F[0][st]

                if si[t] < dgp.dx - 2:
                    if uf < f0[st]:
                        si[t + 1] = si[t]
                    elif (uf >= f0[st]) & (uf < (f0[st] + f0[st + 1])):
                        si[t + 1] = si[t] + 1
                    else:
                        si[t + 1] = si[t] + 2
                elif si[t] == dgp.dx - 2:
                    if uf < f0[st]:
                        si[t + 1] = si[t]
                    else:
                        si[t + 1] = si[t] + 1
                else:
                    si[t + 1] = si[t]

            elif ai[t] == 1:
                si[t + 1] = 0

            us = csvf_state(dgp, V, int(si[t + 1]))
            us += np.random.gumbel(-.577, size=dgp.J)
            ai[t + 1] = np.argmax(us)

        state = np.concatenate((state, si))
        action = np.concatenate((action, ai))
        ids = np.concatenate((ids, np.repeat(i + 1, T)))
        ts = np.concatenate((ts, range(0, T)))

    df = pd.DataFrame({'i': ids, 't': ts, 'action': action, 'state': state})
    return df

def estimate_ccp(data, dgp):
    """Estimates conditional choice probability of maintaining the engine."""
    phat = np.zeros((dgp.dx, dgp.J))

    for x in range(dgp.dx):
        data_x = data.loc[(data.t > 0) & (data.state == x)]
        n_obs = len(data_x)

        if n_obs > 0:
            for a in range(dgp.J):
                phat[x, a] = np.mean(data_x.action == a)
        else:
            phat[x, :] = phat[x-1, :]
            print(f"Warning: State {x} has no observations, imputing from state {x-1}")

    epsilon = .01
    phat = phat + epsilon
    phat = phat / phat.sum(axis=1, keepdims=True)
    return phat

def estimate_transition_maintain(data, dgp):
    """Estimated the transition matrix for maintaining bus engine."""
    data_tmp = copy.copy(data)
    data_tmp['state_shift'] = data_tmp.groupby('i')['state'].shift(-1)
    data_tmp['state_dif'] = data_tmp['state_shift'] - data_tmp['state']
    data_sub = data_tmp.loc[(data_tmp.action == 0) & (data_tmp.t != max(data_tmp.t))]

    unique_dif = pd.unique(data_sub['state_dif'])
    probs = np.zeros(len(unique_dif))
    for idu, val in enumerate(unique_dif):
        probs[idu] = np.mean(data_sub['state_dif'] == val)
    return probs

def estimate_transition(data, dgp):
    """Estimate the transition matrix based on the data."""
    maintain_prob = estimate_transition_maintain(data, dgp)
    transition_mat = transition(maintain_prob[range(dgp.J)], dgp.dx)
    return transition_mat

# =============================================================================
# Exercise 1.1 — Plots by β, θ_1, θ_2
# =============================================================================

print("=" * 60)
print("Exercise 1.1: Plotting CCPs for varying parameters")
print("=" * 60)

beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, .99]
theta_variations = [[-2, 5],[-1, 5],[-0.5, 5], [-0.2, 5],[0, 5], [-0.2, 1],[-0.2, 2.5],[-0.2, 5],[-0.2, 7.5],[-0.2, 10]]
beta_fixed = 0.95
theta_fixed = [-0.2, 5]
dx_fixed = 10
J_fixed = 2

param_list = []
for beta in beta_values:
    param_list.append({"beta": beta, "theta": theta_fixed, "dx": dx_fixed, "J": J_fixed})
for theta in theta_variations:
    param_list.append({"beta": beta_fixed, "theta": theta, "dx": dx_fixed, "J": J_fixed})

for params in param_list:
    params['u'] = utility_matrix(params['dx'], params['J'], params['theta'])

for params in param_list:
    params['F'] = transition([0.3, 0.6], params['dx'])

dgp_list = [
    dgp(params['beta'], params['u'], params['F'], params['dx'], params['J'])
    for params in param_list
]

v_list = [vfi(primitive) for primitive in dgp_list]

results = []
for i in range(21):
    result = prob_from_dgp(dgp_list[i], v_list[i])
    results.append(result[:, 0])

df = pd.DataFrame({
    'beta': [params['beta'] for params in param_list],
    'theta_0': [params['theta'][0] for params in param_list],
    'theta_1': [params['theta'][1] for params in param_list],
})
result_df = pd.DataFrame(results, columns=[f'state_{i}' for i in range(10)])
df = pd.concat([df, result_df], axis=1)

df = df.melt(
    id_vars=['beta', 'theta_0', 'theta_1'],
    value_vars=[f'state_{i}' for i in range(10)],
    var_name='state',
    value_name='value'
)
df['state'] = df['state'].str.extract('(\\d+)').astype(int)

df_beta = df[(df['theta_0'] == -0.2) & (df['theta_1'] == 5)]
df_theta_0 = df[(df['beta'] == beta_fixed) & (df['theta_1'] == 5)]
df_theta_1 = df[(df['beta'] == beta_fixed) & (df['theta_0'] == -0.2)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for beta in df_beta['beta'].unique():
    data = df_beta[df_beta['beta'] == beta]
    axes[0].plot(data['state'], data['value'], label=f'β={beta:.2f}')
axes[0].set_xlabel('State')
axes[0].set_ylabel('Value')
axes[0].set_title('Varying β (θ fixed)')
axes[0].set_ylim(0, 1)
axes[0].legend()

for theta0 in df_theta_0['theta_0'].unique():
    data = df_theta_0[df_theta_0['theta_0'] == theta0]
    axes[1].plot(data['state'], data['value'], label=f'θ_0={theta0:.2f}')
axes[1].set_xlabel('State')
axes[1].set_ylabel('Value')
axes[1].set_title('Varying θ_0 (β,θ_1 fixed)')
axes[1].set_ylim(0, 1)
axes[1].legend()

for theta1 in df_theta_1['theta_1'].unique():
    data = df_theta_1[df_theta_1['theta_1'] == theta1]
    axes[2].plot(data['state'], data['value'], label=f'θ_1={theta1:.2f}')
axes[2].set_xlabel('State')
axes[2].set_ylabel('Value')
axes[2].set_title('Varying θ_1 (β,θ_0 fixed)')
axes[2].set_ylim(0, 1)
axes[2].legend()

plt.tight_layout()
plt.savefig('outputs/ex1_1_plots.png', dpi=150)
plt.close()
print("Saved outputs/ex1_1_plots.png")

# =============================================================================
# Exercise 1.2 — Two-step CCP estimation
# =============================================================================

def likelihood(data, dgp, prob):
    """The likelihood function with the given choice probability."""
    N = int(data.i.max())
    T = int(data.t.max())
    data_sub = data[data.t > 0]
    a = data_sub['action'].astype(int).values
    x = data_sub['state'].astype(int).values
    l = np.sum(np.log(prob[x, a]))
    return -l / (N * T)

def csvf_from_ccp(dgp, phat):
    """Estimates choice-specific value function from choice probabilities."""
    ep = 1e-8
    phat_clipped = np.clip(phat, ep, None)
    csvf = np.zeros((dgp.dx, dgp.J))
    csvf[:, 1] = 0.0
    csvf[:, 0] = np.log(phat_clipped[:, 0]) - np.log(phat_clipped[:, 1])
    return csvf

def likelihood_two_step_ccp(θ, data, dgp, phat):
    """The likelihood function for the two-step method."""
    try:
        csvf = csvf_from_ccp(dgp, phat)

        wmax = np.amax(csvf)
        w_hat = np.log(np.exp(csvf[:, 0] - wmax) + np.exp(csvf[:, 1] - wmax)) + wmax
        Ew_0 = dgp.F[0] @ w_hat
        Ew_1 = dgp.F[1] @ w_hat

        u_mat = utility_matrix(dgp.dx, dgp.J, θ)
        u_diff = u_mat[:, 0] - u_mat[:, 1]

        v_bar = np.zeros((dgp.dx, dgp.J))
        v_bar[:, 0] = np.exp(u_diff + dgp.β * (Ew_0 - Ew_1))
        v_bar[:, 1] = np.exp(0)

        prob = v_bar / np.sum(v_bar, axis=1, keepdims=True)

        l = likelihood(data, dgp, prob)

        if np.isfinite(l):
            return l
        else:
            return 1e10

    except Exception as e:
        return 1e10

def two_step_ccp(data, dgp):
    """The two-step CCP approach."""
    phat = estimate_ccp(data, dgp)
    def ll(x):
        return likelihood_two_step_ccp(x, data, dgp, phat)
    θ = minimize(ll, [0.1, 0.1])
    return θ

# =============================================================================
# Exercise 1.3 — Simulate and estimate with two-step CCP
# =============================================================================

print("\n" + "=" * 60)
print("Exercise 1.3: Simulate data and estimate with two-step CCP")
print("=" * 60)

N = 30
T = 50
f = [0.3, 0.6]
F = transition(f, dx)
β = 0.95
θ_sim = [-.1, 4]
dx = 10
J = 2

umat_sim = utility_matrix(dx, J, θ_sim)
primitive_sim = dgp(β, umat_sim, F, dx, J)
v_sim = vfi(primitive_sim)

np.random.seed(100)
df = draw(primitive_sim, v_sim, N, T)

primitive_sim.F = estimate_transition(df, primitive_sim)
start_time = time.perf_counter()
ccp_out = two_step_ccp(df, primitive_sim)
end_time = time.perf_counter()
print(f"True theta: {θ_sim}")
print(f"Estimated theta: {ccp_out.x}")
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

umat_est = utility_matrix(dx, J, ccp_out.x)
primitive_est = dgp(β, umat_est, primitive_sim.F, dx, J)
v_sim_true = vfi(primitive_sim)
v_sim_est = vfi(primitive_est)

prob_true = prob_from_dgp(primitive_sim, v_sim_true)
prob_est = prob_from_dgp(primitive_est, v_sim_est)
prob_maintain_true = prob_true[:, 0]
prob_maintain_est = prob_est[:, 0]

fig, ax = plt.subplots()
xs = range(primitive_sim.dx)
ax.plot(xs, prob_maintain_true, label='True')
ax.plot(xs, prob_maintain_est, label='Estimated')
ax.set_xlabel('State')
ax.set_ylabel('Probability of maintaining engine')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/ex1_3_plots.png', dpi=150)
plt.close()
print("Saved outputs/ex1_3_plots.png")

# =============================================================================
# Exercise 1.4 — Compare all estimation methods
# =============================================================================

print("\n" + "=" * 60)
print("Exercise 1.4: Compare NFXP, two-step, NPL, MPEC")
print("=" * 60)

# --- NFXP ---

def likelihood_nfxp(θ, data, dgp):
    """The likelihood function for NFXP."""
    dgp_tmp = copy.copy(dgp)
    dgp_tmp.u = utility_matrix(dgp.dx, dgp.J, θ)
    V = vfi(dgp_tmp)
    pr0 = prob_from_dgp(dgp_tmp, V)
    l = likelihood(data, dgp_tmp, pr0)
    return l

def nested_fixed_point(data, dgp):
    """The nested fixed-point algorithm."""
    def ll(x):
        return likelihood_nfxp(x, data, dgp)
    θ = minimize(ll, [0.1, 0.1])
    return θ

# --- NPL ---

def estimate_unconditional_transition(dgp, phat):
    """Estimate the unconditional transition matrix."""
    P = np.zeros((dgp.dx, dgp.dx))
    for a in range(dgp.J):
        P += dgp.F[a] * np.transpose([phat[:, a] for i in range(dgp.dx)])
    return P

def nested_pseudo_likelihood_k(dgp, phat):
    """Step k of the nested pseudo likelihood method."""
    what = np.sum(phat * (dgp.u - np.log(phat)), axis=1)
    I = np.eye(dgp.dx)
    P = estimate_unconditional_transition(dgp, phat)
    inv_factor = np.linalg.inv(I - dgp.β * P)
    vbar = inv_factor @ what
    pnew = prob_from_dgp(dgp, vbar)
    return pnew

def likelihood_npl(θ, data, dgp, phat):
    """The likelihood function for NPL."""
    dgp_tmp = copy.copy(dgp)
    dgp_tmp.u = utility_matrix(dgp.dx, dgp.J, θ)
    pk = nested_pseudo_likelihood_k(dgp_tmp, phat)
    l = likelihood(data, dgp_tmp, pk)
    return l

def nested_pseudo_likelihood(data, dgp, phat, K):
    """Run the nested pseudo likelihood method with K iterations."""
    pk = copy.copy(phat)
    dgpk = copy.copy(dgp)

    for i in range(K):
        def ll(x):
            return likelihood_npl(x, data, dgpk, pk)
        θiter = minimize(ll, [0.1, 0.1]).x
        dgpk.u = utility_matrix(dgp.dx, dgp.J, θiter)
        pk = nested_pseudo_likelihood_k(dgpk, pk)
        if i == K - 1:
            return pk, θiter

# --- MPEC ---

def mpec(data, dgp):
    """MPEC estimation."""
    n_x = dgp.dx
    n_vars = 2 + n_x

    # Pre-extract data arrays once
    data_sub = data[data.t > 0]
    actions = data_sub['action'].astype(int).values
    states = data_sub['state'].astype(int).values
    N = int(data.i.max())
    T = int(data.t.max())
    NT = N * T
    xs = np.arange(n_x)

    def mpec_objective(x):
        θ = x[:2]
        v = x[2:]
        u_mat = np.zeros((n_x, 2))
        u_mat[:, 0] = xs * θ[0] + θ[1]
        w = np.column_stack([
            u_mat[:, a] + dgp.β * dgp.F[a] @ v for a in range(dgp.J)
        ])
        wmax = np.amax(w, axis=1, keepdims=True)
        wdifexp = np.exp(w - wmax)
        prob = wdifexp / np.sum(wdifexp, axis=1, keepdims=True)
        ll = -np.sum(np.log(prob[states, actions])) / NT
        return ll

    def mpec_objective_grad(x):
        θ = x[:2]
        v = x[2:]
        u_mat = np.zeros((n_x, 2))
        u_mat[:, 0] = xs * θ[0] + θ[1]
        w = np.column_stack([
            u_mat[:, a] + dgp.β * dgp.F[a] @ v for a in range(dgp.J)
        ])
        wmax = np.amax(w, axis=1, keepdims=True)
        wdifexp = np.exp(w - wmax)
        prob = wdifexp / np.sum(wdifexp, axis=1, keepdims=True)

        # Gradient of -log L w.r.t. prob
        # d(-ll)/d(prob[x,a]) = -1/(NT * prob[x,a]) for observed (x,a) pairs
        # Using chain rule through softmax:
        # d(-ll)/d(w[x,a]) = (1/NT) * (count[x,a] * prob[x,a] - count[x,a]) ...
        # More directly: for each obs (x_i, a_i), d(-log prob[x_i,a_i])/d(w[x,a])
        #   = prob[x,a] if x==x_i (for all a), minus 1{a==a_i, x==x_i}

        # Count observations per (state, action)
        counts = np.zeros((n_x, 2))
        np.add.at(counts, (states, actions), 1)

        # state_counts[x] = number of obs at state x
        state_counts = counts.sum(axis=1)

        # d(-ll)/d(w[x,a]) = (1/NT) * (state_counts[x] * prob[x,a] - counts[x,a])
        dll_dw = (state_counts[:, None] * prob - counts) / NT

        # d(w[x,0])/d(θ[0]) = x, d(w[x,0])/d(θ[1]) = 1, d(w[x,1])/d(θ) = 0
        grad = np.zeros(n_vars)
        grad[0] = np.sum(dll_dw[:, 0] * xs)  # d/dθ_0
        grad[1] = np.sum(dll_dw[:, 0])        # d/dθ_1

        # d(w[x,a])/d(v[j]) = β * F[a][x,j]
        for a in range(dgp.J):
            grad[2:] += dgp.β * dgp.F[a].T @ dll_dw[:, a]

        return grad

    def mpec_value_constraints(x):
        θ = x[:2]
        v = x[2:]
        u_mat = np.zeros((n_x, 2))
        u_mat[:, 0] = xs * θ[0] + θ[1]
        w = np.column_stack([
            u_mat[:, a] + dgp.β * dgp.F[a] @ v for a in range(dgp.J)
        ])
        return v - vf(w)

    def mpec_value_constraints_jac(x):
        θ = x[:2]
        v = x[2:]
        u_mat = np.zeros((n_x, 2))
        u_mat[:, 0] = xs * θ[0] + θ[1]
        w = np.column_stack([
            u_mat[:, a] + dgp.β * dgp.F[a] @ v for a in range(dgp.J)
        ])
        wmax = np.amax(w, axis=1, keepdims=True)
        ew = np.exp(w - wmax)
        prob = ew / np.sum(ew, axis=1, keepdims=True)

        # d(vf)/d(w[x,a]) = prob[x,a]  (softmax property)
        # d(w[x,a])/d(v[j]) = β * F[a][x,j]
        # d(vf[x])/d(v[j]) = sum_a prob[x,a] * β * F[a][x,j]
        dvf_dv = sum(dgp.β * (prob[:, a:a+1] * dgp.F[a]) for a in range(dgp.J))

        # d(vf[x])/d(θ_0) = prob[x,0] * x, d(vf[x])/d(θ_1) = prob[x,0] * 1
        dvf_dtheta0 = prob[:, 0] * xs
        dvf_dtheta1 = prob[:, 0]

        jac = np.zeros((n_x, n_vars))
        jac[:, 0] = -dvf_dtheta0
        jac[:, 1] = -dvf_dtheta1
        jac[:, 2:] = np.eye(n_x) - dvf_dv
        return jac

    def mpec_lb_constraint(x):
        return x[2:]

    c = [{'type': 'eq', 'fun': mpec_value_constraints, 'jac': mpec_value_constraints_jac},
         {'type': 'ineq', 'fun': mpec_lb_constraint}]

    x0 = np.concatenate((np.array([.1, .1]), np.zeros(dgp.dx)))

    return minimize_ipopt(mpec_objective,
                          jac=mpec_objective_grad,
                          x0=x0,
                          constraints=c,
                          options={"acceptable_tol": 1e-8})

# --- Summary ---

def solve_ddc(data, dgp, method, K):
    """Solves a DDC model."""
    if method == "nfxp":
        out = nested_fixed_point(data, dgp)
    elif method == "two step":
        out = two_step_ccp(data, dgp)
    elif method == "npl":
        phat = estimate_ccp(data, dgp)
        out = nested_pseudo_likelihood(data, dgp, phat, K)
    elif method == "mpec":
        out = mpec(data, dgp)
    return out

methods = ["nfxp", "two step", "npl", "mpec"]

umat_sim = utility_matrix(dx, J, θ_sim)
primitive_sim = dgp(β, umat_sim, F, dx, J)
v_sim = vfi(primitive_sim)

np.random.seed(1)
df = draw(primitive_sim, v_sim, N, T)

def solve_ddc_loop(thetas, K, N, T, f):
    F = transition(f, dx)
    method_list = []
    theta_list = []
    estimate_list = []
    prob_list = []
    time_list = []

    for i in range(len(thetas)):
        umat = utility_matrix(dx, J, thetas[i])
        primitive = dgp(β, umat, F, dx, J)
        v = vfi(primitive)
        np.random.seed(1)
        df = draw(primitive, v, N, T)
        primitive.F = estimate_transition(df, primitive)

        for method in methods:
            print(f"  {thetas[i]}, {method}")
            theta_list.append(thetas[i])
            method_list.append(method)

            start_time = time.perf_counter()
            est = solve_ddc(df, primitive, method, K)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)

            if method == "npl":
                estimate_list.append(est[1])
                prob_list.append(est[0][:, 0])
            else:
                theta_est = est.x[:2] if method == "mpec" else est.x
                estimate_list.append(theta_est)

                umat_est = utility_matrix(dx, J, theta_est)
                primitive_est = dgp(β, umat_est, primitive.F, dx, J)
                v_est = vfi(primitive_est)
                prob_est = prob_from_dgp(primitive_est, v_est)
                prob_list.append(prob_est[:, 0])

    result_df = pd.DataFrame({'theta': theta_list, 'method': method_list,
                              'time': time_list, 'estimate': estimate_list,
                              'prob': prob_list})
    return result_df

thetas = [[-.1, 3], [-.2, 3], [-.3, 3], [-.5, 3],
          [-.1, 5], [-.2, 5], [-.3, 5], [-.5, 5]]

estimates_df = solve_ddc_loop(thetas, K=10, N=30, T=50, f=[.3, .6])

# --- Plots ---

theta_plot = [[-.2, 3], [-.3, 3], [-.2, 5], [-.3, 5]]

for idx, theta in enumerate(theta_plot):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    umat_true = utility_matrix(dx, J, theta)
    primitive_true = dgp(β, umat_true, F, dx, J)
    v_true = vfi(primitive_true)
    prob_true = prob_from_dgp(primitive_true, v_true)
    ax.plot(range(dx), prob_true[:, 0], linewidth=2, label='True', marker='o', zorder=1)

    data = estimates_df[estimates_df['theta'].apply(lambda x: x == theta)].copy()
    prob_expanded = pd.DataFrame(data['prob'].tolist(),
                                 columns=[f'state_{i}' for i in range(dx)])
    data = pd.concat([data.drop('prob', axis=1).reset_index(drop=True),
                       prob_expanded.reset_index(drop=True)], axis=1)

    data = data.melt(
        id_vars=['theta', 'method', 'time', 'estimate'],
        value_vars=[f'state_{i}' for i in range(dx)],
        var_name='state',
        value_name='prob'
    )
    data['state'] = data['state'].str.replace('state_', '').astype(int)

    style = {
        "nfxp": {"linestyle": "-",  "marker": "o", "zorder": 4, "alpha": 0.5},
        "npl":  {"linestyle": "--", "marker": "s", "zorder": 3, "alpha": 0.7},
        "two step": {"linestyle": "-.", "marker": "D", "zorder": 2, "alpha": 0.7},
        "mpec": {"linestyle": ":", "marker": "^", "zorder": 5, "alpha": 0.7},
    }

    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        ax.plot(method_data['state'], method_data['prob'],
                label=method, linewidth=2, **style[method])

    ax.set_xlabel('State')
    ax.set_ylabel('Probability of Maintain')
    ax.set_title(f'Estimated vs True CCPs for θ = {theta}')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'outputs/ex1_4_plots_{idx}.png', dpi=150)
    plt.close()
    print(f"Saved outputs/ex1_4_plots_{idx}.png")

# --- Summary tables ---
estimates_df['pct_diff'] = estimates_df.apply(
    lambda r: np.abs(np.array(r['estimate']) - np.array(r['theta'])) / np.abs(np.array(r['theta'])), axis=1)
print("\nMean percentage error by method:")
print(estimates_df.groupby('method')['pct_diff'].mean())
print("\nMean execution time (seconds) by method:")
print(estimates_df.groupby('method')['time'].mean())
