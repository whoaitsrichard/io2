import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# =============================================================================
# Exercise 3 — Entry model MLE estimation
# =============================================================================

print("=" * 60)
print("Exercise 3: Entry model MLE")
print("=" * 60)

# Read in Data
df = pd.read_csv('data/ps2_ex3.csv')
print(f"Max entrants in data: {max(df['n'])}")

def cond_prob_theta(data, theta):
    """Estimate probability given beta, phi, delta."""
    eps = 1e-8
    beta = theta[0]
    phi = theta[1]
    delta = theta[2]
    prob = pd.DataFrame(np.zeros((len(data['x']), int(max(data['n'])) + 1)))
    for i in range(int(max(data['n'])) + 1):
        if i == 0:
            prob.iloc[:, i] = norm.cdf(phi - beta * data['x'])
        elif i == max(data['n']):
            prob.iloc[:, i] = 1 - norm.cdf(phi - beta * data['x'] + delta * np.log(24))
        else:
            prob.iloc[:, i] = norm.cdf(phi - beta * data['x'] + delta * np.log(i + 1)) - norm.cdf(phi - beta * data['x'] + delta * np.log(i))

    prob = prob.clip(lower=eps, upper=1 - eps)
    return prob

def likelihood(data, prob):
    """The likelihood function with the given choice probability."""
    l = 0.
    for i in range(len(data['x'])):
        num_enter = data['n'].iloc[i]
        l += np.log(prob.iloc[i, num_enter])
    return -l / (len(data['x']) * max(data['n']))

def entry_mle(data, n_starts=20, seed=563):
    """MLE with simple multi-start global search."""
    rng = np.random.default_rng(seed)

    def ll(x):
        prob = cond_prob_theta(data, x)
        return likelihood(data, prob)

    best_res = None
    best_val = np.inf

    for _ in range(n_starts):
        x0 = np.array([1, 1, 1]) + rng.normal(scale=1.0, size=3)
        res = minimize(ll, x0)

        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    return best_res

out = entry_mle(df)
print(f"Estimated parameters (beta, phi, delta): {out.x}")

table = cond_prob_theta(df, out.x)
df['MLE_implied_num'] = table.idxmax(axis=1)
print(f"Fraction correctly predicted: {np.mean(df['MLE_implied_num'] == df['n']):.4f}")

# --- Plot ---

def plot_implied_n_vs_x(data, theta):
    x_grid = np.linspace(0, 18, 100)
    grid_df = pd.DataFrame({"x": x_grid, "n": np.full_like(x_grid, 24)})

    prob = cond_prob_theta(grid_df, theta)

    n_vals = np.arange(prob.shape[1])
    n_exp = prob.to_numpy() @ n_vals

    x_true = np.asarray(data["x"])
    n_true = np.asarray(data["n"])
    order_true = np.argsort(x_true)

    plt.figure()
    plt.scatter(x_true[order_true], n_true[order_true],
                s=12, alpha=0.4, label="Realized n")
    plt.plot(x_grid, n_exp, linewidth=2, label="Implied E[n | x]")

    plt.xlabel("x")
    plt.ylabel("Number of entrants")
    plt.title("Realized vs implied expected entrants")
    plt.xlim(0, 18)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/ex3_plot.png', dpi=150)
    plt.close()
    print("Saved outputs/ex3_plot.png")

plot_implied_n_vs_x(df, out.x)
