'''
Exercise 1: Ebay Auction Simulation
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats 
from scipy.stats import norm, truncnorm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import math
'''
Part 2: Simulate values and report summary stats
'''
# Simulate 1000 Auctions w/ 10 bidders each where b_i \sim N(0,1)
np.random.seed(2115)
auction = np.repeat(range(1,1001),10)
bidders = np.array(list(range(1,11))*1000) 
bids = np.random.normal(0, 1, size=10000)

bid_df = pd.DataFrame({
    "auction": auction,
    "bidder": bidders,
    "bid": bids
})

# Compute mean of highest and second highest bids
bid_df = bid_df.sort_values(by=['auction', 'bid'], ascending=[True, False])
np.mean(bid_df.groupby('auction')['bid'].nth(0)) # highest bids
np.mean(bid_df.groupby('auction')['bid'].nth(1)) # second highest bids

bid_df = bid_df.sort_values(by=['auction', 'bidder'], ascending=[True, True])

'''
Part 4: Simulate ascending auction and compute who enters 
'''
# Compute prices faced by each bidder in ascending auction
bid_df = bid_df.rename(columns = {'bid':'value'}) # rename bids to value to prevent confusion
bid_df['p_ascending'] = np.repeat(-np.inf,10000)
bid_df['submitted_bids'] = np.repeat(-np.inf,10000)

bid_df.loc[bid_df.bidder < 3,'submitted_bids'] = bid_df['value']

# Compute prices and bids using a loop

for i in bid_df['auction'].unique():
    for j in range(3,11):
        index = (i-1)* 10 + j-1
        bid_df.iloc[index,3] = np.sort(np.array(bid_df.loc[bid_df.auction == i,'submitted_bids']))[::-1][1] # replace price with second highest of submitted bids
        bid_df.iloc[index,4] = np.where(bid_df.iloc[index,3] < bid_df.iloc[index,2],bid_df.iloc[index,2],bid_df.iloc[index,4] ) # submit bid if value greater than price

# Compute number of bidders and prices paid
bid_df['entered_auction'] = bid_df['submitted_bids'] > -np.inf
bid_df['entered_auction'] = bid_df['entered_auction'].astype(int)       

np.mean(bid_df.groupby('auction')['entered_auction'].sum())

np.mean(bid_df.groupby('auction')['p_ascending'].max())

# Histogram of submitted bids

bid_df.loc[bid_df.submitted_bids == -np.inf,'submitted_bids'] = np.nan
plt.hist(bid_df.submitted_bids,bins=50,color='skyblue',edgecolor='black')
plt.xlabel('Bids')
plt.ylabel('Frequency')
plt.title('Submitted Bids, Ascending Auction')
plt.show()


'''
Part 5: Mean of bids within each auction and variance across auctions
'''
mean_bids = bid_df.groupby('auction')['submitted_bids'].mean()
np.var(mean_bids)

# t-test
stats.ttest_1samp(mean_bids, 0.0)


'''
Part 6: Use Song (2004) to estimate bid distribution
Assume Normal Distribution
Use first-second order statistic, first-third order statistics
'''
# Prepare Data

highest_bids = np.array(bid_df.groupby('auction')['submitted_bids'].max())
bid_df = bid_df.sort_values(by=['auction', 'submitted_bids'], ascending=[True, False])
second_highest_bids =  np.array(bid_df.groupby('auction')['submitted_bids'].nth(1))
third_highest_bids =  np.array(bid_df.groupby('auction')['submitted_bids'].nth(2))
auction = np.array(range(1,1001))

estimation_df = pd.DataFrame({
    "auction": auction,
    "first_ord": highest_bids,
    "second_ord": second_highest_bids,
    "third_ord": third_highest_bids
})

# Song (2004) Approach, First-Second highest order statistic
## Function thatcomputes log likelihood given \sigma \mu

a = estimation_df['second_ord'].min()
def ll_first_sec(params):
    mu, sigma = params
    a_2 = (a-mu)/sigma
    if sigma <= 0:
        return np.inf
    df = estimation_df.drop(columns = ['third_ord'])
    df = df.dropna()
    num = truncnorm.pdf(df['first_ord'].values,a=a_2,b=b, loc = mu, scale = sigma)
    denom = 1-truncnorm.cdf(df['second_ord'].values,a=a_2,b=b,loc = mu, scale = sigma)
    ll = -np.mean(np.log(num/denom))
    return ll

est = minimize(ll_first_sec, x0=[1,2], bounds=[(None,None),(1e-6,None)], method="L-BFGS-B")

est.x
a_2 = (a-est.x[0])/est.x[1]

# Plot values and estimated histogram
truncated_bids_1  = [x for x in bid_df.submitted_bids if x> a]
truncated_values_1  = [x for x in bid_df.value if x> a]
plt.hist(truncated_bids_1,bins=50,density=True,color='skyblue',edgecolor='black', alpha=0.4,label="Bids")
plt.hist(truncated_values_1,bins=50,density=True,color='forestgreen',edgecolor='black', alpha=0.4,label="Values")
grid = np.linspace(bid_df.value.min(), bid_df.value.max(), 400)
plt.plot(grid, truncnorm.pdf(grid,a=a_2,b=b,loc=est.x[0], scale=est.x[1]), linewidth=2)
plt.xlabel('Valuations')
plt.ylabel('Frequency')
plt.title('Truncated Values and Est. Distribution, 1st & 2nd Order Method ')
plt.legend()
plt.show()

# Song (2004) Approach, First-Third highest order statistic
## Function that computes log likelihood given \sigma \mu
a_3 = estimation_df['third_ord'].min()
def ll_first_third(params):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    df = estimation_df.drop(columns = ['second_ord'])
    df = df.dropna()
    num_1 = 2 * truncnorm.pdf(df['first_ord'].values,a=a_3,b=b,loc = mu, scale = sigma)
    num_2 = truncnorm.cdf(df['first_ord'].values,a=a_3,b=b,loc = mu, scale = sigma)-truncnorm.cdf(df['third_ord'].values,a=a_3,b=b,loc = mu, scale = sigma)
    denom = (1-truncnorm.cdf(df['third_ord'].values,a=a_3,b=b,loc = mu, scale = sigma)) ** 2
    ll = -np.mean(np.log(num_1*num_2/denom))
    return ll

est_2 = minimize(ll_first_third, x0=[2,2], bounds=[(None,None),(1e-6,None)], method="L-BFGS-B")

est_2.x

# Plot values and estimated histogram
plt.hist(bid_df.submitted_bids,bins=50,density=True,color='skyblue',edgecolor='black', alpha=0.4,label="Bids")
plt.hist(bid_df.value,bins=50,density=True,color='forestgreen',edgecolor='black', alpha=0.4,label="Values")
grid = np.linspace(bid_df.value.min(), bid_df.value.max(), 400)
plt.plot(grid, truncnorm.pdf(grid,a=a_3,b=b, loc=est_2.x[0], scale=est_2.x[1]), linewidth=2)
plt.xlabel('Valuations')
plt.ylabel('Frequency')
plt.title('Values w Estimated Normal Overlay, First and Third Order')
plt.legend()
plt.show()


