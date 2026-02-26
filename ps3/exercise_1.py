'''
Exercise 1: Ebay Auction Simulation
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
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

# Song (2004) Approach, First-Second highest order statistic