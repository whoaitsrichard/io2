'''
Exercise 2: Ebay Auction Empirical Exercise
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
Part 2: Summary Statistics
'''
# Read in Data
bids = pd.read_csv('data/bids.csv')
items = pd.read_csv('data/items.csv')
attributes = pd.read_csv('data/sparse_attributes.csv')

## Convert attributes to sparse matrix
wide_attributes = (attributes.pivot_table(index="i", columns="j", values="fill",
                     aggfunc="max", fill_value=0).sort_index())

wide_attributes = wide_attributes.reindex(range(1, 7001), fill_value=0)
bids['log_bid_value'] = np.log(bids['bid_value'])


## Plot Histograms of Mean Log Bid and Num Bids by Item
plt.hist(bids.groupby('item_num')['log_bid_value'].mean(),bins=50,color='skyblue',edgecolor='black')
plt.xlabel('Log Bid')
plt.ylabel('Frequency')
plt.title('Mean of Log Bids by Item')
plt.show()

counts = bids.groupby("item_num")["item_num"].count()
plt.hist(counts,bins=20,color='forestgreen',edgecolor='black')
plt.xlabel('Number of Bids')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Bids')
plt.xticks(range(int(counts.min()), int(counts.max()) + 2, 2))  # step=1
plt.show()
plt.show()