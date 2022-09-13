'''
Bull Lite vault simulator:
'''


# %% Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 

# %% Params

# Price params
S0 = 100
mu = 0
default_sigma = 0.8/np.sqrt(365) # Specify vol per period
T = 365
# Uni Fee Yield
uniFeeYield = 0.0002580686267
minCR = 1.75
maxCR = 3

# %%

def liteGBM(S0, mu, sigma, T, dt=1):
    N = round(T/dt)
    t = np.linspace(0, T, N)
    rng = np.random.default_rng()
    W = rng.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt)
    S =  S0*np.exp((mu-0.5*sigma**2)*t + sigma*W)
    return np.insert(S,0,S0)

# %%
def gbmTest(dt=1):
    prices = liteGBM(S0=S0, mu=mu, sigma=default_sigma, T=T, dt=dt)
    log_returns=np.log(prices[1:len(prices)]/prices[0:len(prices)-1])
    stdev = np.std(log_returns)
    sum_squared_returns=(log_returns**2).sum()
    vol=np.sqrt(sum_squared_returns/len(log_returns))
    ann_factor = np.sqrt(365/dt)
    return stdev*ann_factor, vol*ann_factor, prices[round(T/dt)]

# %%
def iterateGbmTest(iterations=100000):
    sum_vol=0
    sum_spot=0
    sum_vol_no_mean=0
    temp_vol_no_mean=0
    for i in range(0,iterations):
        temp_vol, temp_vol_no_mean, temp_spot=gbmTest()
        sum_vol+=temp_vol
        sum_vol_no_mean+=temp_vol_no_mean
        sum_spot+=temp_spot
    return sum_vol/iterations, sum_vol_no_mean/iterations, sum_spot/iterations

# %%
iterateGbmTest()

# Bull Lite Params
l = 0.3 # % to LP
m = 1 # oSQTH LP to mint ratio

# %% Simulator
def simBullLite(sigma):
    #sigma = 0 # zero vol checks
    S = liteGBM(S0=S0,mu=mu,sigma=sigma,T=T)
    f = sigma**2 # Funding

    z = pd.DataFrame({'S':S})
    # Empty cols
    z['ethReturn'] = np.nan
    z['bullReturn'] = np.nan
    z['bullCumulativeReturn'] = np.nan

    # l: % to LP: 0.2
    # m: oSQTH LP to mint ratio: 3
    # r: ETH return -> ln(S1) - ln(S0) -> what continuously compounded rate would get me from 100 to 105
    # f: funding -> sigma^2

    for i, row in z.iterrows():
        if i == 0:
            z['ethReturn'][i] = 0
        else:
            z['ethReturn'][i] = np.log(z['S'][i]) - np.log(z['S'][i - 1])
        r = z['ethReturn'][i]
        z['bullReturn'][i] = (1+r)*((1-l)/l + 2*np.sqrt(1+r)*np.sqrt(1-f)+m -(1+m)*(1+r)*(1-f))/((1-l)/l+1) - 1 + uniFeeYield
        if i == 0:
            z['bullCumulativeReturn'][i] = z['bullReturn'][i]
        else:
            z['bullCumulativeReturn'][i] = z['bullReturn'][i] + z['bullCumulativeReturn'][i - 1]
    return z

# %%
def simBullLiteSigmas(sigmas):
    for i in range(sigmas):
        result = simBullLite(sigmas[i])
        filepath = Path('/Users/daryakaviani/bullResult' + i + '.csv') 
        filepath.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(filepath) 

def getDelta(l, m):
    return ((1-l)/l + 3 +m- 2*(1+m))/((1-l)/l+1)

def getCR(l, m, vol, ethDeposited=100):
    ethPrice = 2000
    normFactor = 0.5
    ethInVault = (1-l)*ethDeposited
    ethInLP = l*ethDeposited
    squeethPrice = ethPrice*normFactor*np.exp(vol**2 *17.5/365)/10000
    liquidity = ethInLP/np.sqrt(squeethPrice)
    oSQTHInLP = liquidity/np.sqrt(squeethPrice)
    oSQTHToSell = m*oSQTHInLP
    proceedsInETH = oSQTHToSell*squeethPrice

    return (ethInVault+ethInLP+proceedsInETH+oSQTHInLP*ethPrice*normFactor/10000)/((oSQTHInLP+oSQTHToSell)*ethPrice*normFactor/10000)

def simBullLiteSigmasAverage(count, sigmas):
    bullReturnSum = {}
    bullReturnAvg = {}
    for sigma in sigmas:
        bullReturnSum[sigma] = 0
        for i in range(count):
            bullReturnSum[sigma] += simBullLite(sigma)['bullCumulativeReturn'][T]
        bullReturnAvg[sigma] = bullReturnSum[sigma]/count
    return bullReturnAvg

def generateLM():
    lm = pd.DataFrame()
    lm['l'] = np.nan
    lm['m'] = np.nan
    lm['delta'] = np.nan
    lm['CR'] = np.nan
    for l in np.arange(0.1, 1.1, 0.1):
        for m in np.arange(1, 11, 1):
            delta = getDelta(l, m)
            cr = getCR(l, m, 0.8)
            if cr >= minCR and cr <= maxCR and delta >= 0.2 and delta <= 1:
                row = {'l': l, 'm': m, 'delta': delta, 'CR': cr}
                lm = lm.append(row, ignore_index = True)
    filepath = Path('/Users/daryakaviani/bullSim/lm.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lm.to_csv(filepath) 

# %%
# simBullLiteSigmas([sigma/np.sqrt(365) for sigma in np.arange(0.4, 1.3, 0.1)])
# print(getDelta(.2, 3))
# print(getCR(0.2, 3, 0.8))
print(simBullLiteSigmasAverage(1000, [sigma/np.sqrt(365) for sigma in np.arange(0.4, 1.3, 0.1)]))
# %%
