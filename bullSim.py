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
lmDictionary = [(0.7,1), (0.6,1), (0.5,1), (0.618034,0.689808), (0.4,1), (0.3,1), (0.618034,0.48541)]

# %% Simulator
def simBullLite(sigma, l, m):
    #sigma = 0 # zero vol checks
    S = liteGBM(S0=S0,mu=mu,sigma=sigma,T=T)
    f = sigma**2 # Funding

    # l: % to LP: 0.2
    # m: oSQTH LP to mint ratio: 3
    # r: ETH return -> ln(S1) - ln(S0) -> what continuously compounded rate would get me from 100 to 105
    # f: funding -> sigma^2

    r = S[1:]/S[0:-1] - 1
    returns = (1+r)*((1-l)/l + 2*np.sqrt(1+r)*np.sqrt(1-f)+m -(1+m)*(1+r)*(1-f))/((1-l)/l+1) - 1 + uniFeeYield
    return returns

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

# l = ethInVault/ethInLP
# m = oSQTHToSell/oSQTHInLP

def simBullLiteSigmasAverage(count, sigmas):
    lm = generateLM()
    results = pd.DataFrame({'sigma':np.arange(0.4, 1.3, 0.1)})
    for i, row in lm.iterrows():
        results[(row['l'], row['m'])] = np.nan

    for i, row in results.iterrows():
        sigma = results['sigma'][i]

        for (colname, _) in results.iteritems():
            if colname != 'sigma':
                print("colname", colname)
                l, m = colname
                bullReturnSum = 0
                for i in range(count):
                    bullReturnSum += sum(simBullLite(sigma/np.sqrt(T), l, m))
                row[(l, m)] = bullReturnSum/count
                print(sigma, bullReturnSum/count)

    filepath = Path('/Users/daryakaviani/bullSim/results.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath) 

def generateLM():
    lm = pd.DataFrame()
    lm['l'] = np.nan
    lm['m'] = np.nan
    lm['delta'] = np.nan
    lm['CR'] = np.nan
    for l in np.arange(0.01, 1.1, 0.01):
        for m in np.arange(1, 11, .05):
            delta = getDelta(l, m)
            cr = getCR(l, m, 0.8)
            if cr >= minCR and cr <= maxCR and delta >= 0.2 and delta <= 1:
                row = {'l': l, 'm': m, 'delta': delta, 'CR': cr}
                lm = lm.append(row, ignore_index = True)
    filepath = Path('/Users/daryakaviani/bullSim/lm.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lm.to_csv(filepath) 
    return lm

# %%
# simBullLiteSigmas([sigma/np.sqrt(365) for sigma in np.arange(0.4, 1.3, 0.1)])
# print(getDelta(.618034, .48541))
# print(getCR(.618034, .689808, .8))
# print(getDelta(.618034, .689808))
print(simBullLiteSigmasAverage(1000, [sigma/np.sqrt(365) for sigma in np.arange(0.4, 1.3, 0.1)]))
# %%
# generateLM()
