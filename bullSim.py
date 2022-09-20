'''
Bull Lite vault simulator:
'''


# %% Imports
from dataclasses import dataclass
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from operator import add
# 

# %% Params

# Price params
S0 = 100
mu = 0
default_sigma = 0.8/np.sqrt(365) # Specify vol per period
historical_sigma = 0.86807856209

# Since May 1st
# squeeth_rv = 0.9883994462
# eth_rv = 1.038281927
# uni_iv = 1.032225007
# squeeth_iv =  1.083185131
# uniFeeYield = 0.0003535017148

# Since 7/13
# squeeth_rv = 1.01661402
# eth_rv = 1.017327489
# uni_iv = 1.074391249
# squeeth_iv =  0.994825332
# uniFeeYield = 0.0003679422084

# Full 200-day Squeeth period
# eth_rv = 0.9149563106
# uni_iv = 0.8694524784
# squeeth_iv = 1.069018607
# uniFeeYield = 0.0002977576981

# Fake Crab, 7/30
# eth_rv = 0.749532992
# uni_iv = 0.9700340845
# squeeth_iv = 0.9878215474
# uniFeeYield = 0.0003322034536

# 6 months
eth_rv = 0.8727682604
uni_iv = 0.8445048132
squeeth_iv = 0.9856595416
uniFeeYield = 0.0002892139771


tau = 17.5/365

T = 365
# Uni Fee Yield
# Constant slippage
slippage_cost = 0.01
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
def simBullLite(sigma, l, m, d = 1):
    # d is days between hedge
    #sigma = 0 # zero vol checks
    S = liteGBM(S0=S0,mu=mu,sigma=sigma,T=T)
    f = d*((squeeth_iv/np.sqrt(T))**2) # Funding
    f_daily = (squeeth_iv/np.sqrt(T))**2

    # l: % to LP
    # m: oSQTH LP to mint ratio
    # r: ETH return -> ln(S1) - ln(S0) -> what continuously compounded rate would get me from 100 to 105
    # f: funding -> sigma^2

    r = S[d:]/S[0:-d] - 1 # repeated d-day interval
    r_interval = np.array([r[i] for i in range(len(r)) if i % d == 0]) # d-day interval (return results only)
    r_daily = S[1:]/S[0:-1] - 1 # daily interval
    bull_returns = (1+r_interval)*((1-l)/l + 2*np.sqrt(1+r_interval)*np.sqrt(1-f)*(1+(d*uniFeeYield))+m -(1+m)*(1+r_interval)*(1-f))/((1-l)/l+1) - 1
    trade_size = l*(-1*np.sqrt((1-f)*(1+r_interval)) - m*(bull_returns+1) + (1+m)*(1-f)*(1+r_interval))
    net_returns = bull_returns - np.abs(trade_size * slippage_cost)
    CRs = []
    for i in range(len(r_daily)):
        if i % d == 0:
            CRs.append(getCR(l, m, historical_sigma))
        else:
            next_CR = (((1-l)/l + np.sqrt(1-f_daily)*np.sqrt(1+r_daily[i]) + m + np.sqrt(1-f_daily)*np.sqrt(1+r_daily[i])/np.exp(squeeth_iv**2 *tau))/ (1/l +m +1/np.exp(squeeth_iv**2 * tau))) * CRs[i - 1]
            CRs.append(next_CR)
    return net_returns, np.sum(i <= 1.5 for i in CRs)/len(CRs)

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

def simBullLiteSigmasAverage(count):
    lm = generateLM()
    results = pd.DataFrame({'sigma':np.arange(0.4, 1.3, 0.1)})
    for i, row in lm.iterrows():
        results[(row['l'], row['m'])] = np.nan

    for i, row in results.iterrows():
        sigma = results['sigma'][i]

        for (colname, _) in results.iteritems():
            if colname != 'sigma':
                l, m = colname
                print(colname)
                bullReturnSum = 0
                for i in range(count):
                    bullReturnSum += sum(simBullLite(sigma/np.sqrt(T), l, m, 3))
                row[(l, m)] = bullReturnSum/count
    slopes, y_intercepts, ls, ms, deltas, historical = [], [], [], [], [], []
    for (colname, values) in results.iteritems():
        if colname != 'sigma':
            l, m = colname
            a, b = np.polyfit(results['sigma'], values, 1)
            slopes.append(a)
            y_intercepts.append(b)
            ls.append(l)
            ms.append(m)
            deltas.append(getDelta(l, m))
            historical.append((a*historical_sigma) + b)
    x_intercepts = -1*np.divide(y_intercepts,slopes)
    trendlines = pd.DataFrame({'slopes':slopes,
                               'y_intercepts':y_intercepts,
                               'x_intercepts':x_intercepts,
                                'l': ls,
                                'm': ms,
                                'delta': deltas,
                                'historical': historical
                              })

    filepath = Path('/Users/daryakaviani/bullSim/results.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath) 

    filepath = Path('/Users/daryakaviani/bullSim/trendline.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    trendlines.to_csv(filepath) 

def simBullLiteSigmasTrue(count):
    results = pd.DataFrame({'sigma':np.arange(0.4, 1.3, 0.1)})
    results[(1, 0)] = np.nan

    for i, row in results.iterrows():
        sigma = results['sigma'][i]
        l, m = 1, 0
        bullReturnSum = 0
        for i in range(count):
            bullReturnSum += sum(simBullLite(sigma/np.sqrt(T), l, m))
        row[(l, m)] = bullReturnSum/count
    slopes, y_intercepts, ls, ms, deltas, historical = [], [], [], [], [], []
    for (colname, values) in results.iteritems():
        if colname != 'sigma':
            l, m = colname
            a, b = np.polyfit(results['sigma'], values, 1)
            slopes.append(a)
            y_intercepts.append(b)
            ls.append(l)
            ms.append(m)
            deltas.append(getDelta(l, m))
            historical.append((a*historical_sigma) + b)
    x_intercepts = -1*np.divide(y_intercepts,slopes)
    trendlines = pd.DataFrame({'slopes':slopes,
                               'y_intercepts':y_intercepts,
                               'x_intercepts':x_intercepts,
                                'l': ls,
                                'm': ms,
                                'delta': deltas,
                                'historical': historical
                              })

    filepath = Path('/Users/daryakaviani/bullSim/results.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath) 

    filepath = Path('/Users/daryakaviani/bullSim/trendline.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    trendlines.to_csv(filepath) 

def simBullLiteSigmasTrueHistorical(count):
    l, m = 1, 0
    results = pd.DataFrame()
    results[(l, m)] = np.nan

    returns = []
    bullReturnSum = 0
    for i in range(count):
        bullReturns = simBullLite(eth_rv/np.sqrt(T), l , m)
        bullReturnSum += sum(bullReturns)
    returns.append(bullReturnSum/count)
    
    sim = pd.DataFrame({'returns':returns})

    filepath = Path('/Users/daryakaviani/bullSim/sim.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(filepath) 

def simBullLiteSigmasAverageHistorical(count, d = 1):
    lm = generateLM()
    results = pd.DataFrame()
    for i, row in lm.iterrows():
        results[(row['l'], row['m'])] = np.nan

    returns = []
    liqs = []
    ls = []
    ms = []
    for (colname, _) in results.iteritems():
        l, m = colname
        print(colname)
        bullReturnSum = 0
        # CRSum = np.zeros(T)
        CRPercentSum = 0
        for i in range(count):
            bullReturn, CR = simBullLite(eth_rv/np.sqrt(T), l, m, d)
            bullReturnSum += sum(bullReturn)
            # CRSum = list(map(add, CRSum, CRs))
            CRPercentSum += CR
        # CRAvg = CRPercentSum/count
        # liquidations = np.sum(i <= 1.5 for i in CRAvg)
        returns.append(bullReturnSum/count)
        liqs.append(CRPercentSum/count)
        ls.append(l)
        ms.append(m)
    
    sim = pd.DataFrame({'returns':returns,
                        'l':ls,
                        'm':ms,
                        'liquidations': liqs,
                        'CR': [getCR(ls[i], ms[i], eth_rv) for i in range(len(ls))],
                        'delta': [getDelta(ls[i], ms[i]) for i in range(len(ls))]})

    filepath = Path('/Users/daryakaviani/bullSim/simple.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(filepath) 

def generateLM():
    lm = pd.DataFrame()
    lm['l'] = np.nan
    lm['m'] = np.nan
    lm['delta'] = np.nan
    lm['CR'] = np.nan
    for l in np.arange(0.2, 1.1, 0.1):
        for m in np.arange(0, 5, 0.1):
            delta = getDelta(l, m)
            cr = getCR(l, m, eth_rv)
            print(cr, delta)
            if cr >= 1.75 and cr <= 3 and delta >= 0.5 and delta <= 1:
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
# print(simBullLiteSigmasAverage(1000))
print(simBullLiteSigmasAverageHistorical(10000, 3))
# print(getDelta(1, 0))
# print(getCR(1, 0, 0.89))
# print(simBullLiteSigmasTrueHistorical(100000))

# %%
# generateLM()
