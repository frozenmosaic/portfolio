import pandas as pd
import numpy as np
from scipy.stats import norm

def get_csv(filename, skiprows=0):
    data = pd.read_csv(filename+'.csv',header=0,index_col=0,parse_dates=True,skiprows=skiprows)
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    data.columns = data.columns.str.strip()
    return data

def rets_fr_prices(prices):    
    """
    computes monthly returns based on prices
    returns: DataFrame
    """
    return prices.pct_change().round(3)

def cum_ann_rets():
    """
    computes annualized returns based on monthly returns
    returns: DataFrame
    """
    y = np.arange(1926,2021,1)
    mra = pd.Series(index=list(y), dtype='float') # mra is short for market_return_annualized

    for i in range(len(y)):
        mra[y[i]] = (1+mkt_ret[str(y[i])]).prod()-1
    
def ann_rets(r, periods_per_year):        
    """
    r: Series or DataFrame of prices
    returns: float
    """
    compounded_growth = (1+r).prod() # monthly
    n_periods = r.shape[0] # total number of data periods
    return compounded_growth**(periods_per_year/n_periods)-1 # formula: (end value/start value)*(1/num_compounding_periods_per_year)

def ann_vol(r, periods_per_year):
    """
    calculates the annualized volatility of a Series or DataFrame
    """
    return r.std()*(periods_per_year**0.5) # formula: period_vol * (squareroot of periods_per_year)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    computes annualized Sharpe ratio
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period # period excess return
    ann_ex_ret = ann_rets(excess_ret, periods_per_year) # annualized excess return
    ann_v = ann_vol(r, periods_per_year)
    return ann_ex_ret / ann_v

def drawdown(returns_series: pd.Series):
    """
    takes a time series of an asset returns
    computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    drawdowns in percentage
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })

def portfolio_return(weights, returns):
    """
    calculates portfolio returns
    """
    return weights.T @ returns
    
def portfolio_vol(weights,covmat):
    """
    calculates portfolio volatility using covariance matrix
    """
    return (weights.T @ covmat @ weights)**0.5 # stdev = squareroot of variance

from scipy.optimize import minimize # optimizer similar to Excel's solver
def minimize_vol(target_return, er, cov):
    """
    target return -> a weight vector W that yields the minimize volatility
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) # intial guess
    bounds = ((0.0, 1.0),)*n # set the bound for each of the weights
    # first constraint: return must be equal to target_return
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    # second constraint
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method="SLSQP", # quadratic
                       options={'disp':False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    generates a list of weights to run the optimizer on
    """
    target_returns = np.linspace(er.min(), er.max(), n_points) # generate target returns between min and max returns
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights
    

def plot_ef_simple(n_points, er, cov, style=".-"):
    """
    plot the n-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov) # generate list of optimal weights
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)

def plot_ef2(n_points, er, cov, style=".-"):
    """
    plot the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)


def semideviation(r):
    """
    returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0 # boolean mask
    return r[is_negative].std(ddof=0)
    
def skewness(r):
    """
    alternative to scipy.states.skew()
    computes the skewness of the supplied Series or DataFrame
    returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population stdev, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return  exp / sigma_r**3

def kurtosis(r):
    """
    alternative to scipy.states.kurtosis()
    computes the kurtosis of the supplied Series or DataFrame
    returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use the population stdev, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return  exp / sigma_r**4

def is_normal(r, level=0.01): # default value in case there is no param value given
    """
    applies the Jarque-Bera test to determine if a Series is normal or not
    test is applied at the 1% level by default
    returns True if the hypothesis or normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level=5):
    """
    returns the historic VaR at a specified level
    i.e returns the number such that "level" percent of the returns 
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance (r, pd.Series):
        return -np.percentile(r, level)
    else: 
        raise TypeError("Expected r to be Series or DataFrame")
        
from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # compute z score assuming it is Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 - 
                (2*z**3 - 5*z)*(s**2)/36
            ) 
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level) # creates a mask
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else: 
        raise TypeError("expect a Series or DataFrame")
        
from scipy.optimize import minimize # optimizer similar to Excel's solver
def minimize_vol(target_return, er, cov, portfolio):
    """
    target return -> a weight vector W that yields the minimize volatility
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) # intial guess
    bounds = ((0.0, 1.0),)*n # set the bound for each of the weights
    # first constraint: return must be equal to target_return
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    # second constraint
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method="SLSQP", # quadratic
                       options={'disp':False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                      )
    w = results.x
    w = ['{:f}'.format(item) for item in w]
    w = pd.DataFrame(columns=['w'], data=w, index=portfolio)
    return w

def optimal_weights(n_points, er, cov):
    """
    generates a list of weights to run the optimizer on
    """
    target_returns = np.linspace(er.min(), er.max(), n_points) # generate target returns between min and max returns
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights    

def msr(riskfree_rate, er, cov, portfolio):
    """
    returns the weights of the portfolio that gives you the maximum Sharpe ratio
    given the risk-free rate, expected returns and covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) # intial guess
    bounds = ((0.0, 1.0),)*n # set the bound for each of the weights
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        returns the negative of the Sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol
    
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov,), method="SLSQP", # quadratic
                       options={'disp':False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    w = results.x
    w = ['{:f}'.format(item) for item in w]
    w = pd.DataFrame(columns=['w'], data=w, index=portfolio)
    return w

def gmv(cov, portfolio):
    """
    returns the weight of the Global Minimum Variance portfolio
    """
    n = cov.shape[0]
    w = msr(0, np.repeat(1,n), cov, portfolio)
    #w = ['{:f}'.format(item) for item in w]
    w = pd.DataFrame(columns=['w'], data=w, index=portfolio)
    return w

def plot_ef(n_points, er, cov, show_cml=False, style='.-', riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    plot the n-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov) # generate list of optimal weights
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_ew: # equally weighted portfolio
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        v_ew = portfolio_vol(w_ew,cov)
        # display the EW portfolio
        ax.plot([v_ew],[r_ew], color='goldenrod', marker='o', markersize=10)
    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        v_gmv = portfolio_vol(w_gmv,cov)
        # display the EW portfolio
        ax.plot([v_gmv],[r_gmv], color='midnightblue', marker='o', markersize=10)
        
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        v_msr = portfolio_vol(w_msr, cov)
        
        # add CML
        cml_x = [0,v_msr]
        cml_y = [riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color='green',marker="o",linestyle="dashed", markersize=12, linewidth=2)
    
    return ax