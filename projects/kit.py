import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

apac = None
ind_lookup = None
cc_lookup = None

def get_data():
    global apac
    global ind_lookup
    global cc_lookup
    
    apac = pd.read_csv('apac-finstruct.csv',header=0)
    apac.rename(columns={'WB COUNTRY CODE':'cc', 'WB REGION':'reg', 
                         'WB INCOME GROUP':'inc_g', 'YEAR':'year'}, 
                inplace=True)   
    
    c_c = [apac['cc'].unique(), apac['COUNTRY'].unique()]
    cc_lookup = dict(zip(*c_c))
    apac.drop(['COUNTRY', 'reg'], axis=1, inplace=True)
    apac.set_index(['cc','year'],inplace=True)
    arrays = [np.arange(len(apac.columns.values)-1), apac.columns[1:]]
    ind_lookup = dict(zip(*arrays))
    
    # countries in APAC
    countries = apac.index.get_level_values(0).unique().values
    inc_g = [apac.loc[c]['inc_g'].values[0] for c in countries]
    apac.drop('inc_g', axis=1, inplace=True)
    
    # income group of each country
    ig_c = pd.Series(data=inc_g, index=countries)
    c_ig = pd.Series(data=countries, index=inc_g)

    # indicators dataframe
    sze = np.arange(1,12)
    eff = np.arange(12,18)
    sta = [18]
    cim = np.arange(19,27)
    fgl = np.arange(27,32)

    # create dataframe for indicators and groups of indicators
    groups = ["sze"]*len(sze) + ['eff']*len(eff) + ['sta']*len(sta) + ['cim']*len(cim) + ['fgl']*len(fgl)
    ig = pd.DataFrame.from_dict(ind_lookup, orient='index', columns=['indicator'])
    ig['group'] = groups

    # remove less important indicators
    ig.drop([4,6,8,26,27],inplace=True)
    
    return {'apac': apac, 'cc_lookup': cc_lookup, 'ind_lookup': ind_lookup, 
            'ig_c': ig_c, 'c_ig': c_ig, 'countries': countries,
           'ig': ig}

"""
plot an indicator of a list of countries
"""
def plot_si_mc(ind, c_list, figsize=(10,5), year_fr=1960, year_to=2017):
    global apac
    global ind_lookup
    get_data()
    plt.figure(figsize=figsize)

    colors = plt.cm.jet(np.random.rand(len(c_list)))

    for c in c_list:
        v = apac.loc[c].iloc[:,ind].loc[year_fr:year_to]
        plt.plot(v, label=c, marker='.')

    plt.title(ind_lookup[ind])
    plt.legend()
    
"""
plot multiple indicators for a country
"""
def plot_mi_sc(ind_list, country, figsize=(10,5), year_fr=1960, year_to=2017):
    global apac
    global cc_lookup
    global ind_lookup
    get_data()
    
    plt.figure(figsize=figsize)      
    
    for i in ind_list:
        v = apac.loc[country].iloc[:, i].loc[year_fr:year_to]
        plt.plot(v, label=ind_lookup[i], marker='.')
        
    plt.title(cc_lookup[country])
    plt.legend()
    
"""
plot multiple indicators for a maximum of 4 countries
"""

def plot_mi_mc(ind_list, c_list, figsize=(10,5)):
    global apac
    global cc_lookup
    global ind_lookup
    get_data()
    
    plt.figure(figsize=figsize)
    ls = ['-', '--', '-.', ':']
    cl = plt.cm.rainbow(np.random.randn(len(c_list)))
    title = ""
    
    for c in c_list:
        title += cc_lookup[c] + " - "
        c_cl = cl[c_list.index(c)]
        for i in ind_list:
            v = apac.loc[c].iloc[:, i]
            l = ind_lookup[i]
            s = ls[ind_list.index(i)]
            plt.plot(v, label=l+' / '+c, linestyle=s, color=c_cl)

    title = title[:-2]
    
    plt.title(title)
    plt.legend()
    
"""
plot an indicator of all three income groups
"""
def plot_si_incg(ind, figsize=(10,10), sharex=True, sharey=True):
    global apac
    global cc_lookup
    global ind_lookup
    get_data()
    
    g = ['High income', 'Upper middle income', 'Lower middle income']

    cl = plt.cm.jet(np.linspace(0,1,len(countries)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=sharex, sharey=sharey, figsize=figsize)
    
    for c in c_ig[g[0]]:
        v = apac.loc[c].iloc[:,ind]
        ax1.plot(v, label=c, marker='.', color=cl[np.where(countries==c)[0][0]])

    for c in c_ig[g[1]]:
        v = apac.loc[c].iloc[:,ind]
        ax2.plot(v, label=c, marker='.', color=cl[np.where(countries==c)[0][0]])

    for c in c_ig[g[2]]:
        v = apac.loc[c].iloc[:,ind]
        ax3.plot(v, label=c, marker='.', color=cl[np.where(countries==c)[0][0]])    
    
    fig.suptitle(ind_lookup[ind])
    ax1.set_title(g[0])
    ax2.set_title(g[1])    
    ax3.set_title(g[2])
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    plt.show()
    
"""
plot multiple indicators for a single country in multiple subplots
maximum 3 indicators per subplots
"""

def plot_mi_sc_ms(ind_list, c, figsize=(15,10)):
    global apac
    global cc_lookup
    global ind_lookup
    get_data()
    
    n = len(ind_list)
    n_subplots = n//3 + int((n%3)!= 0)
    cl = plt.cm.jet(np.linspace(0,1,n))
    
    fig, axes = plt.subplots(nrows=n_subplots, sharex=True, sharey = True, figsize=figsize)

    for i in ind_list:
        index = list(ind_list).index(i)
        r = index // 3 # subplot number
        if apac.loc[c].iloc[:,i].isna().all() == True:
            continue
        v = apac.loc[c].iloc[:,i]
        axes[r].plot(v, label=ind_lookup[i], marker='.', color=cl[index-1])
        axes[r].legend()    
    
    plt.suptitle(cc_lookup[c])

    plt.show()