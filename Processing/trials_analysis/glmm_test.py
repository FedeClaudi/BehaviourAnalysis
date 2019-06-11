# %%
from Utilities.imports import *
from theano import tensor as tt
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices 
import pymc3 as pm


#%%
# ! Get the data

# ? get the maze params
parfile = "Processing/trials_analysis/params.yml"
params = load_yaml(parfile)

# convert them to df
temp = dict(
    name = [],
    length = [],
    rLen = [],
    iLen = [],
    uLen = [],
    theta_start = [],
    theta_tot = []
)

for exp, pps in params.items():
    for name, ps in pps.items():
        good_name = exp+"_"+name
        temp['name'].append(good_name)
        for i, (dict_list, val) in enumerate(zip(list(temp.values())[1:], ps)):

            dict_list.append(val)

params = pd.DataFrame.from_dict(temp)

#%%
# ? get the Hierarchical Bayes data
f = "Modelling/bayesian/hierarchical_v2.pkl"
data = pd.read_pickle(f)
cols_to_drop = [c for c in data.columns if "asym_prior" not in c and "sym_prior" not in c]
data = data.drop(cols_to_drop, axis=1)

pRs = data.mean()


#%%
# ? Put all the data together

temp = dict(
    maze = [],
    pR = [],
    rLen = [],
    delta_theta_start = [],
    delta_theta_tot = []
)

for name, pR in zip(pRs.index, pRs):
    if "asym" in name:
        temp['maze'].append("asym")
        left = params.iloc[0]
        right = params.iloc[1]
    else:
        temp['maze'].append("sym")
        left = params.iloc[6]
        right = params.iloc[7]

    temp['pR'].append(pR)
    temp['rLen'].append(right.rLen)
    temp['delta_theta_start'].append(0)  # ! in this case it's always the same
    temp['delta_theta_tot'].append(right.theta_tot - left.theta_tot)

data = pd.DataFrame.from_dict(temp).drop("delta_theta_start", axis=1)




#%%
# ! set up for the GLMM

formula = "pR ~ rLen + delta_theta_tot"
Y,X = dmatrices(formula, data=data, return_type='matrix')
Terms  = X.design_info.column_names

X      = np.asarray(X) # fixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)

# ? For now no mixed effects


#%%

# ! Estimate parameters with PyMC3

beta0     = np.linalg.lstsq(X,Y)  # Solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2. 
fixedpred = np.argmax(X,axis=1)

with pm.Model() as glmm1:
    # Fixed effect
    beta = pm.Normal('beta', mu=0., sd=100., shape=(nfixed[1]))

    # random effect
    # s    = pm.HalfCauchy('s', 10.)
    # b    = pm.Normal('b', mu=0., sd=s, shape=(nrandm[1]))
    eps  = pm.HalfCauchy('eps', 5.)
    
    mu_est = pm.Deterministic('mu_est', tt.dot(X,beta)) # +tt.dot(Z,b))
    RT = pm.Normal('RT', mu_est, eps, observed=Y)
    
    trace = pm.sample(1000, tune=1000)
    



#%%
pm.traceplot(trace)
t = pm.trace_to_dataframe(trace)
#%%
