
#%%
import numpy as np
#import pandas as pd
from theano import tensor as tt
get_ipython().magic('pylab inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('qtconsole --colors=linux')


import pandas as pd


# ! Load and clean DATA
Tbl_beh  = pd.read_csv('Utilities/maths/behavioral_data.txt', delimiter='\t')
Tbl_beh["subj"]  = Tbl_beh["subj"].astype('category')

#%% Compute the conditional mean of dataset
condi_sel = ['subj', 'chimera', 'identity', 'orientation']
tblmean = pd.DataFrame({'Nt' : Tbl_beh.groupby( condi_sel ).size()}).reset_index()
tblmean["subj"] = tblmean["subj"].astype('category')
tblmean['rt']  = np.asarray(Tbl_beh.groupby(condi_sel)['rt'].mean())
tblmean['ACC'] = np.asarray(Tbl_beh.groupby(condi_sel)['acc'].mean())
tblmean['group']= np.asarray(Tbl_beh.groupby(condi_sel)['group'].all())

tblmean['cbcond']  = pd.Series(tblmean['chimera'] + '-' + tblmean['identity'], 
                index=tblmean.index)

tbltest = tblmean

#%%
import statsmodels.formula.api as smf
from patsy import dmatrices  # creates two matrices given data and using a formula
# by default the first matrix is Y and the second is X

# ! Create matrixes
formula = "rt ~ group*orientation*identity"  # ? a * b is short-hand for a + b + a:b

# Generate Design Matrix for later use
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)

#%%
# ! PMC3 model
beta0     = np.linalg.lstsq(X,Y)  # Solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2. 

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)

con  = tt.constant(fixedpred)
sbj  = tt.constant(randmpred)

#%%
import pymc3 as pm
with pm.Model() as glmm1:
    # Fixed effect
    beta = pm.Normal('beta', mu=0., sd=100., shape=(nfixed[1]))
    # random effect
    s    = pm.HalfCauchy('s', 10.)
    b    = pm.Normal('b', mu=0., sd=s, shape=(nrandm[1]))
    eps  = pm.HalfCauchy('eps', 5.)
    
    #mu_est = pm.Deterministic('mu_est',beta[con] + b[sbj])
    mu_est = pm.Deterministic('mu_est', tt.dot(X,beta)+tt.dot(Z,b))
    RT = pm.Normal('RT', mu_est, eps, observed=Y)
    
    trace = pm.sample(1000, tune=1000)
    
#%% 
# ! plot
pm.traceplot(trace,varnames=['beta','b','s']) # 
plt.show()


#%%
fixed_pymc = pm.summary(trace, varnames=['beta'])
randm_pymc = pm.summary(trace, varnames=['b'])

b = np.asarray(fixed_pymc['mean'])
q = np.asarray(randm_pymc['mean'])

#%%
f, ax = plt.subplots(figsize=(8,6))
ax.plot(Y,'o',color='w',label = 'Observed', alpha=.25)
fitted = np.dot(X,b).flatten() + np.dot(Z,q).flatten()
ax.plot(fitted,lw=1,label = "test", alpha=.5)
#%%
