import statsmodels.api as sm

data = sm.datasets.scotland.load()  
# dataset with 32 entries, 1 predicted variable and 7 predictor variables

data.exog = sm.add_constant(data.exog) # add a column of ones to the predictors variables
# this way when doing the matrix multiplication to define the model you can estimate the slope
# see: https://matthew-brett.github.io/teaching/glm_intro.html
#%%

# Instantiate a gamma family model with the default link function.
gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())

#%%
