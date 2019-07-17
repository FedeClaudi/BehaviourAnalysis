import sys
sys.path.append('./')

import numpy as np
from scipy import misc, signal, stats

# ! FIT MATH FUNCTIONS
# ? Functions to pass to curve_fit
def polyfit(order, x, y):
	#  calculate polynomial
	z = np.polyfit(x, y, order)
	f = np.poly1d(z)
	return f

def sigmoid(x, a, b):
    y = 1 / (1 + np.exp(-b*(x-a)))
    return y

def half_sigmoid(x, a, b):
	chance = .5
	y = chance + (1-chance) / (1 + np.exp(-b*(x-a)))
	return y

# ? regression
def linear_regression(X,Y, split_per=None):
	import statsmodels.api as sm

	# ! sns.regplot much better
	if split_per is not None: raise NotImplementedError("Fix dataset splitting") # TODO spplit dataset
	# remove NANs
	remove_idx = [i for i,(x,y) in enumerate(zip(X,Y)) if np.isnan(x) or np.isnan(y)]

	X = np.delete(X, remove_idx)
	Y = np.delete(Y, remove_idx)
	# Regression with Robust Linear Model
	X = sm.add_constant(X)
	res = sm.RLM(Y, X, missing="drop").fit()
	# raise ValueError(res.params)
	return X, res.params[0], res.params[1], res

# ! STATISTICAL DISTRIBUTIONS
def get_distribution(dist, *args, n_samples=10000):
	if dist == 'uniform':
		return np.random.uniform(args[0], args[1], n_samples)
	elif dist == 'normal':
		return np.random.normal(args[0], args[1], n_samples)
	elif dist == 'beta':
		return np.random.beta(args[0], args[1], n_samples)
	elif dist == 'gamma':
		return np.random.gamma(args[0], args[1], n_samples)


# ! STATISTICAL DISTRIBUTIONS PARAMETERS
def beta_distribution_params(a=None, b=None, mu=None, sigma=None, omega=None, kappa=None):
	"""[converts parameters of beta into different formulations]
	
	Keyword Arguments:
		a {[type]} -- [a param] (default: {None})
		b {[type]} -- [b param] (default: {None})
		mu {[type]} -- [mean] (default: {None})
		sigma {[type]} -- [standard var] (default: {None})
		omega {[type]} -- [mode] (default: {None})
		kappa {[type]} -- [concentration] (default: {None})
	
	Raises:
		NotImplementedError: [description]
	"""
	if kappa is not None and omega is not None:
		a = omega * (kappa-2) + 1
		b = (1 - omega)*(kappa - 2) + 1 
		return a, b
	elif a is not None and b is not None:
		mu = a / (a+b)
		omega = (a - 1)/(a + b -2)
		kappa  = a + b
		return mu, omega, kappa
	else: raise NotImplementedError

def gamma_distribution_params(mean=None, sd=None, mode=None, shape=None, rate=None):
	if mean is not None and sd is not None:
		if mean < 0: raise NotImplementedError
		
		shape = mean**2 / sd**2
		rate = mean / sd**2
	elif mode is not None and sd is not None:
		if mode < 0: raise NotImplementedError

		rate = (mode+math.sqrt(mode**2 + 4*(sd**2)))/ (2 * (sd**2))
		shape = 1 + mode*rate
	elif shape is not None and rate is not None:
		mu = shape/rate
		sd = math.sqrt(shape)/rate
		return mu, sd
	return shape, rate