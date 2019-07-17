import sys
sys.path.append('./')

import numpy as np
from scipy import misc, signal, stats
import matplotlib.pyplot as plt


# ! FIT MATH FUNCTIONS
# ? Functions to pass to curve_fit
def polyfit(order, x, y):
	#  calculate polynomial
	z = np.polyfit(x, y, order)
	f = np.poly1d(z)
	return f

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-b*(x-a)))

def logistic(x, a, b):
    return np.exp(a + b*x)/(1 + np.exp(a + b*x))

def half_sigmoid(x, a, b):
	chance = .5
	return chance + (1-chance) / (1 + np.exp(-b*(x-a)))

def linear_func(x, a, b):
    return x*a + b


def exponential(x, a, b, c, d):
    return  a*np.exp(-c*(x-b))+d

def step_function(x,a, b, c):
    # Step function
    """
    a: value at x = b
    f(x) = 0 if x<b, a if x=b and 2*a if  x > a
    """
    return a * (np.sign(x-b) + c)

# ? regression
def linear_regression(X,Y):
	import statsmodels.api as sm

	# ! sns.regplot much better
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

if __name__ == "__main__":
    xval = sorted(np.concatenate([np.linspace(-5,5,100),[0]])) # includes x = 0
    yval = step_function(xval)
    plt.plot(xval,yval,'ko-')
    plt.ylim(-0.1,1.1)
    plt.xlabel('x',size=18)
    plt.ylabel('H(x)',size=20)
    plt.show()