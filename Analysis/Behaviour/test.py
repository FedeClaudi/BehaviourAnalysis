# %%
import numpy as np
import math
import matplotlib.pyplot as plt
%matplotlib inline

# %%
def solve_sas(s1, a, s2):
    # get third side of trinagle
    return math.sqrt(s1**2 + s2**2 - 2*s1*s2*math.cos(np.radians(a)))

# %%
f, ax = plt.subplots()
thetas = np.linspace(0, 180, 10)
S = 1000

X = np.linspace(0, S*2, S)

for theta in thetas:
    Y = [solve_sas(x, theta, S)+x for x in X if math.cos(np.radians(theta))*x<=S]
    ax.plot(Y, label=theta)

ax.legend()  

# %%
