# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
%matplotlib inline

from Processing.psychometric_analysis import PsychometricAnalyser
pa = PsychometricAnalyser()

print(pa.paths_lengths)

#%%
# PLot linear utility function
f, ax = plt.subplots()

ax.scatter([pa.paths_lengths["distance"].values[-1] for i in range(4)], pa.paths_lengths.distance,
                    color=white)

for i, psi in enumerate(pa.paths_lengths["georatio"].values):
    if i < 3: 
        ax.plot([0, 1000], [0, psi*1000], color=red, alpha=0.75)
    else:
        ax.plot([0, 1000], [0, psi*1000], color=white, alpha=0.75)


ax.set(title="linear utility function", xlim=[200, 900], ylim=[200, 900], xlabel="$l_R$", ylabel="$l_L$")
#%%
# PLot NON linear utility function
# assume that if Lr = 400, all indifference curves pass through the unity line. 
f, ax = plt.subplots()

ax.scatter([pa.paths_lengths["distance"].values[-1] for i in range(4)], pa.paths_lengths.distance,
                    color=white)

for i, psi in enumerate(pa.paths_lengths["georatio"].values):
    if i < 3: 
        # ? Fit non linear indifference curve
        x = [0,  300,       1000, 800, 1200, pa.paths_lengths["distance"].values[-1]]
        y = [0,  psi*300,      psi*1000, psi*800, psi*1200, psi*pa.paths_lengths["distance"].values[-1]]
        fit = polyfit(4, x, y)

        ax.scatter(x, y, color=grey)

        x = np.linspace(0, 1200)
        y = fit(x)
        ax.plot(x, y, color=red, alpha=0.75)
    else:
        ax.plot([0, 1200], [0, psi*1200], color=white, alpha=0.75)


ax.set(title="non linear steep utility func", xlim=[0, 1100], ylim=[0, 1500], xlabel="$l_R$", ylabel="$l_L$")

#%%
# PLot NON linear utility function with different slope
# assume that if Lr = 400, all indifference curves pass through the unity line. 
f, ax = plt.subplots()

ax.scatter([pa.paths_lengths["distance"].values[-1] for i in range(4)], pa.paths_lengths.distance,
                    color=white)

for i, psi in enumerate(pa.paths_lengths["georatio"].values):
    if i < 3: 
        # ? Fit non linear indifference curve
        dpsi = psi - 1
        x = [0, 200, 400, pa.paths_lengths["distance"].values[-1]]
        y = [0, (1 + dpsi*.5)*200, (1 + dpsi*.75)*400, psi*pa.paths_lengths["distance"].values[-1]]
        fit = polyfit(5, x, y)

        x = np.linspace(0, 1200)
        y = fit(x)
        ax.plot(x, y, color=red, alpha=0.75)
    else:
        ax.plot([0, 1200], [0, psi*1200], color=white, alpha=0.75)


ax.set(title="non linear utility funciton", xlim=[200, 1100], ylim=[200, 1100], xlabel="$l_R$", ylabel="$l_L$")

#%%
