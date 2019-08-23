# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
%matplotlib inline

from Processing.psychometric_analysis import PsychometricAnalyser

# %%
# get dists
pa = PsychometricAnalyser()

# %%
"""
    Define two populations I_1 and I_2 which accumulate evidence with 
    dy_1 = I_1dt + cdW where  I_1 is the drift for 1  ad cdW is white noise. 
"""

def noise():
    return np.random.normal(0, c)

# Define normal distributions for the length of each path
m4 = np.random.normal(590, np.sqrt(590), size=1000)
m3 = np.random.normal(650, np.sqrt(650), size=1000)
m2 = np.random.normal(698, np.sqrt(698), size=1000)
m1 = np.random.normal(822, np.sqrt(822), size=1000)

c = 2000 # std of the white noise
Z = 100000 # threshold
max_iter = 10000


L_dist, R_dist = m4, m1

# %%
# Show one trial
f, ax = plt.subplots()

yL, yL_history = 0, []
yR, yR_history = 0, []

counter = 0
while True:
    L, R  = random.choice(L_dist), random.choice(R_dist)

    yL += L + noise()
    yR += R + noise()

    yL_history.append(yL)
    yR_history.append(yR)

    counter += 1
    if yL >= Z or yR >= Z or counter >=max_iter: 
        break

ax.plot(yL_history, color=red)
ax.plot(yR_history, color=green)
ax.axhline(Z, color=white)

# %%
# SImulat trials
outcomes, rtL, rtR = [], [], []
for i in tqdm(range(1000)):
    for i in range(10):
        yL, yL_history = 0, []
        yR, yR_history = 0, []

        counter = 0
        while True:
            # Arbitrary drift values
            L, R  = random.choice(L_dist), random.choice(R_dist)

            yL += L # + noise()
            yR += R # + noise()

            yL_history.append(yL)
            yR_history.append(yR)

            counter += 1
            if yL >= Z or yR >= Z or counter >=max_iter: 
                if yL >= yR: 
                    outcomes.append(0)
                    rtL.append(counter)
                else: 
                    outcomes.append(1)
                    rtR.append(counter)
                break

print("\n\np(R) = {} -- rt L: {} - rt R: {}".format(round(np.mean(outcomes), 2), round(np.mean(rtL), 2), round(np.mean(rtR), 2)))

#%%


#%%
