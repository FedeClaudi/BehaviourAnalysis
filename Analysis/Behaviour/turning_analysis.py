# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser

# %%
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True)


#%%
# Get Threat Platform data
ea.prep_tplatf_trials_data(filt=False, remove_errors=True,)


#%%
%matplotlib inline


#%%
f, axarr = create_triplot(facecolor=white, figsize=(16, 12))


delta_x_f, delta_y_f = [], []
for i, (condition, trials) in enumerate(ea.trials.items()):
    if condition == "maze1" or condition == "maze4": continue

    ch = MplColorHelper("Greens", 0,300)

    for n, (ii, trial) in enumerate(trials.iterrows()):
        x, y = trial.threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.threat_tracking[:, 1]-trial.threat_tracking[0, 1]

        if "left" in trial.escape_arm.lower():  x -= x

        try:
            ymove = np.where(y > 35)[0][0]
            xmove = np.where(np.abs(x) > 20)[0][0]
        except:
            continue

        delta_x_f.append(xmove); delta_y_f.append(ymove)

        col = ch.get_rgb(xmove)

        axarr.main.plot(x, y, lw=2, color=col, alpha=.40)
        axarr.x.plot(np.abs(x), color=col, alpha=.40)
        axarr.y.plot(y, color=col, alpha=.40)



    axarr.main.set(title=condition, xlim=[-40, 40], ylim=[-50, 175], facecolor=[.2, .2, .2])
    axarr.x.set(facecolor=[.2, .2, .2])
    axarr.y.set(facecolor=[.2, .2, .2])

#%%
f, ax = create_figure(subplots=False, figsize=(12, 12))

plot_kde(ax, delta_x_f, 0, invert=False, vertical=False, normto=None, label=None, kde_kwargs={"bw":2})