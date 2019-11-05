"""
    Analyze behaviour on the threat platform (trying to figure out when they commit to one or the other)
"""

# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Analysis.Behaviour.utils.T_utils import get_T_data
%matplotlib inline

# %%
# Getting data
aligned_trials  = get_T_data(load=False, median_filter=False)


# %%
# Plot slow frames or those in which the angular velocity was above a certiain threshold
f, axarr = create_triplot(facecolor=white, figsize=(15,15))

# thresholds
yth = 200
sth =  2
avelth = 5

for i, trial in tqdm(aligned_trials.iterrows()):
    # Get right aligned tracking data
    if "left" == trial.escape_side:
        x, y, s = 500 + (500 - trial.tracking[:, 0]), trial.tracking[:, 1], trial.tracking[:, 2]
    else:
        x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]

    # Get when mouse goes over the y threshold
    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        below_yth = 0
        above_yth = 0
    else:
        above_yth = below_yth + 1

    # get fast and slow frames [after it went over Y threshold]
    slow_idx = np.where(s[below_yth:] < sth)[0]+below_yth

    # Get fast angular velocity frame
    slow_angle_idx = np.where(np.abs(trial.body_angvel)[below_yth:] > avelth)[0]+below_yth
    
    # plot tracking
    axarr.main.plot(x, y, color=grey, lw=1)
    axarr.main.scatter(x[slow_idx], y[slow_idx], color=green, zorder=99, s=15, alpha=.8)
    axarr.main.scatter(x[slow_angle_idx], y[slow_angle_idx], color=red, zorder=99, s=15, alpha=.8)

# Set axes props
axarr.main.axhline(yth, color=white, lw=2)
_ = axarr.main.set(title="red=slow, green=fast", facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[475, 575]) 




#%%
# Plot tracking and highlight fast and slow frames
f, axarr = create_triplot(facecolor=white, figsize=(15,15))
f2, axarr2 = create_figure(subplots=True, ncols=2, facecolor=white, figsize=(15,15))

# thresholds
yth = 200
sth, high_sth =  2, 4
cdth = 175

median_speed, max_speed, all_speeds, cum_dist_from_line_l = [], [], [], []
for i, trial in tqdm(aligned_trials.iterrows()):
    # Get right aligned tracking data
    if "left" == trial.escape_side:
        x, y, s = 500 + (500 - trial.tracking[:, 0]), trial.tracking[:, 1], trial.tracking[:, 2]
        x_cent, y_cent = 500 + (500 - trial.tracking_centered[:, 0]), trial.tracking_centered[:, 1]
        
    else:
        x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]
        x_cent, y_cent = trial.tracking_centered[:, 0], trial.tracking_centered[:, 1]
        

    # Get when mouse goes over the y threshold
    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        below_yth = 0
        above_yth = 0
    else:
        above_yth = below_yth + 1

    # get fast and slow frames
    slow_idx = np.where(s[below_yth:] < sth)[0]+below_yth
    fast_idx = np.where(s[below_yth:] >= high_sth)[0]+below_yth
    median_speed.append(np.nanmedian(s))
    max_speed.append(np.max(s))
    all_speeds.extend([x for x in s if not np.isnan(x)])

    # Get comulative distance between tracking and a straight line going
    # between the mouse location when it gets above yth and when it leaves T
    p0, p1 = [x[above_yth], y[above_yth]], [x[-1], y[-1]]
    cumdist = np.sum(np.array(cals_distance_between_vector_and_line([p0, p1], np.vstack([x[above_yth:], y[above_yth:]]).T))**2)/len(x[above_yth:])
    cum_dist_from_line_l.append(cumdist)


    # plot tracking
    if cumdist < cdth:
        color = red
        ax2 = axarr2[0]
    else:
        color = green
        ax2 = axarr2[1]

    axarr.main.plot(x[:above_yth], y[:above_yth], color=desaturate_color(color), lw=1)
    axarr.main.plot(x[above_yth:], y[above_yth:], color=color, lw=1)

    ax2.plot(x[:above_yth], y[:above_yth], color=desaturate_color(color), lw=1)
    ax2.plot(x[above_yth:], y[above_yth:], color=color, lw=1)

    # Highlight fast and slow frames
    # axarr.main.scatter(x[slow_idx], y[slow_idx], color=red, s=30, zorder=99, alpha=.5)
    # axarr.main.scatter(x[fast_idx], y[fast_idx], color=green, s=30, zorder=99, alpha=.5)

# Plot kde of speeds distributions
kdeax, _, = plot_kde(data=all_speeds, kde_kwargs={"bw":.1}, fig_kwargs={'facecolor':white},  color=white)
kdeax.axvline(sth, color=red)
kdeax.axvline(high_sth, color=green)

# plot KDE of comulative line distances
_, _, = plot_kde(data=cum_dist_from_line_l, ax=axarr.x, kde_kwargs={"bw":10}, color=white)
axarr.x.axvline(cdth, color=green)

# Set axes props
axarr.main.axhline(yth, color=white, lw=2)

for ax in axarr2: ax.axhline(yth, color=white, lw=2)

_ = axarr.main.set(title="red=slow, green=fast", facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[475, 575]) # 
_ = kdeax.set(title="KDE of median speed", facecolor=[.2, .2, .2], xlim=[0, 15])
_= axarr.x.set(title="comulative line distance", facecolor=[.2, .2, .2], xlim=[-500, 750])

_= axarr2[0].set(title="Low max distance", facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[475, 575])
_= axarr2[1].set(title="High max distance", facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[475, 575])


#%%
