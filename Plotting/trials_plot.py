# %%
import sys
sys.path.append('./')

from Utilities.imports import *


from Processing.exploration_analysis.exploration_database import AllExplorationsPopulate
from Processing.trials_analysis.all_trials_database import AllTrials


trials = AllTrials()
explorations = AllExplorations()

# %%
for session in list(set(trials.fetch("session_uid")[::-1]))[::-1]:
    f, axarr = create_figure(subplots=True, ncols=2, figsize=(16, 16), facecolor=white)

    exploration =   pd.DataFrame((explorations  & "session_uid={}".format(session)))
    session_trials = pd.DataFrame((trials & "session_uid={}".format(session)))
    rec = session_trials['recording_uid'].values[0]

    # Plot exploration
    # axarr[0].scatter(exploration['tracking_data'].values[0][:, 0], 
    #                 exploration['tracking_data'].values[0][:, 1], 
    #                 c=np.arange(len(exploration['tracking_data'].values[0][:, 0])), 
    #                 cmap="Greens", s=10, alpha=.8)

    # Add exploration stats
    stats = ("""Expl. duration: {}\n
Time in shelter: {}\n
Time on threat: {}\n
Distance travelled: {}\n
Median spped: {}\n""".format(exploration['duration'].values[0],
exploration['tot_time_in_shelter'].values[0], exploration['tot_time_on_threat'].values[0],
exploration['total_travel'].values[0], exploration['median_vel'].values[0]))

    axarr[0].text(120, 750, stats, bbox=dict(facecolor=grey, alpha=0.5))
    axarr[0].set(title="Exploration -- session {}".format(rec), facecolor=[.2, .2, .2], xlim=[100, 800], ylim=[100, 900])

    # plot trials
    ch_body = MplColorHelper("Greens", 0, len(session_trials)+10, inverse=True)
    ch_head = MplColorHelper("Reds", 0, len(session_trials)+10)
    for i, trial in session_trials.iterrows():
        color = ch.get_rgb(i)

        x, y = trial['tracking_data'][:, 0], trial['tracking_data'][:, 1]
        tx, ty = trial['tail_tracking_data'][:, 0], trial['tail_tracking_data'][:, 1]
        sx, sy = trial['snout_tracking_data'][:, 0], trial['snout_tracking_data'][:, 1]
        nx, ny = trial['neck_tracking_data'][:, 0], trial['neck_tracking_data'][:, 1]

        axarr[1].plot([tx[0:-1:5], x[0:-1:5]], [ty[0:-1:5], y[0:-1:5]],
                        color=ch_body.get_rgb(i), alpha=.5, lw=1.5)
        axarr[1].plot([x[0:-1:5], nx[0:-1:5]], [y[0:-1:5], ny[0:-1:5]],
                        color=ch_head.get_rgb(i), alpha=.5, lw=1.5, 
                        )
        axarr[1].scatter(x[0], y[0], color=ch_head.get_rgb(i), label=str(i))

    axarr[1].legend()
    axarr[1].set(title="Trials", facecolor=[.2, .2, .2],xlim=[100, 800], ylim=[100, 900])

    break

# %%
