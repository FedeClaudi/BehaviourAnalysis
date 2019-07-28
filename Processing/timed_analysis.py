import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
import statsmodels.api as sm

from scipy.signal import find_peaks, resample

class timedAnalysis:
    def __init__(self):
        pass

    def plot_effect_of_time(self, xaxis_istime=True, robust=False):
            rtdf = self.inspect_rt_metric(load=True, plot=False)

            if xaxis_istime: bw = 60
            else: bw = 0.5

            # crate figure
            f, axarr = create_figure(subplots=True, nrows=5, ncols=3, sharex=True)
            leftcol, centercol, rightcol = [0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]

            # loop over experiments
            for i, (condition, df) in enumerate(self.conditions.items()):
                ax = axarr[leftcol[i]]
                axspeed = axarr[centercol[i]]
                axrt = axarr[rightcol[i]]

                times, times2, ones, zeros, speeds, rts, = [], [], [], [], [], [],
                # loop over trials
                for n, (_, trial) in enumerate(df.iterrows()):
                    # get time of trial and escape arm
                    if xaxis_istime:
                        x = trial.stim_frame_session / trial.fps
                    else:
                        x = trial.trial_number

                    if 'Right' in trial.escape_arm:
                        y = 1
                        ones.append(x)
                    else:
                        y = 0
                        zeros.append(x)

                    # Get escape speed
                    # if y == 1:
                        # escape_speed = np.percentile(line_smoother(trial.tracking_data[:, 2], window_size=51, order=5), 80) / trial.fps
                    escape_speed = np.mean(line_smoother(trial.tracking_data[:, 2], window_size=51, order=5)) / trial.fps
                    times.append(x)
                    speeds.append(escape_speed)

                    # Get reaction time
                    rt = rtdf.loc[rtdf.trialid == trial.trial_id].rt_s.values
                    if np.any(rt): 
                        if not np.isnan(rt[0]):
                            times2.append(x)
                            rts.append(rt[0])

                    # plot
                    ax.scatter(x, y, color=self.colors[i+1], s=50, alpha=.5)

                # linear regression on speed and rt
                try:
                    sns.regplot(times, speeds, ax=axspeed, robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=desaturate_color(self.colors[i+1], k=.8)),
                                line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)
                    sns.regplot(times2, rts, ax=axrt, robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=desaturate_color(self.colors[i+1], k=.8)),
                                line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)
                except:
                    continue

                # Plot KDEs
                ax, kde_right = plot_kde(ax, fit_kde(ones, bw=bw), .8, invert=True, normto=.25, color=self.colors[i+1])
                ax, kde_left = plot_kde(ax, fit_kde(zeros, bw=bw), .2, invert=False, normto=.25, color=self.colors[i+1])

                # Plot ratio of KDEs in last plot
                xxx = np.linspace(np.max([np.min(kde_right.support), np.min(kde_left.support)]), np.min([np.max(kde_right.support), np.max(kde_left.support)]), 1000)
                ratio = [kde_right.evaluate(xx)/(kde_right.evaluate(xx)+kde_left.evaluate(xx)) for xx in xxx]
                axarr[leftcol[4]].plot(xxx, ratio, lw=3, color=self.colors[i+1], label=condition)

            # Set axes correctly
            for i, ax in enumerate(axarr):
                if i in leftcol:
                    kwargs = dict(ylim=[-.1, 1.1], yticklabels=["left", "right"], ylabel="escape", yticks=[0, 1],  ) 
                elif i in centercol:
                    kwargs = dict(ylabel="mean speed", ylim=[0, .45])
                else:
                    kwargs = dict(ylabel="rt (s)", ylim=[0, 6])

                if xaxis_istime:
                    ax.set(xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), **kwargs)
                else:
                    ax.set(xlim=[0, 20], **kwargs)

            axarr[leftcol[0]].set(title="Left/Right")
            axarr[centercol[0]].set(title="Escape speed")
            axarr[rightcol[0]].set(title="Reaction time")

            if xaxis_istime:
                xlab = "time (min)"
            else:
                xlab = "trial #"

            axarr[leftcol[-1]].set(xlabel=xlab)
            axarr[centercol[-1]].set(xlabel=xlab)
            axarr[rightcol[-1]].set(title="Reaction times", xlabel=xlab)


            axarr[leftcol[4]].set(title="balance over time", xlabel=xlab, ylabel="R / L+R")
            make_legend(axarr[leftcol[4]])
            make_legend(axarr[rightcol[-1]])


    def timed_pr(self):
        grouped_modes, grouped_means, grouped_params = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

        # crate figure
        f, axarr = create_figure(subplots=True, nrows=4,sharex=True, sharey=True)

        window_size = 600
        magification_factor = 100
        n_steps = 80

        # loop over experiments
        for i, (condition, trials) in enumerate(self.conditions.items()):
            ax = axarr[i]

            # Get escape arms by time
            trial_times = trials.stim_frame_session.values / trials.fps.values
            trial_outcomes = np.array([1 if "Right" in t.escape_arm else 0 for i,t in trials.iterrows()])

            trial_outcomes = trial_outcomes[np.argsort(trial_times)]
            trial_times = np.sort(trial_times)

            # Sweep over time and do windowed analytical bayes
            means, std = [], []

            for t in np.linspace(np.min(trial_times), np.max(trial_times), n_steps):
                in_window = np.where((trial_times > t-window_size/2) & (trial_times < t+window_size/2))
                if np.any(in_window):
                    outcomes_in_window = trial_outcomes[in_window]

                    if len(outcomes_in_window) < 5: continue # skip times when there are too few trials
                    
                    (a, b, fact), mean, var = self.simple_analytical_bayes(outcomes_in_window)

                    beta, support, density = get_parametric_distribution("beta", a, b, x0=0.05, x1=0.95)
                    try:
                        ax.plot(density/np.max(density)*magification_factor+t, support, alpha=1, color=self.colors[i+1])
                    except:
                        pass

                    # ortholines(ax, [1, 1], [t-window_size/2, t+window_size/2], ls=":", lw=1, color=grey)

            ax.axhline(grouped_means[condition], color=self.colors[i+1], ls="--", lw=1)
            ortholines(ax, [0, 0], [0, 1], ls=":", lw=1, color=grey)
            ax.set(ylim=[-0.1, 1.1],  xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), ylabel=condition)


        axarr[0].set(title="timed grouped bayesian")
        axarr[-1].set(xlabel="time (min)",)

