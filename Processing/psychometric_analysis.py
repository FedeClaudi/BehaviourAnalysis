# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from statistics import mode
import pymc3 as pm

from Utilities.imports import *

from Processing.plot.plot_distributions import plot_fitted_curve

from Processing.analyse_experiments import ExperimentsAnalyser

# %%
# Define class

class PsychometricAnalyser(ExperimentsAnalyser):
    maze_names = {"maze1":"asymmetric_long", "maze2":"asymmetric_mediumlong", "maze3":"asymmetric_mediumshort", "maze4":"symmetric"}

    def __init__(self):
        ExperimentsAnalyser.__init__(self)

        # self.conditions = dict(
        #     maze1 =  self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
        #     maze2 =  self.get_sesions_trials(maze_design=2, naive=None, lights=1, escapes=True),
        #     maze3 =  self.get_sesions_trials(maze_design=3, naive=None, lights=1, escapes=True),
        #     maze4 =  self.get_sesions_trials(maze_design=4, naive=None, lights=1, escapes=True),
        # )
        self.conditions = self.load_trials_from_pickle()

        self.maze_names_r = {v:k for k,v in self.maze_names.items()}

    def get_paths_lengths(self, load=True):
        if not load:
            # Loop over each maze design and get the path lenght (from the tracking data)
            summary = dict(maze=[], left=[], right=[], ratio=[])
            for i in np.arange(4):
                mazen = i +1
                trials = self.get_sesions_trials(maze_design=mazen, naive=None, lights=None, escapes=True)

                # Get the length of each escape
                lengths = []
                for _, trial in trials.iterrows():
                    lengths.append(np.sum(calc_distance_between_points_in_a_vector_2d(trial.tracking_data[:, :2])))
                trials["lengths"] = lengths

                left_trials, right_trials = [t.trial_id for i,t in trials.iterrows() if "left" in t.escape_arm.lower()], [t.trial_id for i,t in trials.iterrows() if "right" in t.escape_arm.lower()]
                left, right = trials.loc[trials.trial_id.isin(left_trials)], trials.loc[trials.trial_id.isin(right_trials)]

                # Make dict for summary df
                l, r  = percentile_range(left.lengths, low=10).low, percentile_range(right.lengths, low=10).low
                summary["maze"].append(self.maze_names_r[self.maze_designs[mazen]])
                summary["left"].append(round(l, 2))
                summary["right"].append(round(r, 2))
                summary["ratio"].append(round(l / r, 4))

            self.paths_lengths = pd.DataFrame.from_dict(summary)
            self.paths_lengths.to_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))
        else:
            self.paths_lengths = pd.read_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))

        geopaths = self.get_arms_lengths_with_agent(load=True)
        self.paths_lengths = pd.merge(self.paths_lengths, geopaths)
        short_arm_dist = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].distance.values[0]
        self.paths_lengths["georatio"] = [round(x / short_arm_dist, 4) for x in self.paths_lengths.distance.values]


    """
        ||||||||||||||||||||||||||||    GETTERS     |||||||||||||||||||||
    """
    def get_hb_modes(self):
        trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False)
        n_bins = 100
        bins = {k:[np.digitize(a, np.linspace(0, 1, n_bins)) for a in v.T] for k,v in trace.items()}
        modes = {k:[np.median(b)/n_bins for b in bins] for k,bins in bins.items()}
        stds = {k:[np.std(m) for m in v.T] for k,v in trace.items()}
        means = {k:[np.mean(m) for m in v.T] for k,v in trace.items()}

        return modes, means, stds

    """
        ||||||||||||||||||||||||||||    BAYES     |||||||||||||||||||||
    """
    def sigmoid_bayes(self, plot=True, load=False, robust=False):
        tracename = os.path.join(self.metadata_folder, "robust_sigmoid_bayes.pkl")
        if not load:
            # Get data
            allhits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
            self.get_paths_lengths()
            
            # Clean data and plot scatterplot
            if plot: f, ax = plt.subplots(figsize=large_square_fig)
            x_data, y_data = [], []
            for i, (condition, hits) in enumerate(allhits.items()):
                failures = [ntrials[condition][ii]-hits[ii] for ii in np.arange(n_mice[condition])]            
                x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values[0]

                xxh, xxf = [x for h in hits for _ in np.arange(h)],   [x for f in failures for _ in np.arange(f)]
                yyh, yyf = [1 for h in hits for _ in np.arange(h)],   [0 for f in failures for _ in np.arange(f)]

                x_data += xxh + xxf
                y_data += yyh + yyf

            if plot:
                ax.scatter(x_data, [y + np.random.normal(0, 0.07, size=1) for y in y_data], color=white, s=250, alpha=.3)
                ax.axvline(1, color=grey, alpha=.8, ls="--", lw=3)
                ax.axhline(.5, color=grey, alpha=.8, ls="--", lw=3)
                ax.axhline(1, color=grey, alpha=.5, ls=":", lw=1)
                ax.axhline(0, color=grey, alpha=.5, ls=":", lw=1)

            # Get bayesian logistic fit + plot
            xp = np.linspace(np.min(x_data)-.2, np.max(x_data)  +.2, 100)
            if not robust:
                trace = self.bayesian_logistic_regression(x_data, y_data) # ? naive
            else:
                trace = self.robust_bayesian_logistic_regression(x_data, y_data) # ? robust

            b0, b0_std = np.mean(trace.get_values("beta0")), np.std(trace.get_values("beta0"))
            b1, b1_std = np.mean(trace.get_values("beta1")), np.std(trace.get_values("beta1"))
            if plot:
                ax.plot(xp, logistic(xp, b0, b1), color=red, lw=3)
                ax.fill_between(xp, logistic(xp, b0-b0_std, b1-b1_std), logistic(xp, b0+b0_std, b1+b1_std),  color=red, alpha=.15)
        
                ax.set(title="Logistic regression", yticks=[0, 1], yticklabels=["left", "right"], ylabel="escape arm", xlabel="L/R length ratio",
                            xticks=self.paths_lengths.georatio.values, xticklabels=self.paths_lengths.georatio.values)

            df = pd.DataFrame.from_dict(dict(b0=trace.get_values("beta0"), b1=trace.get_values("beta1")))
            df.to_pickle(tracename)
        else:
            df = pd.read_pickle(tracename)
        return df

    """
        ||||||||||||||||||||||||||||    PLOTTERS     |||||||||||||||||||||
    """
    def plot_pr_by_condition(self, raw_individuals=False, exclude_experiments=[None], ax=None):
        # Get paths length ratios and p(R) by condition
        self.get_paths_lengths()
        hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
        
        # Get modes on individuals posteriors and grouped bayes
        modes, means, stds = self.get_hb_modes()
        grouped_modes, grouped_means = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

        # Plot each individual's pR and the group mean as a factor of L/R length ratio
        if ax is None: 
            f, ax = create_figure(subplots=False)
            
        lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[], "individuals_y_sigma":[]}
        for i, (condition, pr) in enumerate(p_r.items()):
            x = self.paths_lengths.loc[self.paths_lengths.maze == condition].distance.values

            if raw_individuals: # ? use raw of HB data
                y = pr
                 # ? Plot individual mice's p(R)
                ax.scatter(np.random.normal(x, 0.01, size=len(y)), y, alpha=.2, color=pink, s=250)
            else:
                y = means[condition]
                # ? plot HB PR with errorbars
                ax.errorbar(np.random.normal(x, 7, size=len(y)), y, yerr=stds[condition], 
                            fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=12, 
                            ecolor=desaturate_color(self.colors[i+1], k=.5), elinewidth=3, 
                            capthick=2, alpha=.8)
                            
            if condition not in exclude_experiments:# ? use the data for curves fitting
                k = .4
                lr_ratios_mean_pr["grouped"].append((x[0], np.median(y)))  
                lr_ratios_mean_pr["individuals_x"].append([x[0] for _ in np.arange(len(y))])
                lr_ratios_mean_pr["individuals_y"].append(y)
                lr_ratios_mean_pr["individuals_y_sigma"].append(stds[condition])
            else: 
                k = .1
                del grouped_modes[condition], grouped_means[condition]


        plot_fitted_curve(sigmoid, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), ax, xrange=[400, 1000],  # ? ind. sigmoid
                                fit_kwargs={"sigma": np.hstack(lr_ratios_mean_pr["individuals_y_sigma"]), "bounds":[[500, .001], [700, .2]], "method":"dogbox"},
                                scatter_kwargs={"alpha":0}, 
                                line_kwargs={"color":white, "alpha":1, "lw":4, "label":"logistic - individudals"})

        # Fix plotting
        ortholines(ax, [1, 0,], [590, .5])
        ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
        ax.set(ylim=[-0.05, 1.05], ylabel="p(R)", title="p(R) per mouse per maze", xlabel="Left path length (a.u.)",
                 xticks = self.paths_lengths.distance.values, xticklabels = self.paths_lengths.distance.values)
        style_legend(ax)


    def plot_heirarchical_bayes_effect(self):
        # Get hierarchical Bayes modes and individual mice p(R)
        hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(self.conditions)
        trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False) 

        # Get the mode of the posteriors
        modes, means, stds = self.get_hb_modes()

        f, axarr = plt.subplots(ncols=4, sharex=True, sharey=True)
        
        for i, (exp, ax) in enumerate(zip(trace.keys(), axarr)):
            ax.scatter(np.random.normal(0, .025, size=len(p_r[exp])), p_r[exp], color=self.colors[i+1], alpha=.5, s=200)
            ax.scatter(np.random.normal(1, .025, size=len(modes[exp])), modes[exp], color=self.colors[i+1], alpha=.5, s=200)
            ax.scatter(np.random.normal(2, .025, size=len(means[exp])), means[exp], color=self.colors[i+1], alpha=.5, s=200)
            ax.scatter(np.random.normal(3, .025, size=len(stds[exp])), stds[exp], color=self.colors[i+1], alpha=.5, s=200)

            ax.scatter(0, np.mean(p_r[exp]), color="w", alpha=1, s=300)
            ax.scatter(1, np.mean(modes[exp]), color="w", alpha=1, s=300)
            ax.scatter(2, np.mean(means[exp]), color="w", alpha=1, s=300)
            ax.scatter(3, np.mean(stds[exp]), color="w", alpha=1, s=300)

            ax.set(title=exp, xlim=[-.1, 3.1], ylim=[-.02, 1.02], xticks=[0, 1, 2, 3], xticklabels=["Raw", "hb modes", "hb means", "hb stds"], ylabel="p(R)")


        
if __name__ == "__main__":
    pa = PsychometricAnalyser()

    # f, axarr = plt.subplots(figsize=large_square_fig, ncols=3, nrows=2)
    # axarr = axarr.flatten()
    # pa.plot_pr_by_condition(raw_individuals=False, ax=axarr[0])
    # for i, exp in enumerate(pa.conditions.keys()):
    #     pa.plot_pr_by_condition(raw_individuals=False, exclude_experiments= [exp], ax=axarr[i+1])

    pa.plot_pr_by_condition(raw_individuals=False)
    # pa.sigmoid_bayes(load=False, plot=True, robust=False)
    # pa.plot_heirarchical_bayes_effect()

    plt.show()



#%%
