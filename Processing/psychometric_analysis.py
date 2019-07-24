# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from statistics import mode
import pymc3 as pm
from sklearn.metrics import mean_squared_error as MSE
from scipy.spatial.distance import squareform

from Utilities.imports import *

from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
from Processing.analyse_experiments import ExperimentsAnalyser

# %%
# Define class

class PsychometricAnalyser(ExperimentsAnalyser):
    maze_names = {"maze1":"asymmetric_long", "maze2":"asymmetric_mediumlong", "maze3":"asymmetric_mediumshort", "maze4":"symmetric"}

    # DT simulation params
    speed_mu, speed_sigma = 5, 2.5
    speed_noise = 0
    distance_noise = .1

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

        self.short_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].georatio.values[0]
        self.long_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze1"].georatio.values[0]


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
        ||||||||||||||||||||||||||||    DECISION THEORY MODEL     |||||||||||||||||||||
    """
    def dt_model_speed_and_distances(self, plot=False, sqrt=True):
        speed = stats.norm(loc=self.speed_mu, scale=self.speed_sigma)
        if sqrt: dnoise = math.sqrt(self.distance_noise)
        else: dnoise = self.distance_noise
        distances = {a.maze:stats.norm(loc=a.georatio, scale=dnoise) for i,a in self.paths_lengths.iterrows()}
        
        if plot:
            f, axarr = create_figure(subplots=True, ncols=2)
            dist_plot(speed, ax=axarr[0])

            for k,v in distances.items():
                dist_plot(v, ax=axarr[1], label=k)

            for ax in axarr: make_legend(ax)

        return speed, distances

    def simulate_trials_analytical(self):
        # Get simulated running speed and path lengths estimates
        self.get_paths_lengths()
        speed, distances = self.dt_model_speed_and_distances(plot=False)

        # right arm
        right = distances["maze4"]

        # Compare each arm to right
        pR = {k:0 for k in distances.keys()}
        for left, d in distances.items():
            # p(R) = phi(-mu/sigma) and mu=mu_l - mu_r, sigma = sigma_r^2 + sigma_l^2
            mu_l, sigma_l = d.mean(), d.std()
            mu_r, sigma_r = right.mean(), right.std()

            mu, sigma = mu_l - mu_r, sigma_r**2 + sigma_l**2
            pR[left] = round(1 - stats.norm.cdf(-mu/sigma,  loc=0, scale=1), 3)
        return pR


    def simulate_trials(self, niters=1000):
        # Get simulated running speed and path lengths estimates
        self.get_paths_lengths()
        speed, distances = self.dt_model_speed_and_distances(plot=False, sqrt=False)

        # right arm
        right = distances["maze4"]

        # Compare each arm to right
        trials, pR = {k:[] for k in distances.keys()}, {k:0 for k in distances.keys()}
        for left, d in distances.items():
            # simulate n trials
            for tn in range(niters):
                # Draw a random length for each arms
                l, r = d.rvs(), right.rvs()

                # Draw a random speed and add noise
                if self.speed_noise > 0: s = speed.rvs() + np.random.normal(0, self.speed_noise, size=1) 
                else: s = speed.rvs()

                # Calc escape duration on each arma nd keep the fastest
                # if r/s <= l/s:
                if r <= l:
                    trials[left].append(1)
                else: 
                    trials[left].append(0)

            pR[left] = np.mean(trials[left])
        return trials, pR
        
    def fit_model(self):
        xp =  np.linspace(.8, 1.55, 200)
        xrange = [.8, 1.55]

                # Get paths length ratios and p(R) by condition
        self.get_paths_lengths()
        hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
        
        # Get modes on individuals posteriors and grouped bayes
        modes, means, stds = self.get_hb_modes()
        grouped_modes, grouped_means = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

        # Plot each individual's pR and the group mean as a factor of L/R length ratio
        f, axarr = create_figure(subplots=True, ncols=2)
        ax = axarr[1]
        mseax = axarr[0]
            
        lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[], "individuals_y_sigma":[]}
        for i, (condition, pr) in enumerate(p_r.items()):
            x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values
            y = means[condition]

            # ? plot HB PR with errorbars
            ax.errorbar(x, np.mean(y), yerr=np.std(y), 
                        fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
                        ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, 
                        capthick=2, alpha=1, zorder=0)             


        # ? Fit the model with a range of params and plot the results
        fitted = {k:[]  for k in self.conditions.keys()}
        mserr = []
        minsigma, maxsigma = 0.05, .2
        sigma_range = np.linspace(minsigma, maxsigma, 100)
        ytrue = [np.mean(m) for m in means.values()]
        for sigma in sigma_range:
            self.distance_noise = sigma
            analytical_pr = self.simulate_trials_analytical()
            simulation = self.simulate_trials_analytical()
            {fitted[k].append(pr) for k,pr in simulation.items()}
            mserr.append(MSE(ytrue, list(simulation.values())))
   
        # ? Plot mean square error
        for s in sigma_range[::5]:
            vline_to_curve(mseax, s, sigma_range, mserr, color=desaturate_color(teal), lw=2)
        mseax.plot(sigma_range, mserr, color=teal, lw=4)
        mseax.axhline(0, color=white, lw=4)

        # ? Plot best fit
        best_sigma = sigma_range[np.argmin(mserr)]
        lowest_err = mserr[np.argmin(mserr)]
        vline_to_curve(mseax, best_sigma, sigma_range, mserr, color=white, lw=6)
        hline_to_curve(mseax, lowest_err, sigma_range, mserr, color=white, lw=6)

        analytical_pr = self.simulate_trials_analytical()
        pomp = plot_fitted_curve(sigmoid, self.paths_lengths.georatio.values, np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
            scatter_kwargs={"alpha":0}, 
            line_kwargs={"color":white, "alpha":1, "lw":6, "label":"model pR - $\sigma : {}$".format(round(best_sigma, 2))})


        # Fix plotting
        ortholines(ax, [1, 0,], [1, .5])
        ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
        ax.set(title="best fit logistic regression", ylim=[-0.01, 1.05], ylabel="p(R)", xlabel="Left path length (a.u.)",
                 xticks = self.paths_lengths.georatio.values, xticklabels = self.conditions.keys())
        mseax.set(title="Fit error", ylabel="MSE", xlabel="$\sigma$", xlim=[minsigma, maxsigma], ylim=[0, max(mserr)])
        make_legend(ax)
        

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
        xp =  np.linspace(.8, 1.55, 200)
        xrange = [.8, 1.55]

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
            x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values

            if raw_individuals: # ? use raw of HB data
                y = pr
                 # ? Plot individual mice's p(R)
                ax.scatter(np.random.normal(x, 0.01, size=len(y)), y, alpha=.2, color=pink, s=250)
            else:
                y = means[condition]
                # ? plot HB PR with errorbars
                ax.errorbar(np.random.normal(x, 0.01, size=len(y)), y, yerr=stds[condition], 
                            fmt='o', markeredgecolor=desaturate_color(white, k=.4), markerfacecolor=desaturate_color(white, k=.4), markersize=10, 
                            ecolor=desaturate_color(white, k=.2), elinewidth=3, 
                            capthick=2, alpha=.4, zorder=0)
                ax.errorbar(x, np.mean(y), yerr=np.std(y), 
                            fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
                            ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, 
                            capthick=2, alpha=1, zorder=0)             
                vline_to_point(ax, x, np.mean(y), color=desaturate_color(self.colors[i+1], k=.7), lw=4, ls="--", alpha=.4)


            if condition not in exclude_experiments:# ? use the data for curves fitting
                k = .4
                lr_ratios_mean_pr["grouped"].append((x[0], np.mean(pr), np.std(pr)))  
                lr_ratios_mean_pr["individuals_x"].append([x[0] for _ in np.arange(len(y))])
                lr_ratios_mean_pr["individuals_y"].append(y)
                lr_ratios_mean_pr["individuals_y_sigma"].append(stds[condition])
            else: 
                k = .1
                del grouped_modes[condition], grouped_means[condition]

        # Plot simulation results   + plotted sigmoid
        # ? logistic regression on analytical simulation

        fitted = []
        for i, (nn, col) in enumerate(zip(np.linspace(.05, .2, 2), get_n_colors(10))):
            self.distance_noise = nn
            # sim_trials, sim_pr = self.simulate_trials(niters=10000)
            analytical_pr = self.simulate_trials_analytical()

            # pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], np.hstack(list(sim_pr.values())), ax, xrange=xrange, 
            #                         scatter_kwargs={"alpha":1}, 
            #                         line_kwargs={"color":desaturate_color(green), "alpha":.5, "lw":4, "label":"simulated pR - $\omega : {}$".format(round(nn, 2))})
            pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
                        scatter_kwargs={"alpha":1}, 
                        line_kwargs={"color":desaturate_color(teal), "alpha":0, "lw":4})
            fitted.append(pomp)
        ax.fill_between(xp, sigmoid(xp, *fitted[0]), sigmoid(xp, *fitted[1]),  color=lightblue, alpha=.15, label="model pR - $\sigma : {}-{}$".format(.05, .2))

        # ? logistic regression on analytical simulation
        # analytical_pr = self.simulate_trials_analytical()
        # pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
        #     scatter_kwargs={"alpha":1, "color":teal, "s":200}, 
        #     line_kwargs={"color":teal, "alpha":.5, "lw":4, "label":"analytical pR - $\omega : {}$".format(round(self.distance_noise, 2))})


        # ? Bayesian logistic regression on trial data
        # df = self.sigmoid_bayes(plot=False, load=True, robust=False)
        # b0, b0std, b1, b1std = np.mean(df.b0), sqrt(np.std(df.b0)), np.mean(df.b1), sqrt(np.std(df.b1))
        # ax.plot(xp, logistic(xp, b0, b1), color=teal, lw=3, label="bayesian logistic regression - raw trials")

        # ? Fit sigmoid to median pR of raw data  
        plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], [m[1] for m in lr_ratios_mean_pr["grouped"]], ax, xrange=xrange, 
                                fit_kwargs={"sigma":[m[2] for m in lr_ratios_mean_pr["grouped"]]},
                                scatter_kwargs={"alpha":0}, 
                                line_kwargs={"color":pink, "alpha":.8, "lw":4, "label":"mean raw p(R)"})

        # ? Fit logistic regression to mean p(R)+std(p(R))
        pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], [np.mean(n) for n in lr_ratios_mean_pr["individuals_y"]], ax, xrange=xrange, 
            fit_kwargs={"sigma":[np.std(n) for n in lr_ratios_mean_pr["individuals_y"]]},
            scatter_kwargs={"alpha":0}, 
            line_kwargs={"color":teal, "alpha":.5, "lw":4, "label":"mean hb p(R)"})

        # ? Plot fitted sigmoid
        hbfit = plot_fitted_curve(sigmoid, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), ax, xrange=xrange,  # ? ind. sigmoid
                                fit_kwargs={"sigma": np.hstack(lr_ratios_mean_pr["individuals_y_sigma"]), }, 
                                scatter_kwargs={"alpha":0}, 
                                line_kwargs={"color":grey, "alpha":.8, "lw":4, "label":"individuals hb p(R)s"})

        # Fix plotting
        ortholines(ax, [1, 0,], [1, .5])
        ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
        ax.set(ylim=[-0.01, 1.05], ylabel="p(R)", title="p(R) per mouse per maze", xlabel="Left path length (a.u.)",
                 xticks = self.paths_lengths.georatio.values, xticklabels = self.conditions.keys())
        make_legend(ax)


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

    def plot_utility_function(self):
        def get_utility_matrix():
            n_cols = 500
            x = np.linspace(0, 2, n_cols)
            y = np.linspace(0, 2, n_cols)

            u = np.reshape([2 - (xx-yy) for xx in x for yy in y], (n_cols, n_cols)).T
            up_tr = np.triu(u) 
            up_tr[np.tril_indices(n_cols)] = np.nan
            return np.rot90(up_tr.T, 1)


        self.get_paths_lengths()
        print(self.paths_lengths)
        
        f, axarr = create_figure(subplots=True, ncols=2)
        ax, ax2 = axarr[0], axarr[1]

        # Get background image and show
        util = get_utility_matrix()
        ax.imshow(util, extent=[.8, 1.6, .8, 1.6], alpha=.9)

        # Plt arms combinations
        distances = {}
        for i, row in self.paths_lengths.iterrows():
            maze_dist = calc_distance_between_point_and_line([[.8, .8], [1.6, 1.6]], [self.short_arm_len, row.georatio])
            ax.scatter(self.short_arm_len, row.georatio, color=self.colors[i+1], label=row.maze + " - {}".format(round(maze_dist, 4)), **big_dot)
            distances[row.maze] = maze_dist
            c = row.georatio - self.short_arm_len
            ax.plot([0.5, 1.75], [.5+c, 1.75+c], color=self.colors[i+1], lw=3, ls="--", alpha=.25)

            if i < 3: 
                maze_dist2 = calc_distance_between_point_and_line([[.8, .8], [1.6, 1.6]], [row.georatio, self.long_arm_len])
                ax.scatter(row.georatio, self.long_arm_len, color=random.choice(get_n_colors(10)), label=round(maze_dist2, 4),  **big_dot)

        ax.plot([0.5, 1.75], [.5, 1.75], **grey_line)

        make_legend(ax)
        ax.set(title="utility function", xlabel="Right arm length (a.u.)", ylabel="Left arm length (a.u.)",
                xlim=[.8, 1.6], ylim=[.8, 1.6])

        # Plot p(R) vs distance from line
        hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
        modes, means, stds = self.get_hb_modes()
        for_fitting = {"x":[], "y":[], "e":[]}
        for i, (maze, pr) in enumerate(p_r.items()):
            x, y, e = np.random.normal(distances[maze], 0.005, size=len(means[maze])), means[maze], stds[maze]
            ax2.errorbar(x, y, yerr=e, 
                        fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=10, 
                        ecolor=desaturate_color(self.colors[i+1], k=.4), elinewidth=3,  capthick=2, alpha=.8, zorder=0)
            for_fitting["x"].append(x)
            for_fitting["y"].append(y)
            for_fitting["e"].append(e)


        # ? Fit logistic regression to mean p(R)+std(p(R))
        # pomp = plot_fitted_curve(sigmoid, [np.mean(a) for a in for_fitting["x"]], [np.mean(a) for a in for_fitting["y"]], ax2, xrange=[-.01, .45], 
        #     fit_kwargs={"sigma":[np.mean(a) for a in for_fitting["e"]]},
        #     scatter_kwargs={"alpha":1, "color":lightblue}, 
        #     line_kwargs={"color":lightblue, "alpha":1, "lw":4}) 

        pomp = plot_fitted_curve(sigmoid, np.hstack(for_fitting["x"]), np.hstack(for_fitting["y"]), ax2, xrange=[-.01, .45], 
            fit_kwargs={"sigma":np.hstack(for_fitting["e"])},
            scatter_kwargs={"alpha":0}, 
            line_kwargs={"color":white, "alpha":1, "lw":4}) 

        ortholines(ax2, [0, 0.5,], [.5, 0], ls="--", lw=3, alpha=.3)
        ax2.set(title="p(R) vs distance from unity line", xlabel="distance", ylabel="p(R)")

        plt.show()
        a =1 


if __name__ == "__main__":
    pa = PsychometricAnalyser()

    pa.plot_utility_function()


    plt.show()



#%%
