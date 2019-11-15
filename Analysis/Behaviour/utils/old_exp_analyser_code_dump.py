

"""
    ||||||||||||||||||||||||||||    DECISION THEORY MODEL     |||||||||||||||||||||
"""
def dt_model_speed_and_distances(self, plot=False, sqrt=True):
    speed = stats.norm(loc=self.speed_mu, scale=self.speed_sigma)
    if sqrt: dnoise = math.sqrt(self.distance_noise)
    else: dnoise = self.distance_noise
    distances = {a.maze:stats.norm(loc=a[self.ratio], scale=dnoise) for i,a in self.paths_lengths.iterrows()}
    
    if plot:
        f, axarr = create_figure(subplots=True, ncols=2)
        dist_plot(speed, ax=axarr[0])

        for k,v in distances.items():
            dist_plot(v, ax=axarr[1], label=k)

        for ax in axarr: make_legend(ax)

    return speed, distances

def simulate_trials_analytical(self):
    # Get simulated running speed and path lengths estimates
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
        x = self.paths_lengths.loc[self.paths_lengths.maze == condition][self.ratio].values
        y = means[condition]

        # ? plot HB PR with errorbars
        ax.errorbar(x, np.mean(y), yerr=np.std(y), 
                    fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
                    ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, 
                    capthick=2, alpha=1, zorder=0)             

    def residual(distances, sigma):
        self.distance_noise = sigma
        analytical_pr = self.simulate_trials_analytical()
        return np.sum(np.array(list(analytical_pr.values())))
            
    params = Parameters()
    params.add("sigma", min=1.e-10, max=.5)
    model = Model(residual, params=params)
    params = model.make_params()
    params["sigma"].min, params["sigma"].max = 1.e-10, 1

    ytrue = [np.mean(m) for m in means.values()]
    x = self.paths_lengths[self.ratio].values

    result = model.fit(ytrue, distances=x, params=params)

    # ? Plot best fit
    # best_sigma = sigma_range[np.argmin(mserr)]
    best_sigma = result.params["sigma"].value
    self.distance_noise = best_sigma

    analytical_pr = self.simulate_trials_analytical()
    pomp = plot_fitted_curve(sigmoid, self.paths_lengths[self.ratio].values, np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
        scatter_kwargs={"alpha":0}, 
        line_kwargs={"color":white, "alpha":1, "lw":6, "label":"model pR - $\sigma : {}$".format(round(best_sigma, 2))})


    # Fix plotting
    ortholines(ax, [1, 0,], [1, .5])
    ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
    ax.set(title="best fit logistic regression", ylim=[-0.01, 1.05], ylabel="p(R)", xlabel="Left path length (a.u.)",
                xticks = self.paths_lengths[self.ratio].values, xticklabels = self.conditions.keys())
    make_legend(ax)
    