
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
