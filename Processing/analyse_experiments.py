# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

if __name__ == "__main__": # avoid re importing the tables for every core in during bayesian modeeling
    from Utilities.imports import * 

# %matplotlib inline
import pymc3 as pm
import pickle

# %%
# Define class
class ExperimentsAnalyser:
    maze_designs = {0:"three_arms", 1:"asymmetric_long", 2:"asymmetric_mediumlong", 3:"asymmetric_mediumshort", 4:"symmetric", -1:"nan"}
    naive_lookup = {0: "experienced", 1:"naive", -1:"nan"}
    lights_lookup = {0: "off", 1:"on", 2:"on_trials", 3:"on_exploration", -1:"nan"}
    colors = {0:"b", 1:"r", 2:"g", 3:"m", 4:"orange", -1:"w"}
    arms_colors = {"Left_Far":"g", "Left_Medium":"g", "Right_Medium":"r", "Right_Far":"r", "Centre":"m"}

    # Bayes hyper params
    hyper_mode = (5, 5)  # a,b of hyper beta distribution (modes)
    concentration_hyper = (1, 10)  # mean and std of hyper gamma distribution (concentrations)
    
    # Folders
    metadata_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\Psychometric"

    def __init__(self):
        # Get a bunch of basic stuff
        self.sessions = pd.DataFrame((Session - 'experiment_name="Foraging"').fetch())
        self.session_metadata = pd.DataFrame((Session * Session.Metadata  - 'experiment_name="Foraging"').fetch())

        # Keep only sessions of interest
        self.sessions = self.sessions.loc[self.session_metadata.maze_type > -1]
        self.session_metadata = self.session_metadata.loc[self.session_metadata.maze_type > -1]

    def __str__(self):
        def get_summary(df):
            summary = dict(maze=[], tot_mice=[], naive=[])
            for maze_id, maze_name in self.maze_designs.items():
                if maze_id == -1: continue
                    
                maze_data = df.loc[df.maze_type == maze_id]

                summary["maze"].append(maze_name)
                summary["tot_mice"].append(len(maze_data))
                summary["naive"].append(len(maze_data.loc[maze_data.naive == 1]))

            summary["not_naive"] = [t-n for t,n in zip(summary["tot_mice"], summary["naive"])]
            summary = pd.DataFrame(summary)
            return summary

        data = self.session_metadata
        # Get how many mice were done with the lights on on each maze, divide by naive and not naive
        lights_on_data = data.loc[data.lights==1]
        summary = get_summary(lights_on_data)

        print("Mice per experiment - lights ON")
        print(summary)


        lights_off_data = data.loc[data.lights==0]
        summary = get_summary(lights_off_data)

        print("\n\nMice per experiment - lights OFF")
        print(summary)
        return ""

    def __repr__(self): 
        self.__str__()
        return ""

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         GET DATA
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """

    def get_sessions_by_condition(self, maze_design=None, naive=None, lights=None, df=False):
        data = Session * Session.Metadata  - 'experiment_name="Foraging"'
        if maze_design is not None:
            data = (data & "maze_type={}".format(maze_design))
        if naive is not None:
            data = (data & "naive={}".format(naive))
        if lights is not None:
            data = (data & "lights={}".format(lights))

        if df:
            return pd.DataFrame((data).fetch())
        else: return data

    def get_sessions_tracking(self, bp="body",  maze_design=None, naive=None, lights=None):
        data = self.get_sessions_by_condition(maze_design, naive, lights, df=False)
        andtracking = (data * TrackingData.BodyPartData & "bpname='{}'".format(bp))
        return pd.DataFrame(andtracking.fetch())

    def get_sesions_trials(self, maze_design=None, naive=None, lights=None, escapes=True):
        # Given a dj query with the relevant sessions, fetches the corresponding trials from AllTrials
        sessions = self.get_sessions_by_condition(maze_design, naive, lights, df=True)
        ss = set(sorted(sessions.uid.values))

        all_trials = pd.DataFrame(AllTrials.fetch())
        if escapes:
            all_trials = all_trials.loc[all_trials.is_escape == "true"]
        trials = all_trials.loc[all_trials.session_uid.isin(ss)]

        return trials


    def get_binary_trials_per_condition(self, conditions):
        # ? conditions should be a dict whose keys should be a list of strings with the names of the different conditions to be modelled
        # ? the values of conditions should be a a list of dataframes, each specifying the trials for one condition (e.g. maze design) and the session they belong to

        # Parse data
        # Get trials
        trials = {c:[] for c in conditions.keys()}
        for condition, df in conditions.items():
            sessions = sorted(set(df.session_uid.values))
            for sess in sessions:
                trials[condition].append([1 if "right" in arm.lower() else 0 for arm in df.loc[df.session_uid==sess].escape_arm.values])

        # Get hits and number of trials
        hits = {c:[np.sum(t2) for t2 in t] for c, t in trials.items()}
        ntrials = {c:[len(t2) for t2 in t] for c,t in trials.items()}
        p_r = {c: [h/n for h,n in zip(hits[c], ntrials[c])] for c in hits.keys()}
        return hits, ntrials, p_r

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         ANALYSE STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """

    def model_hierarchical_bayes(self, conditions):
        hits, ntrials, p_r = self.get_binary_trials_per_condition(conditions)

        # Prep hyper and prior params
        k_hyper_shape, k_hyper_rate = gamma_distribution_params(mean=self.concentration_hyper[0], sd=self.concentration_hyper[1])

        # Create model and fit
        n_conditions = len(list(conditions.keys()))
        print("Fitting bayes to conditions:", list(conditions.keys()))
        with pm.Model() as model:
            # Define hyperparams
            modes_hyper = pm.Beta("mode_hyper", alpha=self.hyper_mode[0], beta=self.hyper_mode[1], shape=n_conditions)
            concentrations_hyper = pm.Gamma("concentration_hyper", alpha=k_hyper_shape, beta=k_hyper_rate, shape=n_conditions) # + 2 # ! FIGURE OUT WHAT THIS + 2 IS DOING OVER HERE ????

            # Define priors
            for i, condition in enumerate(conditions.keys()):
                prior_a, prior_b = beta_distribution_params(omega=modes_hyper[i], kappa=concentrations_hyper[i])
                prior = pm.Beta("{}_prior".format(condition), alpha=prior_a, beta=prior_b, shape=len(ntrials[condition]))
                likelihood = pm.Binomial("{}_likelihood".format(condition), n=ntrials[condition], p=prior, observed=hits[condition])

            # Fit
            print("Got all the variables, starting NUTS sampler")
            trace = pm.sample(750, tune=1500, nuts_kwargs={'target_accept': 0.95}, progressbar=True)
            

        return trace

    def save_bayes_trace(self, trace, savepath):
        if not isinstance(trace, pd.DataFrame):
            trace = pm.trace_to_dataframe(trace)

        with open(savepath, 'wb') as output:
            pickle.dump(trace, output, pickle.HIGHEST_PROTOCOL)


    def load_trace(self, loadname):
        trace = pd.read_pickle(loadname)
        return trace



    def naive_hb(self, load=True):
        tracename = os.path.join(self.metadata_folder, "naive_maze1.pkl")
        conditions = dict(
                asym_naive =  self.get_sesions_trials(maze_design=1, naive=1, lights=1, escapes=True),
                asym_experienced =  self.get_sesions_trials(maze_design=1, naive=0, lights=1, escapes=True),
                asym_all =  self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
            )

        if not load:
            trace = self.model_hierarchical_bayes(conditions)
            self.save_bayes_trace(trace, tracename)

            pm.traceplot(trace)
            plt.show()
        else:
            trace = self.load_trace(tracename)

        # Plot by condition
        good_columns = {c:[col for col in trace.columns if c in col] for c in conditions.keys()}
        condition_traces = {c:trace[cols].values for c, cols in good_columns.items()}

        f, axarr = plt.subplots(nrows=3)
        for (condition, data), color, ax in zip(condition_traces.items(), ["w", "m", "g"], axarr):
            for i in np.arange(data.shape[1]):
                if i == 0: label = condition
                else: label=None
                sns.kdeplot(data[:, i], ax=ax, color=color, shade=True, alpha=.45)

        for ax in axarr:
            ax.set(title="p(R) posteriors on asymmetric maze", xlabel="p(R)", ylabel="pdf", facecolor=[.2, .2, .2])
            ax.legend()
        plt.show()

        a = 1

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         PLOT STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """

    def plot_tracking(self, tracking, ax=None, colorby=None, origin=False):
        if ax is None: f, ax = plt.subplots()

        for i, trial in tqdm(tracking.iterrows()):
            if not origin: tr = trial.tracking_data
            else: tr = trial.outward_tracking_data

            if colorby == "arm": 
                c = self.arms_colors[trial.escape_arm]
                cmap=None
            elif colorby == "speed": 
                c=tr[:, 2]
                cmap="gray"
            else: c = "w"
                

            ax.scatter(tr[:, 0], tr[:, 1], c=c,  cmap=cmap, alpha=.25, s=10) # cmap="gray", vmin=0, vmax=10,

        ax.set(facecolor=[.2, .2, .2])

    def plot_pr_by_condition(self):
        conditions = dict(
            asym_naive =  self.get_sesions_trials(maze_design=1, naive=1, lights=1, escapes=True),
            asym_experienced =  self.get_sesions_trials(maze_design=1, naive=1, lights=1, escapes=True),
            asym_all =  self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
        )

        hits, ntrials, p_r = self.get_binary_trials_per_condition(conditions)

        f, ax = plt.subplots()

        for i, ((condition, pr), color) in enumerate(zip(p_r.items(), ["w", "m", "g"])):
            ax.scatter(np.random.normal(i, .1, size=len(pr)), pr, alpha=.4, color=color, s=250, label=condition)

        ax.set(facecolor=[.2, .2, .2])
        ax.legend()

        

if __name__ == "__main__":
    ea = ExperimentsAnalyser()
    ea.naive_hb()
    plt.show()

