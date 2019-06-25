import sys
sys.path.append("./")
from Utilities.imports import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm


from Processing.trials_analysis.all_trials_loader import Trials

# TODO check that iTheta calculations for 3 arms maze are correct

class GLMdata:
    def __init__(self, load_trials_from_file=False):
        # define file paths
        if sys.platform == "darwin":
            self.hierachical_bayes_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/bayes/results/hierarchical_v2.pkl"
            self.parfile = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze/escape_paths_features.yml"
            self.trials_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/saved_dataframes/glm_data.pkl"
        else:
            self.hierachical_bayes_file = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\bayes\\results\\hierarchical_v2.pkl"
            self.parfile = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze\\escape_paths_features.yml"
            self.trials_file = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\saved_dataframes\\glm_data.pkl"

        # Get trials data
        if not load_trials_from_file:
            # ? Load trials
            self.trials = self.reload_trials_data()

            # Get maze desgin params
            self.maze_params = self.load_maze_params()

            # Get bayes posteriors
            self.bayes_pR = self.load_bayes_posteriors()

            # Get exploration data
            self.trials = self.reload_explorations_data()

            #  Merge dataframes
            self.merge_trial_and_maze_dataframes()

            self.save_trials_to_picke()
        else:
            self.trials = self.load_trials_from_file()

        # Split the data based on the experiment
        self.sym_trials = self.trials.loc[self.trials.experiment_asymmetric == 0]
        self.asym_trials = self.trials.loc[self.trials.experiment_asymmetric == 1]

    # ! DATA LOADING AND HANDLIGN

    def load_maze_params(self):
        params = load_yaml(self.parfile)
        # convert them to df
        temp = dict(
            name = [],
            length = [],
            rLen = [],
            iLen = [],
            uLen = [],
            theta_start = [],
            theta_tot = []
        )
        for exp, pps in params.items():
            for name, ps in pps.items():
                good_name = exp+"_"+name
                temp['name'].append(good_name)
                for i, (dict_list, val) in enumerate(zip(list(temp.values())[1:], ps)):
                    dict_list.append(val)

        return pd.DataFrame.from_dict(temp)

    def load_bayes_posteriors(self):
        data = pd.read_pickle(self.hierachical_bayes_file)
        cols_to_drop = [c for c in data.columns if "asym_prior" not in c and "sym_prior" not in c]
        data = data.drop(cols_to_drop, axis=1)
        return data.mean()

    def save_trials_to_picke(self):
        self.trials.to_pickle(self.trials_file)

    def load_trials_from_file(self):
        return pd.read_pickle(self.trials_file)

    def reload_trials_data(self):
        trial_data = Trials(exp_mode=0)


        trials = trial_data.trials.drop(["is_escape"], axis=1)
        trials = trials.copy() # to defues those pesky pandas warnings

        # Get some extra metrics
        mean_escape_speed, max_escape_speed, escape_path_len, total_angular_displacement  = [],[],[],[],
        for i, trial in trials.iterrows():
            escape_path_len.append(trial.tracking_data.shape[0]/trial.fps)
            mean_escape_speed.append(np.mean(trial.tracking_data[:, 2]))
            max_escape_speed.append(np.percentile(trial.tracking_data[:, 2], 95))

            angles = calc_angle_between_points_of_vector(trial.tracking_data[:, :2])
            total_angular_displacement.append(np.sum(np.abs(calc_ang_velocity(angles))))

        trials['mean_speed'] = mean_escape_speed
        trials['max_speed'] = max_escape_speed
        trials['escape_duration'] = escape_path_len
        trials['total_angular_displacement'] = total_angular_displacement
        trials["x_pos"] = [t[0, 0] for t in trials.tracking_data.values]
        trials["y_pos"] = [t[0, 1] for t in trials.tracking_data.values]
        trials["speed"] = [t[0, 2] for t in trials.tracking_data.values]

        # Clean up
        # Make ToT a float
        trials = trials.loc[trials['time_out_of_t'] > 0]
        trials["time_out_of_t"] = np.array(trials['time_out_of_t'].values, np.float64)

        # Fix arms categoricals
        trials["origin_arm_clean"] = ["right" if "Right" in arm[0] else "left" for arm in trials['origin_arm'].values]
        trials["escape_arm_clean"] = ["right" if "Right" in arm else "left" for arm in trials['escape_arm'].values]

        return pd.get_dummies(trials, columns=["escape_arm_clean", "origin_arm_clean", "grouped_experiment_name"], 
                                    prefix=["escape", "origin", "experiment"])

    def reload_explorations_data(self):
        expl_data = pd.DataFrame(AllExplorations().fetch())
        return pd.merge(self.trials, expl_data, on="session_uid", suffixes=("_trial", "_exploration"))

    def merge_trial_and_maze_dataframes(self):
        rLen, iTheta = [], []

        for i, trial in self.trials.iterrows():
            if trial.experiment_asymmetric:
                arm = self.maze_params.iloc[1]
                iTheta.append(135)            

            else:                
                arm = self.maze_params.iloc[7]
                iTheta.append(180)            

            rLen.append(arm.rLen)


        self.trials['rLen'] = rLen
        self.trials['iTheta'] = iTheta

    # ! HELPER FUNCTIONS
    def split_dataset(self, name, fraction=3):
        # ? name is a string that is equivalent to the name dataset to be splitted
        data = self.__getattribute__(name)
        train, test = train_test_split(data, test_sizefraction)
        return trian, test

    @staticmethod
    def print_mre(y, predictions):
        mse = mean_squared_error(y, predictions)
        print("\nMSE: ", round(mse, 2))

    @staticmethod
    def plotter(y, predictions, label, logistic=False):
            x = np.arange(len(predictions))
            sort_idxs_p = np.argsort(predictions)
            sort_idxs_y = np.argsort(y)

            yy = np.zeros_like(y)-.1
            yy[y > 0] = 1.1

            f, axarr = plt.subplots(figsize=(9, 8), ncols=2)

            for ax, sort_idxs, title in zip(axarr, [sort_idxs_y, sort_idxs_p], ["sort Y", "sort Pred"]):
                ax.scatter(x, y[sort_idxs], c=label[sort_idxs], cmap="Reds", label = 'Obs', alpha=.5, vmin=0)

                ax.scatter(x, predictions[sort_idxs],  c=label[sort_idxs], cmap="Greens", label = 'Pred', alpha=.75, vmin=0)

                if logistic:
                    sns.regplot(x, predictions[sort_idxs], logistic=True, 
                                                truncate=True, scatter=False, ax=ax)

                ax.set(title = title, ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])
                ax.legend()

    # ! MODELS
    def run_glm(self, data, eq, regularized=False):
        model = smf.glm(formula = eq, 
                        data = data,
                        family = sm.families.Binomial(link=sm.families.links.logit))
        if not regularized:
            res = model.fit()
        else:
            res = model.fit_regularized()

        y = data.escape_right.ravel()
        predictions = model.predict(res.params)

        # print(res.params)
        # self.print_mre(y, predictions)
        # print("\n\n")

        return model, res, y, predictions

    def run_crossval_glm(self):
        pass
        # TODO make this work with exog predictions

if __name__ == "__main__":
    l = GLMdata(load_trials_from_file=False)
    # TODO check that iTheta calculations for 3 arms maze are correct
    a = 1