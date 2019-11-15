import sys

sys.path.append("./")
from Utilities.imports import *

class PlotsByCondition:
    def __init__(self):
        return

    # ! p(R) and ESCAPE ARMS
    def arm_preference_plot_bycond(self, conditions=None):
        if conditions is None:
            conditions = self.conditions

        f, axarr = create_figure(subplots=True, ncols=2)
        arms = ["left", "right"]
        x = [0, 1]

        for cond, data in conditions.items():
            origins = [len(data.loc[data.origin_arm == arm])/len(data) for arm in arms]
            escapes = [len(data.loc[data.escape_arm == arm])/len(data) for arm in arms]

            axarr[0].plot(x, origins, "o-", label=cond, alpha=.8)
            axarr[1].plot(x, escapes, "o-", alpha=.8)

        axarr[0].legend()
        axarr[0].set(title = "Arm of origin", ylabel="propr", xlabel="arm", xticks=x, xticklabels=arms, ylim=[0, 1])
        axarr[1].set(title = "Arm of escape", ylabel="propr", xlabel="arm", xticks=x, xticklabels=arms, ylim=[0, 1])

        return f, axarr
    
    def plot_pr_bayes_bycond(self, prdata=None):
        if prdata is  None:
            prdata = self.bayes_by_condition_analytical()

        f, ax = create_figure(subplots=False)

        for i, data in prdata.iterrows():
            ax.errorbar(i, data['mean'], 
                            yerr=np.array(data['mean']-data.prange.low, data.prange.high-data['mean']), 
                            marker="o",
                            label=data.condition)
            

        ax.legend()
        ax.set(title="p(R) by condition [grouped bayes]", ylabel="p(R)", ylim=[0, 1],
                xticks=np.arange(len(prdata)), xticklabels=prdata.condition.values, xlabel="Condition")

        return f, ax


    # ! GENERAL TRIALS METRICS
    def time_out_of_t_escape_duration_histograms_bycond(self, conditions=None):
        if conditions is None:
            conditions = self.conditions

        f, axarr = create_figure(subplots=True, ncols=2)

        for cond, data in conditions.items():
            axarr[0].hist(data.time_out_of_t.values, label=cond, alpha=.8, density=True)
            axarr[1].hist(data.escape_duration.values, alpha=.8, density=True)

        axarr[0].legend()
        axarr[0].set(title = "time out of T by condition", ylabel="prop", xlabel="(s)")
        axarr[1].set(title = "escape_duration by condition", ylabel="prop", xlabel="(s)")

        return f, axarr


    # ! TRACKING