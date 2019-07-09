import sys
sys.path.append('./')

from Utilities.imports import *
from mpl_toolkits.mplot3d import Axes3D

from Processing.trials_analysis.all_trials_loader import Trials
from database.database_fetch import get_maze_template

class Plotter(Trials):
    colors = {"Right_Medium": "r", "Left_Far": "g"}
    def __init__(self):
        Trials.__init__(self, exp_mode=0)

        self.trials = self.trials.loc[self.trials.grouped_experiment_name == "asymmetric"]
        

    def plot(self):
        fig = plt.figure(figsize = (16, 16), facecolor=[.1, .1, .1])
        ax = fig.add_subplot(111, projection='3d')

        bg = get_maze_template(exp="pathint2")
        gray_image = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

        # ax.imshow(gray_image, origin="lower", extent=[0, 1000, 0, 1000])b

        for i, trial in self.trials.iterrows():
            x, y, z = trial.tracking_data[:, 0], trial.tracking_data[:, 1], trial.tracking_data[:, 2]
            x, y, z = line_smoother(x)-500, line_smoother(y)-500, line_smoother(z)

            z = np.append(np.diff(y), 0)
            ax.plot(x, y, z, color=self.colors[trial.escape_arm], alpha=.1, lw=4)

        ax.set(xlabel="x position", ylabel="y position", zlabel="speed", facecolor=[.2, .2, .2])


if __name__ == "__main__":
    p = Plotter()
    p.plot()

    plt.show()