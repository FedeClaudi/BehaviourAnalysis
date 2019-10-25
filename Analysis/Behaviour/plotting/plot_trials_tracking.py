import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *
from Analysis.Behaviour.utils.behaviour_variables import *

class TrialsPlotter:
	def __init__(self):
		self.maze_colors = mazes_colors
		self.arms_colors = arms_colors

	def plot_tracking_trace(self, trials, tracking_key, ax=None, colorby=None, color="w", origin=False, minT=0, maxT=-1,
							as_scatter=True, 
							ax_kwargs = {"facecolor":[.2, .2, .2]},
							scatter_kwargs={"alpha":.8, "s":10}, 
							line_kwargs={"alpha":.8, "lw":2}):
		if ax is None: f, ax = create_figure(subplots=False, figsize=(16, 16), facecolor=white)

		if not isinstance(trials, (pd.DataFrame)): raise ValueError("trials should be a dataframe")
		if not tracking_key in trials.columns: raise ValueError("trackign key should a column name of the trials dataframe")

		for i, trial in trials.iterrows():
			tr = trial[tracking_key]
			if colorby == "arm": 
				kwargs = {"color":self.arms_colors[trial.escape_arm]}
			elif colorby == "speed": 
				if not as_scatter: raise ValueError("If you want to color based on speed, you need to plot as scatter")
				kwargs = {"c":tr[minT:maxT, 2], "cmap":"gray"}
			elif colorby is None: 
				kwargs = {"color":color}
			else:
				raise ValueError("Unrecognized value for colorby arguent: {}".format(colorby))
			
			if as_scatter:
				ax.scatter(tr[minT:maxT, 0], tr[minT:maxT, 1], **scatter_kwargs, **kwargs)
			else:
				ax.plot(tr[minT:maxT, 0], tr[minT:maxT, 1], **line_kwargs, **kwargs)

		if ax_kwargs is not None:
			ax.set(**ax_kwargs)
		return ax

