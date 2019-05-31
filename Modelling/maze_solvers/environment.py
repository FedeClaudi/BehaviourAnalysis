import sys
sys.path.append('./')


from Utilities.imports import *

try: from scipy.special import softmax
except: pass
from Utilities.maths.math_utils import calc_distance_between_points_in_a_vector_2d as dist

from Modelling.maze_solvers.world import World


class Environment(World):
	"""[Creates the environment an agent acts in, subclass of the world class]
	"""
	def __init__(self, grid_size=None, **kwargs):
		World.__init__(self, grid_size=grid_size, **kwargs)

		# Define maze
		self.maze, self.free_states = self.get_maze_from_image()

		# Define available actions
		self.actions = {
			0: 'left',
			1: 'right',
			2: 'up',
			3: 'down',
			4: "up-left",
			5: "up-right",
			6: "down-right",
			7: "down-left"
		}
		self.action_lookup = {v:k for k,v in self.actions.items()}
		self.n_actions = len(self.actions)
		
		# Define geodesic distance from shelter at each location
		self.geodesic_distance = geodist(self.maze, self.goal_location)
		self.geodist = geodist

		# initialise
		self.reset()

		# define the location of bridges to block in the model based v2 experiment

		self.bridges_block_states = {
			"alpha0": [[x, 21] for x in np.arange(12, 20)],
			"alpha1":  [[x, 21] for x in np.arange(20, 25)],
			"lambda":  [[x, 11] for x in np.arange(16, 25)],
			"beta0": [[x, 21] for x in np.arange(4, 10)],
			"beta1": [[x, 21] for x in np.arange(30, 37)],

			"alpha0_large": [[x, 542] for x in np.arange(344, 458)],
			"alpha1_large":  [[x, 542] for x in np.arange(537, 650)],
			"lambda_large":  [[x, 286] for x in np.arange(450, 550)],
			"beta0_large": [[x, 542] for x in np.arange(125, 225)],
			"beta1_large": [[x, 542] for x in np.arange(785, 885)],

			"right": [[x, 21] for x in np.arange(24, 30)],
			"right_large": [[x, 500] for x in np.arange(500, 900)],

			"none": [],
		}

		self.model_based_bridges = ["alpha0", "alpha1", "beta0", "beta1"]
		self.model_based_large_bridges = [a+"_large" for a in self.model_based_bridges]
		self.asymmetric_bridges = ["right", "left"]
		self.asymmetric_large_bridges = [a+"_large" for a in self.asymmetric_bridges]


	def get_maze_from_image(self):
		"""[Extract a drawing of a maze and turns it into a gridworld representation of the maze itself]
		"""

		model = cv2.imread(os.path.join(self.maze_models_folder, self.maze_design))

		# blur to remove imperfections
		kernel_size = 21
		model = cv2.blur(model, (kernel_size, kernel_size))

		# resize and change color space
		model = cv2.resize(model, (self.grid_size, self.grid_size))
		model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

		# threshold and rotate  and flip on y axis
		ret, model = cv2.threshold(model, 50, 255, cv2.THRESH_BINARY)
		model = np.rot90(model, 2)
		model = model[:, ::-1]

		# return list of free spaces
		wh = np.where(model == 255)
		return model, [[y, x] for x, y in zip(wh[0], wh[1])]

	def plot_maze(self, ax=None, plot_free_states=False, start=None, goal=None):
		if ax is None: f, axarr = plt.subplots(ncols=2)

		axarr[0].imshow(self.maze, cmap="Greys_r")
		axarr[0].scatter(self.start_location[0], self.start_location[1], c='g', s=200, alpha=.5)
		axarr[0].scatter(self.goal_location[0], self.goal_location[1], c='r', s=200, alpha=.5)
		axarr[0].scatter(self.curr_state[0], self.curr_state[1], c='b', s=200, alpha=1)

		if start is not None:
			axarr[0].scatter(start[0], start[1], c='g', s=200, alpha=.5)
		if goal is not None:
			axarr[0].scatter(goal[0], goal[1], c='m', s=200, alpha=1)

		if plot_free_states:
			axarr[0].scatter([x for x,y in self.free_states], [y for x,y in self.free_states])

		axarr[1].imshow(self.geodesic_distance)

	def reset(self,):
		# reset the environment
		if not self.randomise_start_location_during_training:
			self.curr_state = self.start_location.copy()
		else:
			start_index = np.random.randint(0,len(self.free_states))
			self.curr_state = self.free_states[start_index]

	def get_available_moves(self, current = None):
		"""[Get legal moves given the current position, i.e. moves that lead to a free cell]
		
		Arguments:
			curr {[list]} -- [x,y coordinates of the agent]
		"""
		legals = []
		if current is None: current = self.curr_state

		surroundings = self.maze[current[1]-1:current[1]+2, current[0]-1:current[0]+2]

		actions = ["up-left", "up", "up-right", "left", "still", "right", "down-left", "down", "down-right"] # ! dont delete still from this list

		legals = [a for i, a in enumerate(actions) if surroundings.flatten()[i] and a != "still"]
		if not legals: raise ValueError("No legal actions, thats impossible")
		else: return legals

	def move(self, action, curr):
		"""
			moves from current position to next position based on the action taken
		"""

		def change(current, x, y):
			"""
				change the position by x,y offsets
			"""
			return [current[0] + x, current[1] + y]

		# depending on the action taken, move the agent by x,y offsets
		up, down = - self.stride, self.stride
		left, right = - self.stride, self.stride

		if action == "left":
			nxt_state = change(curr, left, 0)
		elif action == "right":
			nxt_state = change(curr, right, 0)
		elif action == "up":
			nxt_state = change(curr, 0, up)
		elif action == "down":
			nxt_state = change(curr, 0, down)
		elif action == "up-left":
			nxt_state = change(curr, left, up)
		elif action == "up-right":
			nxt_state = change(curr, right, up)
		elif action == "down-right":
			nxt_state = change(curr, right, down)
		elif action == "down-left":
			nxt_state = change(curr, left, down)
		elif action == "still":
			nxt_state = change(curr, 0, 0)
		else: nxt_state = None
		return nxt_state

	def act(self, action, mode="standard"):
		"""
			based on the action selcted execute the movement and evaluate the outcome.
			rewards and punishments vary depending on the mode being used
		"""
		if mode == "standard":
			move_reward = 0
			wall_reward = -1
			goal_reward = 1
		elif mode == "genetic":
			move_reward = 0
			wall_reward = -1
			goal_reward = 200

		# Move
		action_name = self.actions[action]
		self.next_state = self.move(action_name, self.curr_state)


		"""
			EVALUATE THE CONSEQUENCES OF MOVING
		"""
		if self.next_state in self.free_states:
			self.curr_state = self.next_state
			self.reward = move_reward
		else: 
			self.next_state = self.curr_state
			self.reward = wall_reward

		if(self.next_state == self.goal_location):
			self.reward = goal_reward
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over 

	def create_maze_image_from_vals(self, vals):
		"""[creates an image with the shape of the maze and the color from vals]
		
		Arguments:
			vals {[list]} -- [list with same length as self.free_states]
		
		"""

		image = np.full(self.maze.shape, np.nan)

		for (x,y),v in zip(self.free_states, vals):
			image[y, x] = v

		return image


if __name__ == "__main__":
	envs = Environment()
	envs.act(2)  # move up
	envs.plot_maze(plot_free_states=False)
	plt.show()


