import sys
sys.path.append("./")
from Utilities.imports import *

pi = np.pi

class AntHill:
    def __init__(self):
        # params
        n_ants = 100
        grid_size = 100

        jump_prob_foraging = .1
        jump_prop_returning = .01
        recruitement_prob = .25
        detection_range = 2

        anthill_loc = (50, 50)

        # spawn ants
        self.ants = [Ant() for i in range(n_ants)]


class Ant(AntHill):
    def __init__(self):
        AntHill.__init__(self)

        self.position = self.anthill_loc
        self.orientation = np.random.uniform(0, 2*pi, 1)
        self.speed = np.rnadom.uniform(0, 5)