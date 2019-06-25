# %%
import sys
sys.path.append("./")
import copy
from Utilities.imports import *
%matplotlib inline
plt.ion()

# %%
# hyperparams
hypers = dict(
    ant_p_change_direction   = .1,
    ant_p_change_speed       = .25,
    ant_minmax_speed         = (1, 2),
    exploration_phero_evap   = .05,
    return_phero_evap        = .05,
    detection_radius         = 2,
)


# %%
# Create Ant class
class Ant:
    movement_kernels = [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]
    angle_mappings = [3, 2, 1, 0, 7, 6, 5, 4, 0]
    # angle_mappings = [7, 6, 5, 4, 3, 2, 1, 0]

    def __init__(self, start_loc, worldmap):
        self.anthill_loc = start_loc
        self.worldmap = worldmap
        self.pos = np.array(start_loc)
        self.orientation = np.random.randint(0, 8, 1)[0]
        # 1 up left, 2 up, 3 up right... no stay
        self.speed = np.random.randint(hypers["ant_minmax_speed"][0], hypers["ant_minmax_speed"][1], 1)[0]

        self.status = "foraging"

    def move(self):
        if self.status == "foraging":
            # check if there is food around, otherwise move
            p, r = self.pos, hypers["detection_radius"]
            surroundings = self.worldmap[p[1]-r:p[1]+r, p[0]-r:p[0]+r]
            if np.any(~np.isnan(surroundings[:, :, 0])):
                self.status="fetching" #food 
                self.orient_to_point(self.anthill_loc)
                self.move_returning()
            else:
                self.move_foraging() # TODO let them discover and follow pheromons 
        elif self.status == "fetching":
            # check if we are close to the anthill
            home_dist = calc_distance_between_points_2d(self.pos, self.anthill_loc)
            if home_dist < hypers["detection_radius"]:
                self.status = "returning_to_food"
            else:
                self.orient_to_point(self.anthill_loc)
                self.move_returning()
        elif self.status == "returning_to_food":
            pass
            # TODO follow gradient in return phero to find food 

    def orient_to_point(self, point):
        angle = int(np.round(angle_between_points_2d_clockwise(self.pos, self.anthill_loc) / 45)) 
        # raise ValueError(self.pos, angle, angle_between_points_2d_clockwise(self.pos, self.anthill_loc))
        try: self.orientation = self.angle_mappings[angle]
        except: raise ValueError(angle)
    
    def move_foraging(self):
        # Randomly change direction and speed
        random_probs = np.random.uniform(0, 1, size=10)
        if random_probs[0] <= hypers["ant_p_change_direction"]:
            self.orientation = np.random.randint(0, 8, 1)[0]
        if random_probs[1] <= hypers["ant_p_change_speed"]:
            self.speed = np.random.randint(hypers["ant_minmax_speed"][0], hypers["ant_minmax_speed"][1], 1)

        # if close to a boundary change direction else move + pheromone
        move_vector = self.movement_kernels[self.orientation]
        next_pos = self.pos + move_vector
        if next_pos[0] < hypers["detection_radius"] or next_pos[0] > self.worldmap.shape[0]-hypers["detection_radius"] or next_pos[1] < hypers["detection_radius"] or next_pos[1] > self.worldmap.shape[1]-hypers["detection_radius"]:
            self.orientation =  np.random.randint(0, 8, 1)[0]
        else:
            self.pos = next_pos
            self.worldmap[self.pos[1], self.pos[0], 1] = 1  # exploring phero

    def move_returning(self):
        move_vector = self.movement_kernels[self.orientation]
        next_pos = self.pos + move_vector
        if next_pos[0] < hypers["detection_radius"] or next_pos[0] > self.worldmap.shape[0]-hypers["detection_radius"] or next_pos[1] < hypers["detection_radius"] or next_pos[1] > self.worldmap.shape[1]-hypers["detection_radius"]:
            self.orientation =  np.random.randint(0, 8, 1)[0]
        else:
            self.pos = next_pos
            self.worldmap[self.pos[1], self.pos[0], 2] = 1 # returninng phero

# %%
# World class
class AntWorld:
    def __init__(self, grid_size=100, anthill_loc=(50, 50), anthill_radius=10, 
                n_ants=50, n_food_spots=5, food_per_spot=50):
        self.world_cells = np.full((grid_size, grid_size, 3), np.nan)
        # dimensions: food, exploring pheromon, returning pheromon, boundaries

        self.anthill_loc = anthill_loc
 
        # create anthill
        midcell, halfsize = int(grid_size/2), int(anthill_radius/2)
        self.anthill_image = np.full((grid_size, grid_size), np.nan)
        self.anthill_image[midcell-halfsize:midcell+halfsize, midcell-halfsize:midcell+halfsize] = 1

        # crate food
        x,y = np.random.randint(0, grid_size-1, size=n_food_spots), np.random.randint(0, grid_size-1, size=n_food_spots)
        self.world_cells[x, y, 0] = 1

        # spawn ants
        self.ants = [Ant(anthill_loc, self.world_cells) for i in range(n_ants)]

        # For plotting
        self.define_cmap()
        self.frame = cv2.namedWindow("frame")
    
    def define_cmap(self):
        self.food_cmap = copy.copy(plt.cm.get_cmap('Reds')) # get a copy of the gray color map
        self.food_cmap.set_bad(alpha=0, color="k") # set how the colormap handles 'bad' values
        self.exploration_phero_cmap = copy.copy(plt.cm.get_cmap('Blues')) 
        self.exploration_phero_cmap.set_bad(alpha=0, color="k") 
        self.returning_phero_cmap = copy.copy(plt.cm.get_cmap('Greens')) 
        self.returning_phero_cmap.set_bad(alpha=0, color="k") 


    def plot_world(self):
        f, ax = plt.subplots()
        # ax.imshow(self.background, alpha=1, cmap="Greys", aspect="equal")
        ax.imshow(self.world_cells, cmap=self.food_cmap, aspect="equal", origin="lower")
      
        for ant in self.ants:
            ax.scatter(ant.pos[0], ant.pos[1], s=5, alpha=.8, color="w")

        ax.scatter(self.anthill_loc[0], self.anthill_loc[1], color="m", s=100)
        ax.set(facecolor=[.8, .2, .2])

    def tick(self, plot):
        # move ants
        for ant in self.ants:
            ant.move()

        # evaporate pheromones
        self.world_cells[:, :, 1] *= (1 - hypers["exploration_phero_evap"])

        if plot:
            self.plot_world()



if __name__=="__main__":
    world = AntWorld()
    
    for i in tqdm(range(300)):
        if i % 10 == 0:
            plot=True
        else: plot=False
        world.tick(plot)

#%%
plt.imshow(world.world_cells, cmap=world.exploration_phero_cmap)

#%%
