import sys
sys.path.append('./')

import numpy as np
from scipy import misc, signal, stats
import pandas as pd
from scipy.spatial import distance
from math import factorial, atan2, degrees, acos, sqrt, pi
import math
import matplotlib.pyplot as plt
from scipy.signal import medfilt as median_filter
from scipy.interpolate import interp1d
from collections import namedtuple
try:
	from sklearn import preprocessing
except: pass

try:
	import skfmm
except:
	print("didnt import skfmm")


# ! ARRAY NORMALISATION
def interpolate_nans(A):
	nan = np.nan
	ok = ~np.isnan(A)
	xp = ok.ravel().nonzero()[0]
	fp = A[~np.isnan(A)]
	x  = np.isnan(A).ravel().nonzero()[0]

	A[np.isnan(A)] = np.interp(x, xp, fp)

	return A

def remove_nan_1d_arr(arr):
	nan_idxs = [i for i,x in enumerate(arr) if np.isnan(x)]
	return np.delete(arr, nan_idxs)
	
def normalise_to_val_at_idx(arr, idx):
	arr = remove_nan_1d_arr(arr)
	val = arr[idx]
	return arr / arr[idx]

def normalise_1d(arr):
	arr = np.array(arr)
	nan_idxs = [i for i,x in enumerate(arr) if np.isnan(x)]
	arr = np.delete(arr, nan_idxs)
	min_max_scaler = preprocessing.MinMaxScaler()
	normed = min_max_scaler.fit_transform(arr.reshape(-1, 1))
	return normed

# ! COLORS
def get_n_colors(n):
	return [plt.get_cmap("tab20")(i) for i in np.arange(n)]

def desaturate_color(c, k=.5):
    # c needs to be an array of 3 floats that specify RGB color
    # k needs to be a float between 0 and 1
    return [cc*k for cc in c]

# ! MOMENTS
def moving_average(arr, window_size):
    cumsum_vec = np.cumsum(np.insert(arr, 0, 0)) 
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def mean_confidence_interval(data, confidence=0.95):
	mean, var, std = stats.bayes_mvs(data)
	res = namedtuple("confidenceinterval", "mean interval_min interval_max")
	return res(mean.statistic, mean.minmax[0], mean.minmax[1])

def percentile_range(data, low=5, high=95):
	"""[Calculates the range between the low and high percentiles]
	"""

	lowp = np.percentile(data, low)
	highp = np.percentile(data, high)
	median = np.median(data)
	res = namedtuple("percentile", "low median high")
	return res(lowp, median, highp)

# ! MISC
def fill_nans_interpolate(y, pkind='linear'):
	"""
	Interpolates data to fill nan values

	Parameters:
		y : nd array 
			source data with np.NaN values

	Returns:
		nd array 
			resulting data with interpolated values instead of nans
	"""
	aindexes = np.arange(y.shape[0])
	agood_indexes, = np.where(np.isfinite(y))
	f = interp1d(agood_indexes
			, y[agood_indexes]
			, bounds_error=False
			, copy=False
			, fill_value="extrapolate"
			, kind=pkind)
	return f(aindexes)

def calc_IdPhi(phi):
	dPhi = abs(np.diff(phi))
	IdPhi = np.sum(dPhi) # /len(dPhi)
	return IdPhi

def calc_LogIdPhi(phi):
	return math.log(calc_IdPhi(phi))

def get_roi_enters_exits(roi_tracking, roi_id):
	"""get_roi_enters_exits [Get all the timepoints in which mouse enters or exits a specific roi]
	
	Arguments:
		roi_tracking {[np.array]} -- [1D array with ROI ID at each frame]
		roi_id {[int]} -- [roi of interest]
	"""

	in_roi = np.where(roi_tracking == roi_id)[0]
	temp = np.zeros(roi_tracking.shape[0])
	temp[in_roi] = 1
	enter_exit = np.diff(temp)  # 1 when the mouse enters the platform an 0 otherwise
	enters, exits = np.where(enter_exit>0)[0], np.where(enter_exit<0)[0]
	return enters, exits

def turning_points(array):
	''' turning_points(array) -> min_indices, max_indices
	Finds the turning points within an 1D array and returns the indices of the minimum and 
	maximum turning points in two separate lists.
	'''
	idx_max, idx_min = [], []
	if (len(array) < 3):
		return idx_min, idx_max

	NEUTRAL, RISING, FALLING = range(3)

	def get_state(a, b):
		if a < b:
			return RISING
		if a > b:
			return FALLING
		return NEUTRAL

	ps = get_state(array[0], array[1])
	begin = 1
	for i in range(2, len(array)):
		s = get_state(array[i - 1], array[i])
		if s != NEUTRAL:
			if ps != NEUTRAL and ps != s:
				if s == FALLING:
					idx_max.append((begin + i - 1) // 2)
				else:
					idx_min.append((begin + i - 1) // 2)
			begin = i
			ps = s
	return idx_min, idx_max


# ! PROBABILITIES
def calc_prob_item_in_list(ls, it):
	"""[Calculates the frequency of occurences of item in list]
	
	Arguments:
		ls {[list]} -- [list of items]
		it {[int, array, str]} -- [items]
	"""

	n_items = len(ls)
	n_occurrences = len([x for x in ls if x == it])
	return n_occurrences/n_items


# ! ERROR CORRECTION
def correct_speed(speed):
	speed = speed.copy()
	perc99 = np.percentile(speed, 99.5)
	speed[speed>perc99] = perc99
	return median_filter(speed, 31)

def remove_tracking_errors(tracking, debug = False):
	"""
		Get timepoints in which the velocity of a bp tracking is too high and remove them
	"""
	filtered = np.zeros(tracking.shape)
	for i in np.arange(tracking.shape[1]):
		temp = tracking[:, i].copy()
		if i <2:
			temp[temp < 10] = np.nan
		filtered[:, i] = signal.medfilt(temp, kernel_size  = 5)

		if debug:
			plt.figure()
			plt.plot(tracking[:, i], color='k', linewidth=2)
			plt.plot(temp, color='g', linewidth=1)
			plt.plot(filtered[:, i], 'o', color='r')
			plt.show()

	return filtered


# ! GEOMETRY
def calc_distane_between_point_and_line(line_points, p3):
	"""[Calcs the perpendicular distance between a point and a line]
	
	Arguments:
		line_points {[list]} -- [list of two 2-by-1 np arrays with the two points that define the line]
		p3 {[np array]} -- [point to calculate the distance from]
	"""
	p1, p2 = np.array(line_points[0]), np.array(line_points[1])
	return np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
	
def calc_distance_between_points_2d(p1, p2):
	'''calc_distance_between_points_2d [summary]
	
	Arguments:
		p1 {[list, array]} -- [X,Y for point one]
		p2 {[list, array]} -- [X,Y for point two]
	
	Returns:
		[float] -- [eucliden distance]

	Test: - to check : print(zero, oneh, negoneh)
	>>> zero = calc_distance_between_points_2d([0, 0], [0, 0])
	>>> oneh = calc_distance_between_points_2d([0, 0], [100, 0])
	>>> negoneh = calc_distance_between_points_2d([-100, 0], [0, 0])
	'''

	return distance.euclidean(p1, p2)

def calc_distance_between_points_in_a_vector_2d(v1):
	'''calc_distance_between_points_in_a_vector_2d [for each pairwise p1,p2 in the two vectors get distnace]
	
	Arguments:
		v1 {[np.array]} -- [2d array, X,Y position at various timepoints]
	
	Raises:
		ValueError -- [description]
	
	Returns:
		[np.array] -- [1d array with distance at each timepoint]

	>>> v1 = [0, 10, 25, 50, 100]
	>>> d = calc_distance_between_points_in_a_vector_2d(v1)
	'''
	# Check data format
	if isinstance(v1, dict) or not np.any(v1) or v1 is None:
			raise ValueError(
				'Feature not implemented: cant handle with data format passed to this function')


	# loop over each pair of points and extract distances
	dist = []
	for n, pos in enumerate(v1):
		# Get a pair of points
		if n == 0:  # get the position at time 0, velocity is 0
			p0 = pos
			dist.append(0)
		else:
			p1 = pos  # get position at current frame

			# Calc distance
			dist.append(np.abs(distance.euclidean(p0, p1)))

			# Prepare for next iteration, current position becomes the old one and repeat
			p0 = p1

	return np.array(dist)

def calc_distance_between_points_two_vectors_2d(v1, v2):
	'''calc_distance_between_points_two_vectors_2d [pairwise distance between vectors points]
	
	Arguments:
		v1 {[np.array]} -- [description]
		v2 {[type]} -- [description]
	
	Raises:
		ValueError -- [description]
		ValueError -- [description]
		ValueError -- [description]
	
	Returns:
		[type] -- [description]

	testing:
	>>> v1 = np.zeros((2, 5))
	>>> v2 = np.zeros((2, 5))
	>>> v2[1, :]  = [0, 10, 25, 50, 100]
	>>> d = calc_distance_between_points_two_vectors_2d(v1.T, v2.T)
	'''
	# Check dataformats
	if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
		raise ValueError('Invalid argument data format')
	if not v1.shape[1] == 2 or not v2.shape[1] == 2:
		raise ValueError('Invalid shape for input arrays')
	if not v1.shape[0] == v2.shape[0]:
		raise ValueError('Error: input arrays should have the same length')

	# Calculate distance
	if v1.shape[1]<20000 and v1.shape[0]<20000: 
		# For short vectors use cdist
		dist = distance.cdist(v1, v2, 'euclidean')
		dist = dist[:, 0]  
	else:
		dist = [calc_distance_between_points_2d(p1, p2) for p1, p2 in zip(v1, v2)]
	return dist

def calc_distance_from_shelter(v, shelter):
	"""[Calculates the euclidean distance from the shelter at each timepoint]
	
	Arguments:
		v {[np.ndarray]} -- [2D array with XY coordinates]
		shelter {[tuple]} -- [tuple of length 2 with X and Y coordinates of shelter]
	"""
	assert isinstance(v, np.ndarray), 'Input data needs to be a numpy array'
	assert v.shape[1] == 2, 'Input array must be a 2d array with two columns'

	shelter_vector = np.array(shelter)
	shelter_vector = np.tile(shelter_vector, (v.shape[0], 1))
	return calc_distance_between_points_two_vectors_2d(v, shelter_vector)

def angle_between_points_2d_clockwise(p1, p2):
	'''angle_between_points_2d_clockwise [Determines the angle of a straight line drawn between point one and two. 
		The number returned, which is a double in degrees, tells us how much we have to rotate
		a horizontal line anti-clockwise for it to match the line between the two points.]

	Arguments:
		p1 {[np.ndarray, list]} -- np.array or list [ with the X and Y coordinates of the point]
		p2 {[np.ndarray, list]} -- np.array or list [ with the X and Y coordinates of the point]
	
	Returns:
		[int] -- [clockwise angle between p1, p2 using the inner product and the deterinant of the two vectors]

	Testing:  - to check:     print(zero, ninety, oneeighty, twoseventy)
		>>> zero = angle_between_points_2d_clockwise([0, 1], [0, 1])
		>>> ninety = angle_between_points_2d_clockwise([1, 0], [0, 1])
		>>> oneeighty = angle_between_points_2d_clockwise([0, -1], [0, 1])
		>>> twoseventy = angle_between_points_2d_clockwise([-1, 0], [0, 1])
		>>> ninety2 = angle_between_points_2d_clockwise([10, 0], [10, 1])
		>>> print(ninety2)
	'''

	"""
		Determines the angle of a straight line drawn between point one and two. 
		The number returned, which is a double in degrees, tells us how much we have to rotate
		a horizontal line anit-clockwise for it to match the line between the two points.
	"""

	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	ang = degrees(atan2(yDiff, xDiff))
	if ang < 0: ang += 360
	# if not 0 <= ang <+ 360: raise ValueError('Ang was not computed correctly')
	return ang

	# ! old code
	""" This old code below copmutes the angle within the lines that go from the origin to p1 and p2, not the angle of the line to which p1 and p2 belong to
	"""

def calc_angle_between_points_of_vector(v):
	"""calc_angle_between_points_of_vector [Given one 2d array of XY coordinates as a function of T
	calculates the angle theta between the coordintes at one time point and the next]
	
	Arguments:
		v1 {[np.array]} -- [2D array of XY coordinates as a function of time]
	"""

	assert isinstance(v, np.ndarray), 'Input data needs to be a numpy array'
	assert v.shape[1] == 2, 'Input array must be a 2d array with two columns'

	thetas = np.zeros(v.shape[0])
	for i in range(v.shape[0]):
		try: # Get current and previous time points coordinates
			p0, p1 = v[i-1,:], v[i, :]
		except:
			thetas[i] = 0
		else:
			d = calc_distance_between_points_2d(p0, p1)
			if d >= 1:
				try:
					thetas[i] = angle_between_points_2d_clockwise(p0, p1)
				except:
					print('Failed with d: ', d)
					thetas[i] = 0
			else:
				thetas[i] = 0
	return thetas

def calc_angle_between_vectors_of_points_2d(v1, v2):
	'''calc_angle_between_vectors_of_points_2d [calculates the clockwise angle between each set of point for two 2d arrays of points]
	
	Arguments:
		v1 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]
		v2 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]

	Returns:
		[np.ndarray] -- [1d array with clockwise angle between pairwise points in v1,v2]
	
	Testing:
	>>> v1 = np.zeros((2, 4))
	>>> v1[1, :] = [1, 1, 1, 1, ]
	>>> v2 = np.zeros((2, 4))
	>>> v2[0, :] = [0, 1, 0, -1]
	>>> v2[1, :] = [1, 0, -1, 0]
	>>> a = calc_angle_between_vectors_of_points_2d(v2, v1)
	'''

	# Check data format
	if v1 is None or v2 is None or not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
		raise ValueError('Invalid format for input arguments')
	if len(v1) != len(v2): 
		raise ValueError('Input arrays should have the same length, instead: ', len(v1), len(v2))
	if not v1.shape[0] == 2 or not v2.shape[0] == 2:
		raise ValueError('Invalid shape for input arrays: ', v1.shape, v2.shape)

	# Calculate
	n_points = v1.shape[1]
	angs = np.zeros(n_points)
	for i in range(v1.shape[1]):
		p1, p2 = v1[:, i], v2[:, i]
		angs[i] = angle_between_points_2d_clockwise(p1, p2)

	return angs

def calc_ang_velocity(angles):
	'''calc_ang_velocity [calculates the angular velocity ]
	
	Arguments:
		angles {[np.ndarray]} -- [1d array with a timeseries of angles in degrees]
	
	Returns:
		[np.ndarray] -- [1d array with the angular velocity in degrees at each timepoint]
	
	testing:
	>>> v = calc_ang_velocity([0, 10, 100, 50, 10, 0])    
	'''
	# Check input data
	if angles is None or not np.any(angles):
		raise ValueError('Invalid input data format')
	if not isinstance(angles, np.ndarray) and not isinstance(angles, list):
		raise ValueError('Invalid input data format')

	# Calculate
	angles_radis = np.radians(angles) # <- to unwrap
	ang_vel_rads = np.insert(np.diff(np.unwrap(angles_radis)), 0, 0)
	return np.degrees(ang_vel_rads)

def geodist(maze, shelter):
	"""[Calculates the geodesic distance from the shelter at each location of the maze]
	
	Arguments:
		maze {[np.ndarray]} -- [maze as 2d array]
		shelter {[np.ndarray]} -- [coordinates of the shelter]

	"""
	phi = np.ones_like(maze)
	mask = (maze == 0)
	masked_maze = np.ma.MaskedArray(phi, mask)

	masked_maze[shelter[1], shelter[0]] = 0
	# time = skfmm.travel_time(masked_maze, speed = 3.0 * np.ones_like(masked_maze))

	distance_from_shelter = np.array(skfmm.distance(masked_maze))

	distance_from_shelter[distance_from_shelter == 0.] =  np.nan
	distance_from_shelter[shelter[1], shelter[0]] = 0


	return distance_from_shelter

