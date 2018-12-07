import numpy as np
from scipy import misc
import pandas as pd
from scipy.spatial import distance
from math import factorial, atan2, degrees, acos, sqrt, pi
import math


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

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass
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
        dist = dist[0, :]  
    else:
        dist = [calc_distance_between_points_2d(p1, p2) for p1, p2 in zip(v1, v2)]
    return dist


def angle_between_points_2d_clockwise(p1, p2):
    '''angle_between_points_2d_clockwise [summary]
    calculates the clockwise angle between two points and the Y axis
    --> if the determinant of the two vectors is < 0 then p2 is clowise of p1

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
    '''

    def length(v):
        return sqrt(v[0]**2+v[1]**2)

    def dot_product(v, w):
     return v[0]*w[0]+v[1]*w[1]

    def determinant(v, w):
      return v[0]*w[1]-v[1]*w[0]

    def inner_angle(v, w):
        cosx = dot_product(v, w)/(length(v)*length(w))
        rad = acos(cosx)  # in radians
        return rad*180/pi  # returns degrees

    inner = inner_angle(p1, p2)
    det = determinant(p1, p2)
    if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
        inner = 360 - inner
    if inner == 360: 
        inner = 0
    return inner


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
    

def correct_tracking_data(uncorrected, M):

    """[Corrects tracking data (as extracted by DLC) using a transform Matrix obtained via the CommonCoordinateBehaviour
        toolbox. ]

    Arguments:
        uncorrected {[np.ndarray]} -- [n-by-2 or n-by-3 array where n is number of frames and the columns have X,Y and Velocity ]
        M {[np.ndarray]} -- [2-by-3 transformation matrix: https://github.com/BrancoLab/Common-Coordinate-Behaviour]

    Returns:
        corrected {[np.ndarray]} -- [n-by-3 array with corrected X,Y tracking and Velocity data]
    """     

    # Do the correction
    x,y = uncorrected[:, 0], uncorrected[:, 1]
    corrected = np.matmul(np.append(M,np.zeros((1,3)),0), [x, y, 1])

    # Calculate velocity 
    corrected[:, 3] = calc_distance_between_points_in_a_vector_2d(corrected)

    return corrected

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)



