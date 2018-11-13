import numpy as np
from scipy import misc
import pandas as pd
from scipy.spatial import distance
from math import factorial, atan2, degrees
import math
from math import acos
from math import sqrt
from math import pi


def calc_distance_between_points_2d(p1, p2):
    return distance.euclidean(p1, p2)


def calc_distance_between_points_in_a_vector_2d(v1, v2):
    """
    Calculates the euclidean distance between consecutive point in a 2d array

    v1, v2 = 1d numpy arrays with X and Y position at each timepoint
    can handle pandas dataseries too
    """

    # Check data format
    if isinstance(v1, list) or isinstance(v2, list) or isinstance(v1, dict) or isinstance(v2, dict):
            raise ValueError(
                'Feature not implemented: cant handle with data format passed to this function')

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass

    # loop over each pair of points and extract distances
    dist = []
    for n, pos in enumerate(zip(v1, v2)):
        # Get a pair of points
        if n == 0:  # get the position at time 0, velocity is 0
            p0 = pos
            dist.append(0)
        else:
            p1 = pos  # get position at current frame

            # Calc distance
            try:
                dist.append(np.abs(distance.euclidean(p0, p1)))
            except:
                if np.isnan(p1).any():
                    dist.append(np.nan)

            # Prepare for next iteration, current position becomes the old one and repeat
            p0 = p1

    return np.array(dist)


def calc_distance_between_points_two_vectors_2d(v1, v2):
    """
    Calculates the euclidean distance between pair of points in two 2d arrays

    v1, v2 = 2d numpy arrays, each with X and Y position at each timepoint
    """

    # Check dataformats
    if not isinstance(v1, np.ndarray()) or not isinstance(v2, np.ndarray()):
        raise ValueError('Invalid argument data format')
    if not v1.shape[0] == 2 or not v2.shape[0] == 2:
        raise ValueError('Invalid shape for input arrays')
    if not v1.shape[1] == v2.shape[1]:
        raise ValueError('Error: input arrays should have the same length')

    # Calculate distance
    dist = distance.cdist(v1, v2, 'euclidean')

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

    Testing:
        >>> zero = angle_clockwise([0, 1], [0, 1])
        >>> ninety = angle_clockwise([1, 0], [0, 1])
        >>> oneeighty = angle_clockwise([0, -1], [0, 1])
        >>> twoseventy = angle_clockwise([-1, 0], [0, 1])
        >>> error = angle_clockwise('a', 'b')
        >>> print(zero, ninety, oneeighty, twoseventy)
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


def calc_angle_between_vectors_of_points_2d(v1, v2,):
    '''calc_angle_between_vectors_of_points_2d [calculates the clockwise angle between each set of point for two 2d arrays of points]
    
    Arguments:
        v1 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]
        v2 {[np.ndarray]} -- [2d array with X,Y position at each timepoint]

    Returns:
        [np.ndarray] -- [1d array with clockwise angle between pairwise points in v1,v2]
    '''

    # Check data format
    if v1 is None or v2 is None or not isinstance(v1, np.ndarray()) or not isinstance(v2, np.ndarray()):
        raise ValueError('Invalid format for input arguments')
    if len(v1) != len(v2): 
        raise ValueError('Input arrays should have the same length')
    if not v1.shape[0] == 2 or not v2.shape[0] == 2:
        raise ValueError('Invalid shape for input arrays')

    # Calculate
    angs = np.zeros(len(v1))
    for i, (p1, p2) in enumerate(zip(v1, v2)):
        angs[i] = angle_between_points_2d_clockwise(p1, p2)

    return angs

def calc_ang_velocity(orientation, fps: int=False):
    """
    Given a vector of orientation (degrees) per frame, calculates the velocity as either degrees per frame
    or degrees per second (if fps != False).

    :param orientation: vector of angle values
    :param fps:  frame rate of video the orientation was extracted from
    :return: angular velocity as either deg per sec or deg per frame.
    """
    rad_ori = np.radians(orientation.values)
    rad_ang_vel = np.insert(np.diff(np.unwrap(rad_ori)), 0, 0)

    if not fps:    # return and vel as degrees per frame
        return np.degrees(rad_ang_vel)
    else:          # return and vel as degrees per sec
        return np.degrees(np.multiply(rad_ang_vel, fps))


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)



