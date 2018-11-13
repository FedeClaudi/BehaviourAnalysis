import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations

from Processing.tracking_stats.math_utils import *

"""
Gets a dataframe correspondings to the tracking for one session in the shape of:

        tracking   
                - bp
                    - x
                    - y
                    - likelyhood
                - bp 2
                    - x
                    - y
                    - likelyhood
                ...

and adds Velocity and Acceleration to each bp


It also takes each pair of bps and calculates the distance between them, the angle relative to the vertical, angular velocity and angular acceleration

"""


def extract_velocities_from_pose(pose):
    '''extract_velocities_from_pose [see above]
    
    Arguments:
        pose {[pd.DataFrane]} -- [multi layer dataframe for pose data from DLC]
    '''

    # For each bodypart extract velocity and acceleration
    for bp in pose:
        distance_between_frames = calc_distance_between_points_in_a_vector_2d(bp.x.values, bp.y.values)

        velocity = np.insert(np.diff(distance_between_frames), 0, 0)  # inserting 0 at the first frame otherwise the vector will be 1 frame short
        acceleration = np.insert(np.diff(velocity), 0, 0)

        bp['velocity'] = velocity
        bp['acceleration'] = acceleration
    
    # For each pair of bodyparts calculate distance, angle and ang vel
    body_parts = pose.keys()
    body_segments = list(combinations(body_parts, 2))

    for bp1_name, bp2_name in body_segments:
        # get segment length, orientation and angular velocity
        bp1, bp2 = pose[bp1_name], pose[bp2_name]
        bp1_pos = np.array([bp1.x.values, bp1.y.values])
        bp2_pos = np.array([bp2.x.values, bp2.y.values])

        segment_length = calc_distance_between_points_two_vectors_2d(bp1_pos, bp2_pos)
        segment_angle = calc_angle_between_vectors_of_points_2d(bp1_pos, bp2_pos)
        segment_angular_velocity = calc_ang_velocity(segment_angle)

        # Add to pandas dataframe for pose
        # ! I don't know how to do this
        # ? create layerd dataframe: pose > segmnet > [length, orientation, ang vel]
        segment_name = '{}-{}'.format(bp1_name, bp2_name)
        pose[segment_name] = 'aaa'   # First layer of pandas dataframe
        # with sublayers: length, angle, angular velocity
        pose[segment_name]['length'] = segment_length
        # ...








        




















        











