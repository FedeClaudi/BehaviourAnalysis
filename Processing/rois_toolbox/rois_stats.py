import sys
sys.path.append('./')  
from Utilities.file_io.files_load_save import load_yaml
import numpy as np
import cv2
from collections import namedtuple
import matplotlib.pyplot as plt
import os

"""
    Functions to extract time spent by the mouse in each of a list of user defined ROIS 

    Example usage:
    rois        -->  a dictionary with name and position of each roi
    tracking    -->  a pandas dataframe with X,Y,Velocity for each bodypart
    bodyparts   -->  a list with the name of all the bodyparts
    
    -----------------------------------------------------------------------------------
    
    from collections import namedtuple
    
    data = namedtuple('tracking data', 'x y velocity')
    results = {}
    for bp in bodyparts:
        bp_tracking = data(tracking.bp.x.values, tracking.bp.y.values, tracking.bp.Velocity.values)
        res = get_timeinrois_stats(bp_tracking, roi, fps=30)
        results[bp] = res
    
"""

def convert_roi_id_to_tag(ids):
    rois_lookup = load_yaml('Processing/rois_toolbox/rois_lookup.yml')
    rois_lookup = {v:k for k,v in rois_lookup.items()}
    return [rois_lookup[int(r)] for r in ids]



def get_arm_given_rois(rois, direction):
        """
            Get arm of origin or escape given the ROIs the mouse has been in
            direction: str, eitehr 'in' or 'out' for outward and inward legs of the trip
        """
        rois_copy = rois.copy()
        rois = [r for r in rois if r not in ['t', 's']]

        if not rois:
            return None

        if direction == 'out':
            vir = rois[-1]  # very important roi
        elif direction == "in":
            vir = rois[0]

        if 'b15' in rois:
            return 'Centre'
        elif vir == 'b13':
            return 'Left2'
        elif vir == 'b10':
            if 'p1' in rois or 'b4' in rois:
                return 'Left_Far'
            else:
                return 'Left_Medium'
        elif vir == 'b11':
            if 'p4' in rois or 'b7' in rois:
                return 'Right_Far'
            else:
                return 'Right_Medium'
        elif vir == 'b14':
            return 'Right2'
        else:
            return None


def load_rois(display=False):
    components = load_yaml('Processing/rois_toolbox/template_components.yml')
    rois = {}
    # Get platforms
    for pltf, (center, radius) in components['platforms'].items():
        rois[pltf] = tuple(center)

    # Get bridges
    for bridge, pts in components['bridges'].items():
        x, y = zip(*pts)
        center = (max(x)+min(x))/2., (max(y)+min(y))/2.
        rois[bridge] =  center

    if display:
        [print('\n', n, ' - ', v) for n,v in rois.items()]

def get_roi_at_each_frame(experiment, session_name, bp_data, rois=None):
    """
    Given position data for a bodypart and the position of a list of rois, this function calculates which roi is
    the closest to the bodypart at each frame

    :param bp_data: numpy array: [nframes, 2] -> X,Y position of bodypart at each frame
                    [as extracted by DeepLabCut] --> df.bodypart.values
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                    two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :return: tuple, closest roi to the bodypart at each frame
    """

    def check_roi_tracking_plot(session_name, rois, centers, names, bp_data, roi_at_each_frame):
        save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Maze_templates\\ignored\\Matched'
        rois_ids = {p:i for i,p in enumerate(rois.keys())}
        roi_at_each_frame_int = np.array([rois_ids[r] for r in roi_at_each_frame])

        f, ax = plt.subplots()
        ax.scatter(bp_data[:, 0], bp_data[:, 1], c=roi_at_each_frame_int, alpha=.4)
        for roi, k in zip(centers, names):
            ax.plot(roi[0], roi[1], 'o', label=k)
        ax.legend()
        # plt.show()
        f.savefig(os.path.join(save_fld, session_name+'.png'))

    if rois is None:
        rois = load_rois()
    elif not isinstance(rois, dict): 
        raise ValueError('rois locations should be passed as a dictionary')

    if not isinstance(bp_data, np.ndarray):
            pos = np.zeros((len(bp_data.x), 2))
            pos[:, 0], pos[:, 1] = bp_data.x, bp_data.y
            bp_data = pos

    # Get the center of each roi
    centers, roi_names = [], [] 
    for name, points in rois.items():  # a pointa is  two 2d XY coords for top left and bottom right points of roi
        points = points.values[0]
        if not isinstance(points, np.ndarray): continue # maze component not present in maze for this experiment
        try:
            center_x = points[1] + (points[3] / 2)
        except:
            # raise ValueError('Couldnt find center for points: ',points, type(points))
            center_x = points[0]
            center_y = points[1]
        else:
            center_y = points[0] + (points[2] / 2)
        
        # Need to flip ROIs Y axis to  match tracking
        dist_from_midline = 500 - center_y
        center_y = 500 + dist_from_midline
        center = np.asarray([center_x, center_y])
        centers.append(center)
        roi_names.append(name)

    # Calc distance to each roi for each frame
    data_length = bp_data.shape[0]
    distances = np.zeros((data_length, len(centers)))

    for idx, center in enumerate(centers):
        cnt = np.tile(center, data_length).reshape((data_length, 2))
        dist = np.hypot(np.subtract(cnt[:, 0], bp_data[:, 0]), np.subtract(cnt[:, 1], bp_data[:, 1]))
        distances[:, idx] = dist

    # Get which roi the mouse is in at each frame
    sel_rois = np.argmin(distances, 1)
    roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])
    # print('the mouse has visited these platforms ', set(roi_at_each_frame))
    # print('and has spent this time in shelter ', roi_at_each_frame.count('s'))

    # Check we got cetners correctly
    check_roi_tracking_plot(session_name, rois, centers, roi_names, bp_data, roi_at_each_frame)
    return roi_at_each_frame


def get_timeinrois_stats(data, rois, fps=None):
    """
    Quantify number of times the animal enters a roi, comulative number of frames spend there, comulative time in seconds
    spent in the roi and average velocity while in the roi.

    In which roi the mouse is at a given frame is determined with --> get_roi_at_each_frame()


    Quantify the ammount of time in each  roi and the avg stay in each roi
    :param data: tracking data passed as a namedtuple (x,y,velocity)
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :param fps: framerate at which video was acquired
    :return: dictionary
    """

    def get_indexes(lst, match):
        return np.asarray([i for i, x in enumerate(lst) if x == match])

    # get roi at each frame of data
    data_rois = get_roi_at_each_frame(data, rois)
    data_time_inrois = {name: data_rois.count(name) for name in set(data_rois)}  # total time (frames) in each roi

    # number of enters in each roi
    transitions = [n for i, n in enumerate(list(data_rois)) if i == 0 or n != list(data_rois)[i - 1]]
    transitions_count = {name: transitions.count(name) for name in transitions}

    # avg time spend in each roi (frames)
    avg_time_in_roi = {transits[0]: time / transits[1]
                       for transits, time in zip(transitions_count.items(), data_time_inrois.values())}

    # avg time spend in each roi (seconds)
    if fps is not None:
        data_time_inrois_sec = {name: t / fps for name, t in data_time_inrois.items()}
        avg_time_in_roi_sec = {name: t / fps for name, t in avg_time_in_roi.items()}
    else:
        data_time_inrois_sec, avg_time_in_roi_sec = None, None

    # get avg velocity in each roi
    avg_vel_per_roi = {}
    for name in set(data_rois):
        indexes = get_indexes(data_rois, name)
        vels = [data.velocity[x] for x in indexes]
        avg_vel_per_roi[name] = np.average(np.asarray(vels))

    results = dict(transitions_per_roi=transitions_count,
                   comulative_time_in_roi=data_time_inrois,
                   comulative_time_in_roi_sec=data_time_inrois_sec,
                   avg_time_in_roi=avg_time_in_roi,
                   avg_time_in_roi_sec=avg_time_in_roi_sec,
                   avg_vel_in_roi=avg_vel_per_roi)

    return results


if __name__ == "__main__":
    load_rois(display=True)







