import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
import time
import random
import yaml

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d

from database.database_fetch import *

from Processing.tracking_stats.math_utils import calc_distance_from_shelter as calc_dist

class Trip:
    def __init__(self, session_uid = None, recording_uid=None):
        """
            fake class used to store data about each shelter - threat - shelter trip
        """
        self.shelter_exit = None        # The frame number at which the mouse leaves the shelter
        self.threat_enter = None        # The frame number at which the mouse enters the therat
        self.threat_exit = None         # The frame number at which the mouse leaves the threat
        self.shelter_enter = None       # The frame number at which the mouse reaches the shelter
        self.time_in_shelter = None     # time spent in the shelter before leaving again    
        self.tracking_data = None       # Numpyu array with X, Y, S, O... at each frame
        self.is_trial = None            # Bool
        self.recording_uid = recording_uid       # Str
        self.duration = None            # Escape duration from levae threat to at shelter in seconds
        self.is_escape = None           # Bool
        self.escape_arm = None           # Str
        self.origin_arm = None
        self.experiment_name = None     # Str
        self.max_speed = None         #
        self.stim_frame = None
        self.stim_type = None
        self.all_threat_exits = None
        self.all_threat_enters = None
        self.session_uid = session_uid
        
    def _as_dict(self):
        return self.__dict__


class analyse_all_trips:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
    """

    def __init__(self, erase_table=False, fill_in_table=False):
        self.debug = False   # Plot stuff to check things

        if erase_table:
            self.erase_table()

        if fill_in_table: # Get tracking data
            self.table = AllTrips()

            # Prepare variables
            self.exclude_first_n_seconds = 120   # exclude the time while I am putting the mouse into the arena 
            self.min_round_trip_diration = 5     # exclude trips that lasted less then N seconds
            self.min_stay_in_threat = 1          # Exclude trips in which the mouse didnt stay on threat for long enough   
            self.min_escape_duration = 1         # exclude trips in which the return journey lasts less then N seconds

            # ! arbritary
            self.duration_lims = dict(Left_Far=12,
                            Left_Medium=6,
                            Centre=4,
                            Right_Medium=6,
                            Right_Far=12,
                            Right2 = 15,
                            Left2 = 15)

            self.naughty_experiments = ['Lambda Maze',]
            self.good_experiments = ['PathInt', 'PathInt2', 'Square Maze', 'TwoAndahalf Maze','PathInt','FlipFlop Maze', 'FlipFlop2 Maze',
                                    "PathInt2 D", "PathInt2 DL", "PathInt2 L", 'PathInt2-L', 'TwoArmsLong Maze', "FourArms Maze"   ]

            # Get all the times the mouse goes from the threat to the shelter
            print('     ... getting trips')
            self.trips = []  # * Store each trip in a list here
            self.get_trips()




    def erase_table(self):
        """ drops table from DataJoint database """
        AllTrips.drop()
        print('Table erased, exiting...')
        sys.exit()






    #####################################################################
    @staticmethod
    def get_arm_given_rois(rois, direction):
        """
            Get arm of origin or escape given the ROIs the mouse has been in
            direction: str, eitehr 'in' or 'out' for outward and inward legs of the trip
        """
        rois_copy = rois.copy()
        rois  = [r for r in rois if r not in ['t', 's']]

        if not rois:
            raise ValueError(rois_copy)

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
        



    #####################################################################
    def get_trips(self):
        sessions, session_names, experiments = (Sessions).fetch("uid","session_name", "experiment_name")
        sessions_in_table = [int(s) for s in (AllTrips).fetch("session_uid")]

        for n, (sess, sess_name, exp) in enumerate(zip(sessions, session_names, experiments)):
            print(' Processing session {} of {} - {}'.format(n, len(sessions), sess_name))
            # Only process sessions from interesting experiments
            if exp in self.naughty_experiments or exp not in self.good_experiments: 
                print('Not processing session because experiment: ', exp)
                continue

            if sess in sessions_in_table:
                print("Session already in table")
                continue

            # Get all the recordings linked with each session
            recordings = get_recordings_given_sessuid(sess, as_dict=True)
            if not recordings:
                print("No recordings found for session, please load recordings first. ")
                continue


            # Loop over each recording and extract the tracking data
            for rec_n, rec in enumerate(recordings):
                print('  ...  ', rec['session_name'])
                rec_uid = rec['recording_uid']
                try:
                    tracking = get_tracking_given_recuid_and_bp(rec_uid, 'body')['tracking_data'].values[0]
                    snout_tracking = get_tracking_given_recuid_and_bp(rec_uid, 'snout')['tracking_data'].values[0]
                except:
                    print("No tracking data found for session, please load tracking data first. ")
                    continue

                # Get video fps
                fps = get_videometadata_given_recuid(rec_uid)[0]

                # Exclude first N seconds of first recording
                if rec_n == 0:
                    exclude_time = self.exclude_first_n_seconds * fps
                    tracking = tracking [exclude_time:, :]

                # Get all the times in which the mouse enters and exits the shelter platform
                in_shelter = np.where(tracking[:, -1] == 0)[0]
                temp = np.zeros(tracking.shape[0])
                temp[in_shelter] = 1
                enter_exit = np.diff(temp)  # 1 when the mouse enters the platform an 0 otherwise
                enters, exits = np.where(enter_exit>0)[0], np.where(enter_exit<0)[0]

                # Remove enters that happened before the first exit, and make sure vectors have same length
                enters = [e for e in enters if e > exits[0]]
                if not np.any(enters) or not np.any(exits):
                    print("No enters of exits found")
                    continue

                if len(enters)>len(exits): 
                    enters = enters[:len(exits)]
                elif len(enters)<len(exits): 
                    exits = exits[:len(enters)]

                # ? DEBUG PLOTS
                if self.debug:
                    f, ax = plt.subplots()
                    model = get_maze_template()
                    ax.imshow(model)
                    ax.plot(tracking[:, 0], tracking[:, 1])
                    ax.scatter(tracking[enters, 0], tracking[enters, 1], c='g')
                    ax.scatter(tracking[exits, 0], tracking[exits, 1], c='r')
                    ax.set(ylim=[0, 1000])

                # Loop over each enter-exit pair and get the corresponding tracking data
                max_trip_duration = np.max(np.subtract(enters, exits))

                trips_tracking = np.full((len(enters), max_trip_duration, 4), np.nan)

                trips = []
                for trip_n, (ex, en) in enumerate(zip(exits, enters)):
                    duration = en-ex
                    # Only keep trips in wich the mouse reach threat platform and that lasted above min threshold
                    if duration/fps > self.min_round_trip_diration:
                        trip = tracking[ex:en, :]
                        snout_trip = snout_tracking[ex:en, :]

                        on_threat = np.where(trip[:, -1]==1)[0]
                        if not np.any(on_threat): continue

                        rois_after_left_threat = tracking[ex + on_threat[-1]: en+5, -1]

                        # Exclude trips in which the escape is too fast or incomplete
                        if on_threat[0] > self.min_escape_duration*fps and trip.shape[0] - on_threat[-1] > self.min_escape_duration*fps and 0.0 in rois_after_left_threat:

                            # Exclude trials in which the moyuse doesn't stay on threat for long enough
                            if len(on_threat) > self.min_stay_in_threat*fps:              
                                trips_tracking[trip_n, :duration, :] = trip[:, [0, 1, 2, -1]]  # store XY, Speed and ROI for all frames in trip

                                # Only some bridges can be taken from S, use this to check that the trip si real and not an error in tracking
                                allowed_bridges_from_shelter = [8.0, 12.0, 22.0, 13.0, 0.0]
                                if trip[1, -1] in allowed_bridges_from_shelter or trip[-2, -1] in allowed_bridges_from_shelter:

                                    # ! We have identified a good trip, get info about it and start populating the Trip() class
                                    if trip_n < len(enters)-1:
                                        time_in_shelter = (exits[trip_n+1] - en)/fps
                                    else:
                                        time_in_shelter = -1

                                    # Merge Snout and Body tracking data
                                    tracking_data = np.zeros((trip.shape[0], 4, 2))
                                    tracking_data[:, :, 0] = trip[:, [0, 1, 2, -1]]
                                    tracking_data[:, :, 1] = snout_trip[:, [0, 1, 2, -1]]

                                    # Get all the times the mouse entered and left the threat platform
                                    temp = np.zeros(trip.shape[0])
                                    temp[on_threat] = 1
                                    t_enter_exit = np.diff(temp)  # 1 when the mouse enters the platform an 0 otherwise
                                    t_enters, t_exits = np.where(t_enter_exit>0)[0], np.where(t_enter_exit<0)[0]
                                    if len(t_enters) != len(t_exits): raise ValueError

                                    # get escape duration
                                    escape_duration = (en - t_exits[-1])/fps

                                    # Create Trip()
                                    _trip = Trip(session_uid = str(sess), recording_uid=rec_uid)
                                    _trip.tracking_data = tracking_data
                                    _trip.shelter_exit, _trip.shelter_enter = ex, en
                                    _trip.threat_enter, _trip.threat_exit =  t_enters[0],  t_exits[-1]
                                    _trip.duration = escape_duration
                                    _trip.time_in_shelter = escape_duration
                                    _trip.max_speed = np.percentile(trip[:, 2], 85)
                                    _trip.experiment_name = exp
                                    _trip.all_threat_exits = t_exits
                                    _trip.all_threat_enters = t_enters
                                    trips.append(_trip)
                                
                # ? DEBUG PLOTS
                if self.debug:
                    f, ax = plt.subplots()
                    model = get_maze_template()
                    ax.imshow(model)

                    for i in np.arange(trips_tracking.shape[0]):
                        ax.scatter(trips_tracking[i, :, 0], trips_tracking[i, :, 1], alpha=.3)

                # * Loop over the trips again but now get arm of origin and escape, if it includes a trial...
                # ? DEBUG PLOTS
                if self.debug:
                        f, ax = plt.subplots()
                for trip_n, trip in enumerate(trips):
                    # Get ROIs for outward and inward trips
                    outw, inw = trip.tracking_data[0:trip.threat_enter, -1, 0], trip.tracking_data[trip.threat_exit-1:, -1, 0]

                    # Convert them to platforms names for clarity
                    rois_lookup = load_yaml('Processing\\rois_toolbox\\rois_lookup.yml')
                    rois_lookup = {v:k for k,v in rois_lookup.items()}
                    outw = [rois_lookup[int(r)] for r in outw]
                    inw = [rois_lookup[int(r)] for r in inw]

                    # ? DEBUG PLOTS
                    if self.debug:
                        model = get_maze_template()
                        ax.imshow(model)
                        ax.scatter(trip.tracking_data[0:trip.threat_enter, 0, 0], trip.tracking_data[0:trip.threat_enter, 1, 0])
                        ax.scatter(trip.tracking_data[trip.threat_exit:, 0, 0], trip.tracking_data[trip.threat_exit:, 1, 0])

                    if not outw or not inw or len(outw) < 6 or len(inw) < 6:
                        raise ValueError

                    # Get arm of origin and escape
                    arm_of_origin = self.get_arm_given_rois(outw, 'out')
                    arm_of_escape = self.get_arm_given_rois(inw, 'in')

                    # Check if a stimulus occured while the mouse was on the t platform
                    stim_frame, stim_type = -1, 'nan' # placeholder valyes in case there is no stim
                    is_trial = 'false'
                    stimuli = get_stimuli_give_recuid(trip.recording_uid)
                    stimz_in_time = []
                    for stim in stimuli:
                        if 'stim_start' in stim.keys():
                            start = stim['stim_start']
                        else:
                            start = stim['overview_frame']

                        # Only consider trials that happened betwee the first time mouse got on T and the first time it left T
                        if (trip.shelter_exit + trip.all_threat_enters[0]) < start < (trip.shelter_exit + trip.all_threat_exits[0]):
                            stimz_in_time.append((start, stim['stim_type']))
                            is_trial = 'true'

                    if stimz_in_time:
                        stim_frame = stimz_in_time[-1][0]
                        stim_type = stimz_in_time[-1][1]

                    # Check if the inward trip was fast enough for it to be classified an escape
                    if arm_of_escape is not None:
                        if trip.duration <= self.duration_lims[arm_of_escape]:
                            is_escape = 'true'
                        else:
                            is_escape = 'false'

                        # * Good, we have we need. Fill in the remaining fields of the trip class
                        trip.escape_arm = arm_of_escape
                        trip.origin_arm = arm_of_origin
                        trip.is_escape = is_escape
                        trip.is_trial = is_trial
                        trip.stim_frame = stim_frame
                        trip.stim_type = stim_type

                self.insert_trips_in_table(trips)


    def insert_trips_in_table(self, trips):
        for trip in trips: 
            key = trip._as_dict() 
            try:
                last_index = (AllTrips).fetch("trip_id")[-1]
            except:
                last_index = -1

            key['trip_id'] = last_index+1
            try:
                self.table.insert1(key)
                print('     inserted: ', key['recording_uid'])
            except:
                print(' !!! - did not insert !!! - ', key['recording_uid'])


    #####################################################################
    #####################################################################
    #####################################################################

    @staticmethod
    def plot_all_trip_by_arm():
        frames_per_escape = 100

        trips = pd.DataFrame(AllTrips .fetch())
        arms = set(trips['escape_arm'].values)

        f, axarr = plt.subplots(2, len(arms))

        for i, arm in enumerate(arms):
            arm_data_escapes = trips.loc[trips['escape_arm'] == arm]
            trackings = arm_data_escapes['tracking_data'].values
            threat_leaves = arm_data_escapes['threat_exit'].values

            for ex, tr in zip(threat_leaves, trackings):
                frames = np.linspace(ex, tr.shape[0]-1, frames_per_escape).astype(np.int16)
                axarr[1, i].scatter(tr[frames, 0], tr[frames, 1], c=tr[frames, -1], alpha=.4)

            arm_data_origins = trips.loc[trips['origin_arm'] == arm]
            trackings = arm_data_origins['tracking_data'].values
            threat_leaves = arm_data_origins['threat_enter'].values

            for ex, tr in zip(threat_leaves, trackings):
                frames = np.linspace(0, ex, frames_per_escape).astype(np.int16)
                axarr[0, i].scatter(tr[frames, 0], tr[frames, 1], c=tr[frames, -1], alpha=.4)

            axarr[0, i].set(title=arm)


    def plt_n_random_trips(self, n = 10):
        while True:
            trips = pd.DataFrame((AllTrips & "is_trial='true'").fetch())

            selected_trips = random.sample(list(np.arange(trips.shape[0])), n)

            trips = trips.reindex(selected_trips)

            f, ax = plt.subplots()
            ax.imshow(get_maze_template())

            for i, row in trips.iterrows():
                
                start, stop = row['stim_frame'] - row['shelter_exit'], row['shelter_enter'] - row['shelter_exit']
                tr = row['tracking_data']
                ax.scatter(tr[start:stop, 0], tr[start:stop, 1],  s=tr[start:stop, 2], alpha=.5)

            ax.set(ylim=[0, 1000])
            plt.show()


def check_all_trials_included(table):
    """
    In principle all trials should be included in All Returns, lets check
    """
    print("checking stuff")
    data = pd.DataFrame(table.fetch())
    recordings = Recordings().fetch("recording_uid")
    stimuli = pd.DataFrame(BehaviourStimuli.fetch())

    
    for i, rec in enumerate(recordings):
        print('Checking rec {} of {}'.format(i, len(recordings)))
        r_stim_times = stimuli.loc[stimuli['recording_uid'] == rec]['stim_start'].values
        rtrips = data.loc[(data['recording_uid'] == rec)&(data['is_trial']=='true')]

        if not np.any(rtrips['recording_uid'].values): continue

        for si, stim_time in enumerate(r_stim_times):
            if stim_time == -1: continue # just a place holder empty stim table entry

            # Check if there is a trial trip that includes it
            entry_in_t_before_stim = np.where(rtrips['threat_enter'].values <= stim_time)[0]

            if not len(entry_in_t_before_stim): 
                # raise ValueError('Didnt find any good time')
                print('     Didnt find any good time')
                continue
            else:
                last_entry_index = entry_in_t_before_stim[-1]

            try:
                prev_enter = rtrips['threat_enter'].values[last_entry_index]
                next_exit = rtrips['threat_exit'].values[last_entry_index]
            except:
                raise ValueError('      smth went wrong')
            else:
                if not prev_enter < stim_time < next_exit: 
                    rec_tracking = get_tracking_given_recuid_and_bp(rtrips['recording_uid'].values[0], 'body')['tracking_data'].values[0]
                    t0, t1 = -600, 600
                    plt.scatter(rec_tracking[stim_time+t0:stim_time+t1, 0], rec_tracking[stim_time+t0:stim_time+t1, 1], c=rec_tracking[stim_time+t0:stim_time+t1, -1])


                    # raise ValueError('Timings wrong')
                    print('     Timings wrong')
                    print('     {} - stim {} of {}'.format(rec, si, len(r_stim_times)))

                    tracking = rtrips['tracking_data'].values[last_entry_index]
                    plt.scatter(tracking[:, 0], tracking[:, 1])

    # If we got here everything was good
    print('All trials are included')

if __name__ == '__main__':

    at = analyse_all_trips(erase_table=False, fill_in_table=True)
    # at.plot_all_trip_by_arm()
    # at.plt_n_random_trips()

    check_all_trials_included(AllTrips())

    plt.show()

