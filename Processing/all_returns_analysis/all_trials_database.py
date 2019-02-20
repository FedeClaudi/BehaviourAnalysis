import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d

class analyse_all_trals:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
    """

    def __init__(self, erase_table=False, fill_in_table=False):
        self.debug = False   # Plot stuff to check things
        if erase_table:
            self.erase_table()

        # ! arbritary
        self.duration_lims = dict(Left_Far=19,
                                    Left_Medium=12,
                                    Centre=8,
                                    Right_Medium=12,
                                    Right_Far=19,
                                    Right2=22,
                                    Left2=22)

        self.naughty_experiments = ['Lambda Maze', ]
        self.good_experiments = ['PathInt', 'PathInt2', 'Square Maze', 'TwoAndahalf Maze', 'PathInt', 'FlipFlop Maze', 'FlipFlop2 Maze',
                                    "PathInt2 D", "PathInt2 DL", "PathInt2 L", 'PathInt2-L', 'TwoArmsLong Maze', "FourArms Maze"]


        if fill_in_table:  # Get tracking data
            self.table = AllTrials()
            self.fill()


    def erase_table(self):
        """ drops table from DataJoint database """
        AllTrials.drop()
        print('Table erased, exiting...')
        sys.exit()

    """
        trial_id: int
        ---
        session_uid: int
        recording_uid: varchar(128)
        experiment_name: varchar(128)
        tracking_data: longblob
        stim_frame: int
        stim_type: enum('audio', 'visual')
        stim_duration: int

        is_escape: enum('true', 'false')

        scape_arm: enum('Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2', 'Left2') 
        origin_arm:  enum('Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2', 'Left2')

    """


    def fill(self):
        """
            Loop over each session
            get the stimuli
            extract trial info
                a trial is defined as the time between a stim and the first of these options
                    - the next stim
                    - mouse got to shelter
                    - 30s elapsed
                    - recording finished
        """
        sessions, session_names, experiments = (Sessions).fetch("uid","session_name", "experiment_name")
        sessions_in_table = [int(s) for s in (AllTrials).fetch("session_uid")]

        for n, (uid, sess_name, exp) in enumerate(zip(sessions, session_names, experiments)):
            print(' Processing session {} of {} - {}'.format(n, len(sessions), sess_name))

            session_stims = get_stimuli_given_sessuid(uid)
            number_of_stimuli = len(session_stims)

            for stim_n, stim in enumerate(session_stims):
                # Get the tracking data for the stimulus recordings
                bps = ['body', 'snout']
                rec_tracking = {bp: get_tracking_given_recuid(stim['recording_uid'], just_body=False, , bp=bp)['tracking_data']
                                 for bp in bps}

                # Get video FPS
                fps = get_videometadata_given_recuid(stim['recording_uid'])[0]

                # Get frame at which stim start
                if 'stim_start' in stim.keys():
                    start = stim['stim_start']
                else:
                    start = stim['overview_frame']

                # Get stim duration
                if 'stim_duration' in stim.keys():
                    stim_duration = stim['stim_duration']
                else:
                    stim_duration = stim['duration']

                if start == -1 or stim_duration == .1:
                    continue  # ? placeholder stim entry

                # Get either the frame at which the next stim starts of the recording ends
                if stim_n < number_of_stimuli:
                    next_stim = session_stims[stim_n+1]
                    if 'stim_start' in next_stim.keys():
                        stop = next_stim['stim_start']
                    else:
                        stop = next_stim['overview_frame']
                else:
                    stop = rec_tracking['body'].shape[0]

                # Now we have the max possible length for the trial
                # But check if the mouse got to the shelter first or if 30s elapsed
                if stop - start > 30*fps:  # max duration > 30s
                    stop = start + 30*fps

                # Okay get the tracking data between provisory start and stop
                trial_tracking = {bp:tr[start:stop, :] for bp,tr in rec_tracking.items{}}

                # Now get shelter enters-exits from that tracking
                shelter_enters, shelter_exits = get_roi_enters_exits(trial_tracking['body'][:, -1], 0)

                if shelter_enters: # if we have an enter, crop the tracking accordingly
                    shelter_enter = shelter_enters[0]
                    trial_tracking = {bp:tr[:shelter_enter, :] for bp,tr in trial_tracking.items()}

                # Get arm of escape
                escape_arm = get_arm_given_rois(trial_tracking['body'][:, -1], 'in')

                # Get threat enters and exits
                threat_enters, threat_exits = get_roi_enters_exits(trial_tracking['body'][:, -1], 1)

                if threat_exits:
                    time_to_exit = threat_exits[0]/fps
                else:
                    time_to_exit = -1


                # Get the tracking data up to the stim frame so that we can extract arm of origin
                out_trip_tracking = rec_tracking[:start, :]
                out_shelter_enters, out_shelter_exits = get_roi_enters_exits(out_trip_tracking[:, -1], 0)
                out_trip_tracking = out_trip_tracking[out_shelter_exits[-1], :]

                # Get arm of origin
                origin_arm = get_arm_given_rois(out_trip_tracking[:, -1], 'out')

                # Check if the trial can be considered an escape
                trial_duration = trial_tracking['body'].shape[0]/fps
                if trial_duration <= self.duration_lims[escape_arm]:
                    is_escape = 'true'
                else:
                    is_escape = 'false'

                # Create multidimensionsal np.array for tracking data
                insert_tracking = np.zeros((trial_tracking['body'].shape[0], trial_tracking['body'].shape[1], len(trial_tracking.keys())))
                insert_tracking[:, :, 0] = trial_tracking['body']
                insert_tracking[:, :, 1] = trial_tracking['snout']


                # Insert in table
                try:
                    last_index = (AllTrials).fetch("trip_id")[-1]
                except:
                    last_index = -1

                key = dict(
                    trial_id = last_index+1,
                    session_uid = uid,
                    recording_uid = stim['recording_uid'],
                    experiment_name = exp,
                    tracking_data = insert_tracking
                    stim_frame = start
                    stim_type = stim['stim_type'],
                    stim_duration = stim_duration,
                    is_escape = is_escape,
                    escape_arm = escape_arm,
                    origin_arm = origin_arm,
                    time_to_out_of_T=time_to_exit
                )

                try:
                    self.table.insert1(key)
                    print('Succesfulli inserted: ', trial_id)
                except:
                    print('||| Could not insert  |||', uid, ' - ', stim['recording_uid'])






                















