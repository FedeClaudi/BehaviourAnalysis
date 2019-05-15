import sys
sys.path.append('./')

from Utilities.imports import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.velocity_analysis import get_expl_speeds

class analyse_all_trals:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
    """

    def __init__(self, erase_table=False, fill_in_table=False):
        self.debug = False   # Plot stuff to check things
        if erase_table:
            self.erase_table()

        

        self.naughty_experiments = ['Lambda Maze', ]
        self.good_experiments = ['PathInt', 'PathInt2', 'Square Maze', 'TwoAndahalf Maze', 'PathInt', 'FlipFlop Maze', 'FlipFlop2 Maze',
                                    "PathInt2 D", "PathInt2 DL", "PathInt2 L", 'PathInt2-L', 'TwoArmsLong Maze', "FourArms Maze"]


        if fill_in_table:  # Get tracking data
            self.table = AllTrials()
            self.fill()

    def define_duration_limits(self):

        """
            Speed limits not duration limits. A trial is considered an escape if themean of the velocity during the trial
            is bigger than the 95th percentile of the velocity during the exploration on an experiment by experiment basis
        """
        _, self.escape_speed_thresholds, _, _ = get_expl_speeds()



    def erase_table(self):
        """ drops table from DataJoint database """
        AllTrials.drop()
        print('Table erased, exiting...')#
        sys.exit()

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
        # Define the min requirements on speed for trials to be escapes
        self.define_duration_limits()

        sessions, session_names, experiments = (Sessions).fetch("uid","session_name", "experiment_name")
        sessions_in_table = [int(s) for s in (AllTrials).fetch("session_uid")]


        for n, (uid, sess_name, exp) in enumerate(sorted(zip(sessions, session_names, experiments))):

            # if 'model based' not in exp.lower(): 
            #     print('skipped experiment')
            #    continue # ! wfbaefbEWYLBFLbfWLIUBLWEBVKJEBVLJHREAB
            
            print(' Processing session {} of {} - {}'.format(n, len(sessions), sess_name))

            if uid in sessions_in_table: continue

            session_trials = []

            session_stims = get_stimuli_given_sessuid(uid, as_dict=True)
            if session_stims is None:
                print('No stimuli found for session')
                continue

            number_of_stimuli = len(session_stims)

            # Def get the tracking for each recording
            bps = ['body', 'snout', 'left_ear', 'right_ear', 'neck', 'tail_base']
            recordings = set([s['recording_uid'] for s in session_stims])
            recs_trackins = {}
            try:
                for r in recordings:
                    rec_tracking = {bp: get_tracking_given_recuid(r, just_body=False, bp=bp, just_trackin_data=True)[0] for bp in bps}
                    recs_trackins[r] = rec_tracking
            except:
                print("Smth went wrong while getting tracking data, maybe there is no data there, maybe smth else is wrong")
                continue


            for stim_n, stim in enumerate(session_stims):
                print(' ... stim {} of {}'.format(stim_n+1, number_of_stimuli))

                # Get the tracking data for the stimulus recordings
                
                rec_tracking = recs_trackins[stim['recording_uid']]

                # Get video FPS
                fps = get_videometadata_given_recuid(stim['recording_uid'])
                if fps == 0: fps = 40  # ? dabsigebrlgb`kj

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
                    continue  # ? placeholder stim entry%R

                # Get either the frame at which the next stim starts of the recording ends
                if stim_n < number_of_stimuli-1:
                    next_stim = session_stims[stim_n+1]
                    if 'stim_start' in next_stim.keys():
                        temp_stop = next_stim['stim_start']
                    else:
                        temp_stop = next_stim['overview_frame']

                    if temp_stop > start: 
                        stop = temp_stop   # ? if next stim is in next recording it will have a low frame number and that ields
                    else:
                        stop = rec_tracking['body'].shape[0]
                else:
                    stop = rec_tracking['body'].shape[0]

                # Now we have the max possible length for the trial
                # But check if the mouse got to the shelter first or if 30s elapsed
                if stop - start > 20*fps:  # max duration > 30s
                    stop = start + 20*fps

                # Okay get the tracking data between provisory start and stop
                trial_tracking = {bp:remove_tracking_errors(tr[start:stop, :]) for bp,tr in rec_tracking.items()}

                # Now get shelter enters-exits from that tracking
                shelter_enters, shelter_exits = get_roi_enters_exits(trial_tracking['body'][:, -1], 0)

                check_got_at_shelt = False
                if np.any(shelter_enters): # if we have an enter, crop the tracking accordingly
                    check_got_at_shelt = True
                    shelter_enter = shelter_enters[0]
                    trial_tracking = {bp:tr[:shelter_enter, :] for bp,tr in trial_tracking.items()}

                # Get threat enters and exits
                threat_enters, threat_exits = get_roi_enters_exits(trial_tracking['body'][:, -1], 1)

                # Get arm of escape
                if not np.any(threat_exits):
                    t = 0
                else:
                    t = threat_exits[-1]
 
                escape_rois = convert_roi_id_to_tag(trial_tracking['body'][t:, -1]) # ? only look at arm taken since last departure from T

                if not  escape_rois: 
                    raise ValueError
                
                escape_arm = get_arm_given_rois(escape_rois, 'in')

                if np.any(threat_exits):
                    time_to_exit = threat_exits[0]/fps
                else:
                    time_to_exit = -1


                # Get the tracking data up to the stim frame so that we can extract arm of origin
                out_trip_tracking = rec_tracking['body'][:start, :]
                out_shelter_enters, out_shelter_exits = get_roi_enters_exits(out_trip_tracking[:, -1], 0)
                out_trip_tracking = out_trip_tracking[out_shelter_exits[-1]:, :]

                # Get arm of origin
                origin_rois = convert_roi_id_to_tag(out_trip_tracking[:, -1])
                if not origin_rois: raise ValueError
                origin_arm = get_arm_given_rois(origin_rois, 'out')


                # Check if the trial can be considered an escape
                if escape_arm is not None:

                    trial_duration = trial_tracking['body'].shape[0]/fps
                    # if trial_duration <= self.duration_lims[escape_arm] and check_got_at_shelt:
                    #     is_escape = 'true'
                    # else:
                    #     is_escape = 'false'

                    trial_speed = correct_speed(trial_tracking['body'][:, 2])
                    if np.mean(trial_speed)>self.escape_speed_thresholds[exp] and check_got_at_shelt:
                        is_escape = "true"
                    else:
                        is_escape = "false"
                else:
                    is_escape = 'false'
                    trial_duration = -1

                # Create multidimensionsal np.array for tracking data
                useful_dims = [0, 1, 2, -1]
                insert_tracking = np.zeros((trial_tracking['body'].shape[0], len(useful_dims), len(trial_tracking.keys())))
                
                for i, bp in enumerate(bps):
                    insert_tracking[:, :, i] = trial_tracking[bp][:, useful_dims]

                if escape_arm is None:
                    escape_arm = 'nan'
                if origin_arm is None:
                    origin_arm = 'nan'

                # Add to list
                key = dict(
                    session_uid = uid,
                    recording_uid = stim['recording_uid'],
                    experiment_name = exp,
                    tracking_data = insert_tracking,
                    stim_frame = start,
                    stim_type = stim['stim_type'],
                    stim_duration = stim_duration,
                    is_escape = is_escape,
                    escape_arm = escape_arm,
                    origin_arm = origin_arm,
                    time_out_of_t=time_to_exit,
                    fps = fps,
                    number_of_trials = number_of_stimuli,
                    trial_number = stim_n,
                    threat_exits = threat_exits,
                    escape_duration = trial_duration,
                )

                session_trials.append(key)

            self.insert_in_table(session_trials)



    def insert_in_table(self, trials):
        for key in trials:

            last_index = pd.DataFrame(AllTrials.fetch()).shape[0] + 1
            key['trial_id'] = last_index+1
            try:
                self.table.insert1(key)
                # print('Succesfulli inserted: ', key['trial_id'])
            except:
                print('||| Could not insert  |||', key['session_uid'], ' - ', key['recording_uid'])




def check_arm_assignment():
    arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2', 'Left2']
    f, axs = plt.subplots(4, 2)
    axs = axs.flatten()

    n_datapoints = 100

    for arm, ax in zip(arms, axs):
        trackings, threat_exits = (AllTrials & "escape_arm='{}'".format(arm)).fetch("tracking_data", "threat_exits")

        for tr, ex in zip(trackings, threat_exits):
            if not np.any(ex):
                t = 0
            else:
                t = ex[-1]

            xx = np.linspace(t, tr.shape[0]-1, n_datapoints).astype(np.int8)
            ax.scatter(tr[xx, 0], tr[xx, 1], c=tr[xx, 2], alpha=.3)
            ax.set(title=arm, xlim=[0, 1000], ylim=[0, 1000])
            
    plt.show()






if __name__ == "__main__":
    # a = analyse_all_trals(erase_table=True, fill_in_table=False)
    # a = analyse_all_trals(erase_table=False, fill_in_table=True)
    print(AllTrials())










