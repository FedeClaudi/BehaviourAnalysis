import datajoint as dj

dj.config['Database.host'] = '127.0.0.1'
dj.config['Database.user'] = 'root'
dj.config['Database.password'] = 'simple'

schema = dj.schema('Database', locals())



@schema
class Mice(dj.Manual):
    definition = """
    # Mouse table lists all the mice used and the relevant attributes
      mouse_id: varchar(128)                        # unique mouse id
      ---
      strain:   varchar(128)                        # genetic strain
      dob: varchar(128)                             # mouse date of birth 
      sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
      single_housed: enum('Y', 'N')                 # single housed or group caged
      enriched_cage: enum('Y', 'N')                 # presence of wheel or other stuff in the cage
    """

@schema
class Sessions(dj.Manual):
    definition = """
    # A session is one behavioural experiment performed on one mouse on one day
    uid: smallint     # unique number that defines each session
    name: varchar(128)  # unique name that defines each session - YYMMDD_MOUSEID
    ---
    -> Mice                # mouse used for the experiment
    date: date             # date in the YYYY-MM-DD format
    num_recordings: smallint   # number of recordings performed within that session
    experiment_name: varchar(128)  # name of the experiment the session is part of 
    experimenter: varchar(128)      # name of the person performing the experiment
    """

@schema
class Recordings(dj.Imported):
    definition = """
    # Within one session one may perform several recordings. Each recording has its own video and metadata files
    recording_uid: varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
    ---
    -> Sessions()
    rec_num: smallint       # recording number within that session
    video_file_path: varchar(532) # path to the file storing the video data
    video_file_format: enum('tmds', 'mp4', 'avi)  # format in which the video was recorded
    converted_video_file_path: varchar(532)  # if video was recorded in.tdms and converted to video,where is the video stored
    metadata_file_path: varchar(532) # path to the .tdms file storing the metadata
    """

@schema
class PoseData(dj.Imported):
    definition = """
    # pose data extracted from dlc processing the video of a whole recording
    -> Recordings 
    ---
    data longblob  # pose loaded from an .h5 data file
    """

@schema
class Trials(dj.Imported):
    definition = """
    # Metadata of each trial (e.g. stim type and frame of onset)
    -> Recordings
    uid: varchar(128)  # uniquely identifuing ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
    ---
    stim_type: varchar(128)
    stim_start: int   # number of frame at start of stim
    stim_duration: int   # duration in frames
    """















#
# @schema
# class Test(dj.Manual):
#     definition = """
#     id: varchar(128)    # name
#     data: composite     # data
#
#     """
#
# @schema
# class Mouse(dj.Manual):
#       definition = """
#       # Mouse table lists all the mice used and the relevant attributes
#       mouse_id: varchar(128)                        # unique mouse id
#       ---
#       strain:   varchar(128)                        # genetic strain
#       dob: varchar(128)                             # mouse date of birth
#       sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
#       single_housed: enum('Y', 'N')                 # single housed or group caged
#       enriched_cage: enum('Y', 'N')                 # presence of wheel or other stuff in the cage
#       """
#
# @schema
# class Experiment(dj.Manual):
#     definition = """
#     # Brief description of each experiment [collection of sessions]
#     name: varchar(128)                              # name of the experiment
#     ---
#     description: varchar(1024)                      # brief description
#     notes: varchar(256)                             # path to file containing a more detialed description, url..
#     """
#
# @schema
# class Manipulation(dj.Manual):
#     definition = """
#     # Description of manipulations being performed in experiments
#     name: varchar(128)                              # unique name describing the manipulation
#     ---
#     type: varchar(128)                              # DREADDs, opto...
#     virus: varchar(128)                             # virus being used, if any
#     compund: varchar(128)                           # name of drug used, if any
#     """
#     #    target: "composite   "                            # target area, population, projection..
#
# @schema
# class Surgery(dj.Manual):
#     definition = """
#     # description of surgical procedures
#     name: varchar(128)                              # name identifying surgical procedure
#     ---
#     type: varchar(128)                              # implant, injection..
#     description: varchar(1024)                      # brief description (e.g. location of injection..)
#     """
#
# @schema
# class NeuronalRecording(dj.Manual):
#     definition = """
#     name: varchar(128)                              # name identifying recording
#     ---
#     type: varchar(128)                              # e.g. probe, miniscope
#     tgt_area: varchar(128)                          # area being targeted
#     tgt_pop: varchar(128)                           # neuronal population being targeted
#     """
#     #     -> BehaviourRecording
#
#     #    data_files paths: composite                     # dictionary with paths to relevant files
#    #  other: composite                                # other potentially useful info can be stored as a dict here
#
# @schema
# class Session(dj.Manual):
#     definition = """
#     # One experiment on one day with one mouse
#     id: varchare(128)                               # YYYYMMDD_MOUSEID
#     """
# """
# ---
# date: date                                      # day of the experiment
# software: enum('Mantis', 'Behaviour')           # software used for the experiment
# -> Mouse
# -> Experiment
# -> Manipulation
# """
#
# @schema
# class BehaviourRecording(dj.Manual):
#     definition = """
#     # Each individual recording being performed during a Session
#     id: varchar(128)                                # unique ID: YYYYMMDD_MOUSEID_RECNUM
#     ---
#     number: int                                     # recording number
#     metadata path: varchar(512)                     # path to metadata .tmds
#     video format: enum('mp4', 'avi', 'tmds')        # file format of recorded video
#     -> Session
#
#     """
#     #     data files paths: composite                     # dictionary with path to metadata, video and tracking files
#
#
# @schema
# class BehaviourTrial(dj.Manual):
#     definition = """
#     # Table defining the metadata of each trial
#     id: varchar(129)                                # unique ID: YYYYMMDD_MOUSEID_RECNUM_TRIALNUM
#     ---
#     -> Recording
#     Stimlus_type: varchar(128)                      # loom, ultrasound
#     Stimulus_timestamp: int                         # either timestamp in ms or in frames from start of Recording
#     """
#     #     Stimulus_metadata: composite                    # python dictionary with stim metadata: parameters...
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#









