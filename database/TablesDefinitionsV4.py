import sys
sys.path.append('./')

from Utilities.imports import *

from database.dj_config import print_erd

schema = dj.schema(dbname)

from database.TablePopulateFuncsV4 import *

@schema
class Mouse(dj.Manual):
	definition = """
		# Mouse table lists all the mice used and the relevant attributes
		mouse_id: varchar(128)                        # unique mouse id
		---
		strain:   varchar(128)                        # genetic strain
		sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
	"""

	# ? population method is in populate_database

@schema
class Session(dj.Manual):
	definition = """
	# A session is one behavioural experiment performed on one mouse on one day
	uid: smallint     # unique number that defines each session
	session_name:           varchar(128)        # unique name that defines each session - YYMMDD_MOUSEID
	mouse_id: varchar(128)                        # unique mouse id
	---
	date:                   date                # date in the YYYY-MM-DD format
	experiment_name:        varchar(128)        # name of the experiment the session is part of 
	"""
	# ? population method is in populate_database

	class Metadata(dj.Part):
		definition = """
		-> Session
		---
		maze_type: int  # maze design id
		naive: int      # was the mouse naive
		lights: int     # light on, off, na, or part on part off
		"""
	def get_experiments_in_table(self):
		return set(Session.fetch("experiment_name"))
@schema
class MazeComponents(dj.Imported):
	definition = """
	# stores the position of each maze template for one experiment
	-> Session
	---
	s: longblob  # Shelter platform template position
	t: longblob  # Threat platform
	p1: longblob  # Other platforms
	p2: longblob
	p3: longblob
	p4: longblob
	p5: longblob
	p6: longblob
	b1: longblob  # Bridges
	b2: longblob
	b3: longblob
	b4: longblob
	b5: longblob
	b6: longblob
	b7: longblob
	b8: longblob
	b9: longblob
	b10: longblob
	b11: longblob
	b12: longblob
	b13: longblob
	b14: longblob
	b15: longblob
	"""

	def make(self, key):
		new_key = make_templates_table(key)
		if new_key is not None:
			self.insert1(new_key)


@schema
class CCM(dj.Imported):
	definition = """
	# stores common coordinates matrix for a session
	-> Session
	camera: varchar(32)
	---
	maze_model:             longblob        # 2d array with image used for correction
	correction_matrix:      longblob        # 2x3 Matrix used for correction
	alignment_points:       longblob        # array of X,Y coords of points used for affine transform
	top_pad:                int             # y-shift
	side_pad:               int             # x-shift
	"""
	def make(self, key):
		make_commoncoordinatematrices_table(self, key)

@schema
class Recording(dj.Imported):
	definition = """
		# Within one session one may perform several recordings. Each recording has its own video and metadata files
			uid: smallint     # unique number that defines each session
		session_name:           varchar(128)        # unique name that defines each session - YYMMDD_MOUSEID
		mouse_id: varchar(128)   
		---
		software:           enum('behaviour', 'mantis')
		ai_file_path:       varchar(256)
	""" 
	
	def make(self, key):
		make_recording_table(self, key)

	def get_experiments_in_table(self):
		return set((Session * Recording).fetch("experiment_name"))


	class FilePaths(dj.Part):
		definition = """
			# stores a reference to all the relevant files
			-> Recording
			---
			ai_file_path:           varchar(256)        # path to mantis .tdms file with analog inputs and stims infos
			overview_video:         varchar(256)        # path to the videofile
			threat_video:           varchar(256)        # path to converted .mp4 video, if any, else is same as video filepath    
			overview_pose:          varchar(256)        # path to .h5 pose file
			threat_pose:            varchar(256)        # path to .h5 pose file
			visual_stimuli_log:     varchar(256)        # path to .yml file with the log of which visual stimuli were delivered
		"""

	def make_paths(self, populator):
		fill_in_recording_paths(self, populator)

	class AlignedFrames(dj.Part):
		definition = """
			# Alignes the overview and threat camera frames to facilitate alignement
			-> Recording
			---
			overview_frames_timestamps:     longblob
			threat_frames_timestamps:       longblob 
			aligned_frame_timestamps:       longblob
		"""

	def make_aligned_frames(self):
		fill_in_aligned_frames(self)

@schema
class Stimuli(dj.Imported):
	definition = """
		# Store data about the stimuli delivered in each recording
		-> Recording
		stimulus_uid:       varchar(128)      # uniquely identifying ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
		---
		overview_frame:     int             # frame number in overview camera (of onset)
		overview_frame_off: int
		duration:           float                   # duration in seconds
		stim_type:          varchar(128)         # audio vs visual
		stim_name:          varchar(128)         # name 
	"""

	sampling_rate = 25000

	class VisualStimuliLogFile(dj.Part): # ? This is populated in the make method
		definition = """
			-> Stimuli
			---
			filepath:       varchar(128)
		"""

	class VisualStimuliMetadata(dj.Part):  # ? This is populated separately
		definition = """
			-> Stimuli
			---
			stim_type:              varchar(128)    # loom, grating...
			modality:               varchar(128)    # linear, exponential. 
			time:                   varchar(128)    # time at which the stim was delivered
			units:                  varchar(128)    # are the params defined in degrees, cm ...
	
			start_size:             float       
			end_size:               float
			expansion_time:         float
			on_time:                float
			off_time:               float
	
			color:                  float
			background_color:        float
			contrast:               float
	
			position:               blob
			repeats:                int
			sequence_number:        float           # sequential stim number in the session
		 """

	def insert_placeholder(self, key):
		key['stimulus_uid'] 		= key['recording_uid']+'_0'
		key['duration']  			= -1
		key['stim_type'] 			= 'nan'
		key['overview_frame'] 		= -1
		key['overview_frame_off'] 	= -1
		key['stim_name'] 			= "nan"
		self.insert1(key)

	def make(self, key):
		make_stimuli_table(self, key)

	def make_metadata(self):
		make_visual_stimuli_metadata(self)	
			

@schema
class TrackingData(dj.Imported):
	experiments_to_skip = ['FlipFlop Maze', 'FlipFlop2 Maze', 'FourArms Maze', 'Lambda Maze', 
							'Model Based', 
							'PathInt2 Close', 
							'TwoArmsLong Maze', "Foraging"]

	bodyparts = ['snout', 'neck', 'body', 'tail_base',]

	definition = """
		# store dlc data for bodyparts and body segments
		-> Recording
		camera: 		varchar(32)
		---
	"""

	class BodyPartData(dj.Part):
		definition = """
			# stores X,Y,Velocity... for a single bodypart
			-> TrackingData
			bpname: varchar(128)        # name of the bodypart
			---
			tracking_data: longblob     # pandas dataframe with X,Y,Velocity, MazeComponent ... 
		"""
	def make(self, key):
		make_trackingdata_table(self, key)


	def get_experiments_in_table(self):
		return set((TrackingData() * Recording() * Session()).fetch("experiment_name"))

@schema
class AllExplorations(dj.Manual):
	definition = """
		exploration_id: int
		---
		session_uid: int
		experiment_name: varchar(128)
		tracking_data: longblob
		total_travel: float               # Total distance covered by the mouse
		tot_time_in_shelter: float        # Number of seconds spent in the shelter
		tot_time_on_threat: float         # Number of seconds spent on threat platf
		duration: float                   # Total duration of the exploration in seconds
		median_vel: float                  # median velocity in px/s 
		session_number_trials: int      # Number of trials in the session following the expl
		exploration_start: int              # frame start exploration
	"""


@schema
class AllTrials(dj.Manual):
	definition = """
		trial_id: int
		---
		session_uid: int
		recording_uid: varchar(128)
		experiment_name: varchar(128)
		tracking_data: longblob
        snout_tracking_data: longblob
		tail_tracking_data: longblob

		outward_tracking_data: longblob

		stim_frame: int
        stim_frame_session: int  # stim frame relative to the start of the session and not the start of the recording
		stim_type: enum('audio', 'visual')
		stim_duration: int

		number_of_trials: int
		trial_number: int

		is_escape: enum('true', 'false')
		escape_arm: enum('Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2', 'Left2', 'nan', 'alpha0', 'alpha1', 'beta0', 'beta1', 'lambda') 
		origin_arm:  enum('Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2', 'Left2', 'nan', 'alpha0', 'alpha1', 'beta0', 'beta1', 'lambda')         
		time_out_of_t: float
		fps: int
		escape_duration: float        # duration in seconds

		threat_exits: longblob
	"""








if __name__ == "__main__":
	AllTrials.drop()
	# print_erd()
	# plt.show()