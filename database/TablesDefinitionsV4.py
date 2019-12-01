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
class Maze(dj.Manual):
	definition = """
		# stores info about maze designs
		maze_id: int
		maze_name: varchar(128)
		---
		left_path_length: float
		right_path_length: float # length of right path in CM
		center_path_length: float

		left_path_angle: int  # angle of start of path from T
		right_path_angle: int
		center_path_angle: float

		has_catwalk: int # 1 or 0
		maze_image: longblob # numpy array with maze image
		maze_image_binary: longblob # numpy array with maze image binary 
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

	class Shelter(dj.Part):
		definition = """
		-> Session
		---
		shelter: int  # 0 if there is no shelter 1 otherwise
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
		recording_uid: varchar(128)
		-> Session
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
	experiments_to_skip = ['Lambda Maze', 'Model Based', 
							'PathInt2 Close', "Foraging"]

	bodyparts = ['snout', 'neck', 'body', 'tail_base',]
	skeleton = dict(head = ['snout', 'neck'], body_upper=['neck', 'body'],
				body_lower=['body', 'tail_base'], body=['tail_base', 'neck'])

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
			tracking_data: longblob     # pandas dataframe with X,Y,Speed,Dir of mvmt
			x: longblob
			y: longblob
			likelihood: longblob
			speed: longblob
			direction_of_mvmt: longblob
			
		"""

	class BodySegmentData(dj.Part):
		definition = """
			# Store orientaiton, ang vel..
			-> TrackingData
			segment_name: varchar(128)
			---
			bp1: varchar(128)
			bp2: varchar(128)
			orientation: longblob
			angular_velocity: longblob
			likelihood: longblob
		"""
	
	def make(self, key):
		make_trackingdata_table(self, key)


	def get_experiments_in_table(self):
		return set((TrackingData() * Recording() * Session()).fetch("experiment_name"))

@schema
class Explorations(dj.Imported):
	definition = """
		-> Session
		---
		start_frame: int
		end_frame: int
		total_travel: float               # Total distance covered by the mouse
		tot_time_in_shelter: float        # Number of seconds spent in the shelter
		tot_time_on_threat: float         # Number of seconds spent on threat platf
		duration: float                   # Total duration of the exploration in seconds
		median_vel: float                  # median velocity in px/s 
	"""

	def make(self, key):
		make_exploration_table(self, key)


@schema
class Trials(dj.Imported):
	definition = """
		-> Stimuli
		-> TrackingData
		---
		-> Recording

		out_of_shelter_frame: int
		at_threat_frame: int
		stim_frame: int              # stim frame relative to recording
		out_of_t_frame: int
		at_shelter_frame: int

		escape_duration: float        # duration in seconds
		time_out_of_t: float

		escape_arm: enum('left', "center", "right") 
		origin_arm:  enum('left', "center", "right")        

		fps: int

	"""

	class TrialSessionMetadata(dj.Part):
		definition = """
			-> Trials
			---
			stim_frame_session: int    # frame of the number relative tot the session and not the recording
		"""

	class TrialTracking(dj.Part):
		definition = """
			-> Trials
			---
			body_xy: longblob
			body_speed: longblob
			body_dir_mvmt: longblob
			body_rois: longblob # in which ROI the mouse is at each frame
			body_orientation: longblob
			body_angular_vel: longblob

			head_orientation: longblob
			head_angular_vel: longblob

			snout_xy: longblob
			snout_speed: longblob
			snout_dir_mvmt: longblob

			neck_xy: longblob
			neck_speed: longblob
			neck_dir_mvmt: longblob

			tail_xy: longblob
			tail_speed: longblob
			tail_dir_mvmt: longblob
		"""

	class ThreatTracking(dj.Part):
		definition = """
			-> Trials
			---
			body_xy: longblob
			body_speed: longblob
			body_dir_mvmt: longblob
			body_rois: longblob # in which ROI the mouse is at each frame
			body_orientation: longblob
			body_angular_vel: longblob

			head_orientation: longblob
			head_angular_vel: longblob

			snout_xy: longblob
			snout_speed: longblob
			snout_dir_mvmt: longblob

			neck_xy: longblob
			neck_speed: longblob
			neck_dir_mvmt: longblob

			tail_xy: longblob
			tail_speed: longblob
			tail_dir_mvmt: longblob
		"""

	def make(self, key):
		make_trials_table(self, key)

@schema
class Homings(dj.Manual):
	definition = """
		homing_id: varchar(128) # recording_uid + t_enter time
		---
		-> Session
		-> Recording
		stim_id:  varchar(128)  			# stimulus_uid entry in Stimuli table, if it was spontaneous = "none"
		stim_frame: int						# stim frame relative to threat enter
		fps: int  
		is_trial: enum("true", "false")
		tracking_data: longblob 			# Mx2x4 array with tracking data for each frame, X,Y each body part (snout, neck, body, tail)
		outward_tracking_data: longblob		# same as above, but for shelter -> threat platform
		threat_tracking_data: longblob	    # just tracking when on threat
		
		homing_arm: varchar(128) 					# 0 for left and 1 for right
		outward_arm: varchar(128) 					# 0 for left and 1 for right
		
		time_out_of_t: float 				# time in seconds out of the threat platform
		frame_out_of_t: int 				# time in frames out of the threat platform
		homing_duration: float 				# homing duration in seconds

		last_shelter_exit: int  			# frame at which it exited the shelter
		threat_enter: int 					# frame at which it entered the threat
		last_t_exit: int 					# frame at which it last left the threat platform
		first_s_enter: int 					# frame at which it first got back in the shelter
	"""





if __name__ == "__main__": 
	# pass
	Trials.drop()
	# print(Homings())
	# print_erd() 
	# plt.show()
	# Recording.drop()