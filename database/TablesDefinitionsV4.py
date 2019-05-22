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
	---
	-> Mouse
	date:                   date                # date in the YYYY-MM-DD format
	experiment_name:        varchar(128)        # name of the experiment the session is part of 
	"""
	# ? population method is in populate_database

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
		recording_uid:      varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
		-> Session
		---
		software:           enum('behaviour', 'mantis')
		ai_file_path:       varchar(256)
	""" 
	
	def make(self, key):
		make_recording_table(self, key)

		
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

	class VisualStimuliLogFile(dj.Part):
		definition = """
			-> Stimuli
			---
			filepath:       varchar(128)
		"""

	class VisualStimuliMetadata(dj.Part):
		definition = """
			-> Stimuli
			---
			stim_type:              varchar(128)    # loom, grating...
			modality:               varchar(128)    # linear, exponential. 
			params_file:            varchar(128)    # name of the .yml file with the params
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


@schema
class TrackingData(dj.Imported):
	definition = """
		# store dlc data for bodyparts and body segments
		-> Recording
	"""

	class BodyPartData(dj.Part):
		definition = """
			# stores X,Y,Velocity... for a single bodypart
			-> TrackingData
			bpname: varchar(128)        # name of the bodypart
			---
			tracking_data: longblob     # pandas dataframe with X,Y,Velocity, MazeComponent ... 
		"""


if __name__ == "__main__":
	Recording.drop()
	# print(str(Stimuli))
	# print_erd()
	plt.show()