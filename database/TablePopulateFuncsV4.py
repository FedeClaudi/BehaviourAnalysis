import sys
sys.path.append('./')

from Utilities.imports import *
from database.database_toolbox import ToolBox
from database.TablesDefinitionsV4 import *

cur_dir = os.getcwd()
os.chdir("C:\\Users\\Federico\\Documents\\GitHub\\VisualStimuli")
from Utils.contrast_calculator import Calculator as ContrastCalc
os.chdir(cur_dir)

import scipy.signal as signal
from collections import OrderedDict

from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix
from Utilities.maths.stimuli_detection import *
from Utilities.dbase.stim_times_loader import *

from Processing.tracking_stats.correct_tracking import correct_tracking_data
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Processing.tracking_stats.extract_velocities_from_tracking import complete_bp_with_velocity, get_body_segment_stats

"""
			! TEMPLATES 
"""
def make_templates_table(key):
	from database.TablesDefinitionsV4 import Session
	# Load yaml with rois coordinates
	rois = load_yaml("database/maze_components/MazeModelROILocation.yml")

	# Only keep the rois relevant for each experiment
	experiment_name = (Session & key).fetch("experiment_name")[0]
	rois_per_exp = load_yaml("database/maze_components/MazeModelROI_perExperiment.yml")
	rois_per_exp = rois_per_exp[experiment_name]
	selected_rois = {k:(p if k in rois_per_exp else -1)  for k,p in rois.items()}

	# return new key
	return {**key, **selected_rois}


"""
			! RECORDING
"""

def make_recording_table(table, key):
	def behaviour(table, key, software, tb):
		videos, metadatas = tb.get_behaviour_recording_files(key)
		if videos is None: return

		# Loop over the files for each recording and extract info
		for rec_num, (vid, met) in enumerate(zip(videos, metadatas)):
			if vid.split('.')[0].lower() != met.split('.')[0].lower():
				raise ValueError('Files dont match!', vid, met)

			name = vid.split('.')[0]
			try:
				recnum = int(name.split('_')[2])
			except:
				recnum = 1

			if rec_num+1 != recnum:
				raise ValueError('Something went wrong while getting recording number within the session')

			rec_name = key['session_name']+'_'+str(recnum)
			
			# Insert into table
			rec_key = key.copy()
			rec_key['recording_uid'] = rec_name
			rec_key['ai_file_path'] = os.path.join(tb.raw_metadata_folder, met)
			rec_key['software'] = software
			table.insert1(rec_key)

	def mantis(table, key, software, tb):
		# Get AI file and insert in Recordings table
		rec_name = key['session_name']
		aifile = [os.path.join(tb.analog_input_folder, f) for f 
					in os.listdir(tb.analog_input_folder) 
					if rec_name in f]
		if not aifile:
			print("could not find AI file for: ", key)
			return
		else:
			aifile = aifile[0]
		
		key_copy = key.copy()
		key_copy['recording_uid'] = rec_name
		key_copy['software'] = software
		key_copy['ai_file_path'] = aifile
		table.insert1(key_copy)

	# See which software was used and call corresponding function
	# print(' Processing: ', key)
	tb = ToolBox()
	if key['uid'] < 184:
		behaviour(table, key, 'behaviour', tb)
	else:
		mantis(table, key, 'mantis', tb)


def fill_in_recording_paths(recordings, populator):
	# fills in FilePaths table
	videos = 	 [os.path.join(populator.raw_video_folder, v) for v in os.listdir(populator.raw_video_folder)]
	poses = 	 [os.path.join(populator.raw_pose_folder, p) for p in os.listdir(populator.raw_pose_folder)]
	metadatas =  [os.path.join(populator.raw_metadata_folder, m) for m in os.listdir(populator.raw_metadata_folder)]
	ais =		 [os.path.join(populator.raw_ai_folder, m) for m in os.listdir(populator.raw_ai_folder)]

	recs_in_part_table = recordings.FilePaths.fetch("recording_uid")

	for rec in tqdm(recordings):
		key = dict(rec)
		if key["recording_uid"] in recs_in_part_table: continue  # ? its already in table

		try:
			key['overview_video'] = [v for v in videos if key['recording_uid'] in v and "Threat" not in v and "tdms" not in v][0]
			key['overview_pose'] = [v for v in poses if key['recording_uid'] in v and "_pose" in v and ".h5" in v and "Threat" not in v][0]
		except:
			if key["recording_uid"][-1] == "1":
				vids = [v for v in videos if key['recording_uid'][:-2] in v and "Threat" not in v and "tdms" not in v]
				if vids:
					key['overview_video'] = vids[0]
					try:
						key['overview_pose'] = [v for v in poses if key['recording_uid'][-2:] in v and "_pose" in v and ".h5" in v and key["session_name"] in v][0]
					except:
						print("No pose file found for rec: --> ", key["recording_uid"])
						continue
				else:
					continue  # ! remove this
					raise FileNotFoundError(key["recording_uid"])
			else:
				continue  # ! remove this
				raise FileNotFoundError(key["recording_uid"])

		if "Overview" not in key['overview_pose']:
			if key["recording_uid"][-1] != key["overview_pose"].split("_pose")[0][-1]: 
				print("Something went wrong: ", key)
				continue # ! re,pve this
				raise ValueError(key)

		threat_vids = [v for v in videos if key['recording_uid'] in v and "Overview" in v and "mp4" in v]
		if not threat_vids:
			key['threat_video'] = ""
		else:
			key['threat_video'] = threat_vids[0]

		threat_poses = [v for v in poses if key['recording_uid'] in v and "_pose" in v and ".h5" in v and "Overview" in v]
		if not threat_poses:
			key['threat_pose'] = ""
		else:
			key["threat_pose"] = threat_poses[0]

		visual_stim_logs = [v for v in ais if key['recording_uid'] in v and "visual_stimuli_log" in v]
		if not visual_stim_logs:
			key['visual_stimuli_log'] = ""
		else:
			key['visual_stimuli_log'] = [0]

		del key["software"]

		recordings.FilePaths.insert1(key, allow_direct_insert=True)

def fill_in_aligned_frames(recordings):
	from Utilities.dbase.db_data_alignment import ThreatDataProcessing

	recs_in_part_table = recordings.AlignedFrames.fetch("recording_uid")
	for rec in tqdm(recordings):
		key = dict(rec)
		if key["recording_uid"] in recs_in_part_table: continue  # ? its already in table

		tdp = ThreatDataProcessing(recordings.AlignedFrames, key)
		if tdp.feather_file is not None:
			tdp.load_a_feather()
			tdp.process_channel(tdp.threat_ch, "threat")
			tdp.process_channel(tdp.overview_ch, "overview")
			tdp.align_frames()
			tdp.insert_in_table()


"""
			! CCM
"""
def make_commoncoordinatematrices_table(table, key):
	from database.TablesDefinitionsV4 import Recording, Session

	"""make_commoncoordinatematrices_table [Allows user to align frame to model
	and stores transform matrix for future use]

	"""
	# Get the maze model template according to the experiment being processed
	if int(key["uid"]<248) or int(key["uid"]>301):
		old_mode = True
		maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel_old.png')
	else:
		old_mode = False
		maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel_old.png')

	maze_model = cv2.resize(maze_model, (1000, 1000))
	maze_model = cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)

	# Get path to video of first recording
	try:
		videopath = (Recording.FilePaths & key).fetch("overview_video")[0]
	except:
		print("While populating CCM, could not find video, make sure to run recording.make_paths")
		warnings.warn("did not pupulate CCM for : {}".format(key))
		return
		
	if not videopath: 
		print("While populating CCM, could not find video, make sure to run recording.make_paths")
		return

	# Apply the transorm [Call function that prepares data to feed to Philip's function]
	""" 
		The correction code is from here: https://github.com/BrancoLab/Common-Coordinate-Behaviour
	"""

	matrix, points, top_pad, side_pad = get_matrix(videopath, maze_model=maze_model, old_mode=old_mode)
	if matrix is None:   # somenthing went wrong and we didn't get the matrix
		# Maybe the videofile wasn't there
		print('Did not extract matrix for video: ', videopath)
		return

	# Return the updated key
	key['maze_model'] 			= maze_model
	key['correction_matrix'] 	= matrix
	key['alignment_points'] 	= points
	key['top_pad'] 				= top_pad
	key['side_pad'] 			= side_pad
	key['camera'] 				= "overview" # TODO make this work
	table.insert1(key)


"""
					 # ! STIMULI
"""
def make_stimuli_table(table, key):
	from database.TablesDefinitionsV4 import Recording, Session
	from Utilities.video_and_plotting.video_editing import Editor
	tb = ToolBox()

	def make_behaviourstimuli(table, key):
		# Get file paths    
		tdms_path = (Recording.FilePaths & key).fetch1("ai_file_path")
		videopath = (Recording.FilePaths & key).fetch1("overview_video")

		# Get stimuli
		stimuli = tb.extract_behaviour_stimuli(tdms_path)

		# If no sti add empty entry to table to avoid re-loading everyt time pop method called
		if not stimuli:
			table.insert_placeholder(key)
			return
		else:
			# Add in table
			for i, stim in enumerate(stimuli):
				stim_key = key.copy()
				stim_key['stimulus_uid'] = key['recording_uid']+'_{}'.format(i)

				if 'audio' in stim.name: stim_key['duration'] = 9 # ! hardcoded
				else: stim_key['duration']  = 5
				
				stim_key['stim_type'] = stim.type
				stim_key['overview_frame'] = stim.frame
				stim_key['overview_frame_off'] = int(stim.frame + stim_key['duration']*30)  # !hardcoded
				stim_key['stim_name'] = stim.name
				table.insert1(stim_key)


	def make_mantisstimuli(table, key):
		def plot_signals(audio_channel_data, stim_start_times, overview=False, threat=False):
			f, ax = plt.subplots()
			ax.plot(audio_channel_data)
			ax.plot(stim_start_times, audio_channel_data[stim_start_times], 'x', linewidth=.4, label='audio')
			if overview:
				ax.plot(audio_channel_data, label='overview')
			if threat:
				ax.plot(audio_channel_data, label='threat')
			ax.legend()
			ax.set(xlim=[stim_start_times[0]-5000, stim_start_times[0]+5000])

		# Get the FPS for the overview video
		try:
			videofile = (Recording.FilePaths & key).fetch1("overview_video")
		except:
			print("could not find videofile for ", key)
			return
		nframes, width, height, fps = Editor.get_video_params(videofile)

		# Get feather file 
		aifile =(Recording.FilePaths & key).fetch1("ai_file_path")
		fld, ainame = os.path.split(aifile)
		ainame = ainame.split(".")[0]
		feather_file = os.path.join(fld, "as_pandas", ainame+".ft")
		groups_file = os.path.join(fld, "as_pandas",  ainame+"_groups.yml")
		visual_log_file = os.path.join(fld, ainame + "visual_stimuli_log.yml")

		if not os.path.isfile(feather_file) or not os.path.isfile(groups_file):
			# ? load the AI file directly
			# Get stimuli names from the ai file and then stimuli
			tdms_df, cols = tb.open_temp_tdms_as_df(aifile, move=True, skip_df=True)
			groups = tdms_df.groups()
		else:
			# load feather and extract info
			tdms_df = load_feather(feather_file)
			groups = [g.split("'/'")[0][2:] for g in load_yaml(groups_file) if "'/'" in g]

		# Get which stimuli are in the data loaded
		if not isinstance(tdms_df, pd.DataFrame):
			if 'WAVplayer' in groups:
				stimuli_groups = tdms_df.group_channels('WAVplayer')
			elif 'AudioIRLED_analog' in groups:
				stimuli_groups = tdms_df.group_channels('AudioIRLED_analog')
			else:
				stimuli_groups = []
			stimuli = {s.path:s.data[0] for s in stimuli_groups}
		else:
			# stimuli = {g+str(i):0 for i,g in enumerate(groups) if "WAV" in g}
			stimuli = {}

		# See if there are visual stimuli
		if "LDR_signal_AI" in groups: visuals_check = True
		else: visuals_check = False

		# ? If there is no stimuli of any sorts insert a fake place holder to speed up future analysis
		if not len(stimuli.keys()) and not visuals_check:
			# There were no stimuli, let's insert a fake one to avoid loading the same files over and over again
			table.insert_placeholder(key)
			return

		# ? If there are audio stimuli, process them 
		if len(stimuli.keys()):
			if visuals_check: raise NotImplementedError("This wont work like this: if we got visual we got feather, if we got feather this dont work")
			# Get stim times from audio channel data
			if  'AudioFromSpeaker_AI' in groups:
				audio_channel_data = tdms_df.channel_data('AudioFromSpeaker_AI', '0')
				th = 1
			else:
				# First recordings with mantis had different params
				audio_channel_data = tdms_df.channel_data('AudioIRLED_AI', '0')
				th = 1.5
			
			# Find when the stimuli start in the AI data
			stim_start_times = find_audio_stimuli(audio_channel_data, th, table.sampling_rate)

			# Check we found the correct number of peaks
			if not len(stimuli) == len(stim_start_times):
				print('Names - times: ', len(stimuli), len(stim_start_times),stimuli.keys(), stim_start_times)
				sel = input('Which to discard? ["n" if youd rather look at the plot]')
				if not 'n' in sel:
					sel = int(sel)
				else:
					plot_signals(audio_channel_data, stim_start_times)
					plt.show()
					sel = input('Which to discard? ')
				if len(stim_start_times) > len(stimuli):
					stim_start_times = np.delete(stim_start_times, int(sel))
				else:
					del stimuli[list(stimuli.keys())[sel]]

			if not len(stimuli) == len(stim_start_times):
				raise ValueError("oopsies")

			# Go from stim time in number of samples to number of frames
			overview_stimuli_frames = np.round(np.multiply(np.divide(stim_start_times, table.sampling_rate), fps))
			
			# Instert these stimuli into the table
			for i, (stimname, stim_protocol) in enumerate(stimuli.items()):
				stim_key = key.copy()
				stim_key['stimulus_uid'] = stim_key['recording_uid']+'_{}'.format(i)
				stim_key['overview_frame'] = int(overview_stimuli_frames[i])
				stim_key['duration'] = 9 # ! hardcoded
				stim_key['overview_frame_off'] =    int(overview_stimuli_frames[i]) + fps*stim_key['duration'] 
				stim_key['stim_name'] = stimname
				stim_key['stim_type'] = 'audio' 
				table.insert1(stim_key)
			

		# ? if there are any visual stimuli, process them
		if visuals_check:
			# Check how many audio stim were inserted to make sure that "stimulus_uid" table key is correct
			n_audio_stimuli = len(stimuli)

			# Get the stimuli start and ends from the LDR AI signal
			ldr_signal = tdms_df["/'LDR_signal_AI'/'0'"].values
			ldr_stimuli = find_visual_stimuli(ldr_signal, 0.24, table.sampling_rate)
			
			# Get the metadata about the stimuli from the log.yml file
			log_stimuli = load_visual_stim_log(visual_log_file)

			if len(ldr_stimuli) != len(log_stimuli): 
				if len(ldr_stimuli) < len(log_stimuli):
					warnings.warn("Something weird going on, ignoring some of the stims on the visual stimuli log file")
					log_stimuli = log_stimuli.iloc[np.arange(0, len(ldr_stimuli))]
				else:
					raise ValueError("Something went wrong with stimuli detection")

			# Add the start time (in seconds) and end time of each stim to log_stimuli df
			log_stimuli['start_time'] = [s.start/table.sampling_rate for s in ldr_stimuli]
			log_stimuli['end_time'] = [s.end/table.sampling_rate for s in ldr_stimuli]
			log_stimuli['duration'] = log_stimuli['end_time'] - log_stimuli['start_time']

			# Insert the stimuli into the table, these will be used to populate the metadata table separately
			for stim_n, stim in log_stimuli.iterrows():
				stim_key = key.copy()
				stim_key['stimulus_uid'] =          stim_key['recording_uid']+'_{}'.format(stim_n + n_audio_stimuli)  
				stim_key['overview_frame'] =        int(np.round(np.multiply(stim.start_time, fps)))
				stim_key['duration'] =              stim.duration
				stim_key['overview_frame_off'] =    int(stim_key['overview_frame'] + fps*stim_key['duration'])
				stim_key['stim_name'] =             stim.stim_name
				stim_key['stim_type'] =             'visual' 
				table.insert1(stim_key)

				# Keep record of the path to the log file in the part table 
				part_key = key.copy()
				part_key['filepath'] =       visual_log_file
				part_key['stimulus_uid'] =   stim_key['stimulus_uid']
				table.VisualStimuliLogFile.insert1(part_key)

	if int(key["uid"])  < 184: make_behaviourstimuli(table,key)
	else: make_mantisstimuli(table, key)


def make_visual_stimuli_metadata(table):
	stims_already_in_table = table.VisualStimuliMetadata.fetch("stimulus_uid")

	for stim in tqdm(table):
		key = dict(stim)
		if key['stimulus_uid'] in stims_already_in_table: continue # ? dont process it agian you fool

		stim_type = (table & key).fetch1("stim_type")

		if stim_type == "audio": 
			continue # this is only for visualz

		# Load the metadata
		try:
			metadata = load_yaml((table.VisualStimuliLogFile & key).fetch1("filepath"))
		except:
			# print("Could not get metadata for ", key)
			continue

		# Get the stim calculator
		contrast_calculator = ContrastCalc(measurement_file="C:\\Users\\Federico\\Documents\\GitHub\\VisualStimuli\\Utils\\measurements.xlsx")
		
		stim_number = key["stimulus_uid"].split("_")[-1]
		stim_metadata = metadata['Stim {}'.format(stim_number)]
		
		# Convert strings to numners
		stim_meta = {}
		for k,v in stim_metadata.items():
			try:
				stim_meta[k] = float(v)
			except:
				stim_meta[k] = v
			
		if 'background_luminosity' not in stim_meta.keys(): stim_meta['background_luminosity'] = 125 # ! hardcoded
		
		# get contrst
		stim_meta['contrast'] = contrast_calculator.contrast_calc(stim_meta['background_luminosity'], stim_meta['color'])

		# prepare key for insertion into the table
		key['stim_type']             = stim_meta['Stim type']
		key['modality']              = stim_meta['modality']
		key['time']                  = stim_meta['stim_start']
		key['units']                 = stim_meta['units']
		key['start_size']            = stim_meta['start_size']
		key['end_size']              = stim_meta['end_size']
		key['expansion_time']        = stim_meta['expand_time']
		key['on_time']               = stim_meta['on_time']
		key['off_time']              = stim_meta['off_time']
		key['color']                 = stim_meta['color']
		key['background_color']      = stim_meta['background_luminosity']
		key['contrast']              = stim_meta['contrast']
		key['position']              = stim_meta['pos']
		key['repeats']               = stim_meta['repeats']
		key['sequence_number']       = stim_number

		table.insert1(key, allow_direct_insert=True)




"""
			! TRACKING DATA
"""
def make_trackingdata_table(table, key):
	from database.TablesDefinitionsV4 import Recording, Session, CCM, MazeComponents
	# skip experiments that i'm not interested in 
	experiment = (Session & key).fetch1("experiment_name")
	if experiment in table.experiments_to_skip: 
		# print("Skipping experiment: ", experiment)
		return

	# Get videos and CCM
	try:
		vid = (Recording.FilePaths & key).fetch1("overview_video")
	except:
		# Try alternative names for the recording UID
		alt_uids = [key['recording_uid']+"Overview", key['recording_uid']+"_1Overview"]
		vid = None
		for uid in alt_uids:
			newkey = key.copy()
			newkey['recording_uid'] = uid
			try:
				vid = (Recording.FilePaths & newkey).fetch1("overview_video")
			except:
				continue
	
		if vid is None:
			print("Could not fetch video for: ", key)
			return
	ccm = (CCM & key).fetch(format="frame")

	# load pose data
	pose_file  = (Recording.FilePaths & key).fetch1("overview_pose")
	try:
		posedata = pd.read_hdf(pose_file)
	except:
		print("Could not find {}".format(pose_file))
		return

	# Insert entry into MAIN CLASS for this videofile
	key['camera'] = 'overview' 
	table.insert1(key)

	# Get the scorer name and the name of the bodyparts
	first_frame = posedata.iloc[0]
	bodyparts = first_frame.index.levels[1]
	scorer = first_frame.index.levels[0]

	"""
		Loop over bodyparts and populate Bodypart Part table
	"""
	bp_data = {}
	for bp in bodyparts:
		if bp not in table.bodyparts: continue  # skip unwanted body parts
		# Get XY pose and correct with CCM matrix
		xy = posedata[scorer[0], bp].values[:, :2]
		try:
			corrected_data = correct_tracking_data(xy, ccm['correction_matrix'][0], ccm['top_pad'][0], ccm['side_pad'][0], experiment, key['uid'])
		except:
			raise ValueError("Something went wrong while trying to correct tracking data, are you sure you have the CCM for this recording? {}".format(key))
		corrected_data = pd.DataFrame.from_dict({'x':corrected_data[:, 0], 'y':corrected_data[:, 1]})

		# Correct the data
		like = posedata[scorer[0], bp].values[:, 2]
		corrected_data[like < .999999] = np.nan
		corrected_data.x = interpolate_nans(corrected_data.x.values)
		corrected_data.y = interpolate_nans(corrected_data.y.values)
		
		# get velocity
		vel = calc_distance_between_points_in_a_vector_2d(corrected_data.values)

		# Add new vals
		corrected_data['velocity'] = vel

		# If bp is body get the position on the maze
		if 'body' in bp:
			# Get position of maze templates - and shelter
			rois = pd.DataFrame((MazeComponents & key).fetch())

			del rois['uid'], rois['session_name'], 

			# Calcualate in which ROI the body is at each frame - and distance from the shelter
			corrected_data['roi_at_each_frame'] = get_roi_at_each_frame(experiment, key['recording_uid'], corrected_data, dict(rois))  # ? roi name
			rois_ids = {p:i for i,p in enumerate(rois.keys())}  # assign a numeric value to each ROI
			corrected_data['roi_at_each_frame'] = np.array([rois_ids[r] for r in corrected_data['roi_at_each_frame']])
			
		# Insert into part table
		bp_data[bp] = corrected_data
		bpkey = key.copy()
		bpkey['bpname'] = bp
		bpkey['tracking_data'] = corrected_data.values 

		table.BodyPartData.insert1(bpkey)







if __name__ == "__main__":
	a = 1