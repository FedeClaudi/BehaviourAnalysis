import sys
sys.path.append('./')

from Utilities.imports import *
from database.database_toolbox import ToolBox
from database.TablesDefinitionsV4 import *
from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix

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
	poses = 	 [os.path.join(populator.raw_video_folder, p) for p in os.listdir(populator.raw_pose_folder)]
	metadatas =  [os.path.join(populator.raw_video_folder, m) for m in os.listdir(populator.raw_metadata_folder)]
	ais =		 [os.path.join(populator.raw_video_folder, m) for m in os.listdir(populator.raw_ai_folder)]

	recs_in_part_table = recordings.FilePaths.fetch("recording_uid")

	for rec in tqdm(recordings):
		key = dict(rec)
		if key["recording_uid"] in recs_in_part_table: continue  # ? its already in table

		try:
			key['overview_video'] = [v for v in videos if key['recording_uid'] in v and "Threat" not in v and "tdms" not in v][0]
			key['overview_pose'] = [v for v in poses if key['recording_uid'] in v and "_pose" in v and ".h5" in v and "Overview" in v][0]
		except:
			key['overview_video'] = ""
			key['overview_pose'] = ""

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






"""
			! CCM
"""
def make_commoncoordinatematrices_table(table, key):
	from database.TablesDefinitionsV4 import Recording

	"""make_commoncoordinatematrices_table [Allows user to align frame to model
	and stores transform matrix for future use]

	"""
	# TODO make this work for THREAT camera too

	# Get the maze model template
	maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
	maze_model = cv2.resize(maze_model, (1000, 1000))
	maze_model = cv2.cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)

	# Get path to video of first recording
	videopath = (Recording.FilePaths & key).fetch("overview_video")[0]
	if not videopath: return

	# Apply the transorm [Call function that prepares data to feed to Philip's function]
	""" 
		The correction code is from here: https://github.com/BrancoLab/Common-Coordinate-Behaviour
	"""
	matrix, points, top_pad, side_pad = get_matrix(videopath, maze_model=maze_model)
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








if __name__ == "__main__":
	a = 1