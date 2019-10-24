#%%
import sys
sys.path.append('./')

import collections

from Utilities.imports import *

import warnings

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.velocity_analysis import get_expl_speeds



['Lambda Maze',  'FlipFlop Maze', 'FlipFlop2 Maze', 'TwoArmsLong Maze', "FourArms Maze", "Foraging"]

["Psychometric - short", "Psychometric - long", "PathInt2", "PathInt2-L", "PathInt2-D", "PathInt2-L",
	"PathInt2 L", "Square Maze", "TwoAndahalf Maze", "PathInt",  
	"PathInt2 D", "noshelter m1", "shortexploration m1"]

table = Homings()

#%%
# Get all recordings uids
recordings = pd.DataFrame((Recording * Session).fetch())

#%%
# Loop over each recording
for i,rec in recordings.iterrows():
	print("Processing recording: {} [{} of {}]".format(rec.recording_uid, i, len(recordings)))

	# Get tracking and body tracking
	tracking = pd.DataFrame((TrackingData.BodyPartData & "recording_uid='{}'".format(rec.recording_uid)).fetch())
	if not np.any(tracking): 
		continue
	body_tracking = tracking.loc[tracking.bpname == "body"].tracking_data[0]

	tracking_data = np.zeros((len(body_tracking), 4, 4)) # whole tracking data
	for bpi, bp in enumerate(["snout", "neck", "body", "tail_base"]):
		track  = tracking.loc[tracking.bpname == bp].tracking_data.values[0]
		tracking_data[:, :track.shape[1], bpi] = track

	# get video fps
	if rec.uid < 184: fps = 30
	else: fps = 40 # !hardcoded fps

	# Get recording's stimuli
	stimulidf = pd.DataFrame((Stimuli & "recording_uid='{}'".format(rec.recording_uid)).fetch())
	stimuli = {frame:uid for frame,uid in zip(stimulidf['overview_frame'].values, stimulidf['stimulus_uid'].values)}

	# Get times in shelter and times in threat as time series
	in_shelter = np.zeros(len(body_tracking))
	in_shelter[np.where(body_tracking[:, -1] == 0)] = 1
	in_threat = np.zeros(len(body_tracking))
	in_threat[np.where(body_tracking[:, -1] == 1)] = 1

	# Get ROI enters and exits
	shelter_enters, shelter_exits = np.where(np.diff(in_shelter)>0)[0], np.where(np.diff(in_shelter)<0)[0]
	threat_enters, threat_exits = np.where(np.diff(in_threat)>0)[0], np.where(np.diff(in_threat)<0)[0]

	# Loop over each threat enter
	for ten, t_enter in enumerate(threat_enters):
		print("		processing threat enter: {} of {}".format(ten, len(threat_enters)))

		homing_id = rec.recording_uid + "_{}".format(t_enter)
		if homing_id in table.fetch("homing_id"): continue # it's already in table

		key  = {} # <- will be used to make an entry in the table

		# Find last previous shelter exits
		try:
			last_s_exit = [se for se in shelter_exits if se < t_enter][-1]
		except:
			continue # the mouse hasnt been in the shetler yet

		# Find next first shelter enter
		try:
			first_s_enter  = [se for se in shelter_enters if se > t_enter][0]
		except:
			continue # there was no shelter enter after the mouse elft the threat platform

		# If there is another threat enter before the first shelter enter, we want to use that tenter not this one
		following_t_enters = [t for t in threat_enters if t > t_enter and t < first_s_enter]
		if following_t_enters:
			continue # skip this one

		# Find last threat exit before shelter enter
		this_t_exits = [te for te in threat_exits if te < first_s_enter and te >= t_enter] # all T exits before entering shelter
		try:
			last_t_exit = this_t_exits[-1]
		except: 
			continue # there were no threat exits, disregard

		# Check if there was a stimulus, and get the ID
		if not stimuli:
			key['stim_id'] = "none"
			key['stim_frame'] = -1
			key['is_trial'] = 'false'
		else:
			stims_in_time = [sframe for sframe in list(stimuli.keys()) if t_enter <= sframe <= last_t_exit]
			if stims_in_time:
				key['stim_id'] = stimuli[stims_in_time[-1]]
				key['stim_frame'] = stims_in_time[-1] - t_enter
				key["is_trial"] = "true"
			else:
				key['stim_id'] = "none"
				key['stim_frame'] = -1
				key['is_trial'] = 'false'

		# Prep tracking data
		key['outward_tracking_data'] = tracking_data[last_s_exit:t_enter, :, :]
		key['tracking_data'] = tracking_data[t_enter:first_s_enter, :, :]
		key['threat_tracking_data'] = tracking_data[t_enter:last_t_exit, :, :]

		# Get homing and orgin arms
		escape_arm = get_arm_given_rois(convert_roi_id_to_tag(body_tracking[last_t_exit:, -1]), 'in')
		origin_arm = get_arm_given_rois(convert_roi_id_to_tag(body_tracking[:t_enter, -1]), 'out')

		if escape_arm is None:  key['homing_arm'] = "none"
		else: 
			if "left" in escape_arm.lower():
				key['homing_arm'] = "left"
			elif "right" in escape_arm.lower():
				key['homing_arm'] = "right"
			else:
				key['homing_arm'] = escape_arm
		if origin_arm is None:  key['outward_arm'] = "none"
		else: 
			if "left" in origin_arm.lower():
				key['outward_arm'] = "left"
			elif "right" in origin_arm.lower():
				key['outward_arm'] = "right"
			else:
				key['outward_arm'] = origin_arm

		# Add times and frames to key
		# ! all time are relative to when the mouse entered the threat platform!!
		key["homing_id"] = homing_id
		key["uid"] = rec.uid
		key["session_name"] = rec.session_name
		key["recording_uid"] = rec.recording_uid
		key["time_out_of_t"] = (last_t_exit - t_enter)/fps
		key["frame_out_of_t"] = last_t_exit - t_enter
		key["homing_duration"] = (first_s_enter - last_t_exit)/fps # ! this time is relative to shelter exit
		key["last_shelter_exit"] = last_s_exit
		key["threat_enter"] = t_enter
		key["last_t_exit"] = last_t_exit
		key["first_s_enter"] = first_s_enter
		key['fps'] = fps

		table.insert1(key)


#%%
