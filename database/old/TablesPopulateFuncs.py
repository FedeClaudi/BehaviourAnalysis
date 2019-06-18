import sys
sys.path.append('./')

from Utilities.imports import *
<<<<<<< HEAD:database/TablesPopulateFuncs.py

try:
    cur_dir = os.getcwd()
    os.chdir("C:\\Users\\Federico\\Documents\\GitHub\\VisualStimuli")
    from Utils.contrast_calculator import Calculator as ContrastCalc
    os.chdir(cur_dir)
except: pass

from nptdms import TdmsFile
import scipy.signal as signal
from collections import OrderedDict

from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix
from Utilities.Maths.stimuli_detection import *
from Utilities.dbase.stim_times_loader import *

from Processing.tracking_stats.correct_tracking import correct_tracking_data
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Processing.tracking_stats.extract_velocities_from_tracking import complete_bp_with_velocity, get_body_segment_stats


from database.auxillary_tables import *
=======
from database.database_toolbox import ToolBox
>>>>>>> 73e8c3b9e154dbfea99730a9503071dcd7c0148d:database/old/TablesPopulateFuncs.py

""" 
    Collection of functions used to populate the dj.Import and dj.Compute
    tables defined in NewTablesDefinitions.py
"""



""" 
##################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
##################################################
"""

def make_dlcmodels_table(table):
    """make_dlcmodels_table [Fills in dlc models table from dlcmodels.yml. making sure that
    only one model per camera is present in the table]
    
    Arguments:
        table {[class]} -- [dj table]
    """

    names_in_table = table.fetch('model_name')
    cameras_in_table = table.fetch('camera')
    models = load_yaml('dlcmodels.yml')
    
    for model in models.values():
        if model['camera'] in cameras_in_table:
            continue
            # one with the same camera is already present
            # if same name: overwrite
            # else: replace?
            if model['model_name'] in names_in_table:
                var = 'model_name'
                print('A model for camera {} with name {} exists already'.format(model['camera'], model['model_name']))
            else:
                var = 'camera'
                print('A model for camera {} already exists, replace?'.format(model['camera']))
            
            print('Old model: ', (table & '{}={}'.format(var, model[var])))
            print('New model: ', model)
            yn = input('Overwrite? [y/n]')
            if yn != 'y': continue
            else:
                (table & '{}={}'.format(var, model[var]).delete())
                table.insert1(model)
        else:
            table.insert1(model)
    print(table)
    

def make_commoncoordinatematrices_table(table, key, sessions, videofiles, fast_mode=False):
    """make_commoncoordinatematrices_table [Allows user to align frame to model
    and stores transform matrix for future use]
    
    Arguments:
        key {[dict]} -- [key attribute of table autopopulate method from dj]
    """
    if fast_mode: # ? just do one session per day
        # If an entry with the same date exists already, avoid re doing the points mapping
        this_date = [s for s in sessions.fetch(as_dict=True) if s['uid']==key['uid']][0]['date']
        old_entries = [e for e in sessions.fetch(as_dict=True) if e['uid'] in table.fetch('uid')]
        
        if old_entries:
            old_entry = [o for o in old_entries if o['date']==this_date]
            if old_entry:
                old_matrix = [m for m in table.fetch(as_dict=True) if m['uid']==old_entry[0]['uid']][0]
                key['maze_model'] = old_matrix['maze_model']
                key['correction_matrix'] = old_matrix['correction_matrix']
                key['alignment_points'] = old_matrix['alignment_points']
                key['top_pad'] = old_matrix['top_pad']
                key['side_pad'] = old_matrix['top_pad']
                table.insert1(key)
                return

    # Get the maze model template
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    maze_model = cv2.cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)

    # Get path to video of first recording
    rec = [r for r in videofiles if r['session_name']
            == key['session_name'] and r['camera_name']=='overview']

    if not rec:
        print('Did not find recording or videofiles while populating CCM table. Populate recordings and videofiles first! Session: ', key['session_name'])
        return
    else:
        rec = rec[0]
        if not '.' in rec['converted_filepath']:
            videopath = rec['video_filepath']
        else:
            videopath = rec['converted_filepath']

    if 'joined' in videopath:
        raise ValueError


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
    key['maze_model'] = maze_model
    key['correction_matrix'] = matrix
    key['alignment_points'] = points
    key['top_pad'] = top_pad
    key['side_pad'] = side_pad
    table.insert1(key)


def make_templates_table(key, sessions, ccm):
    """[allows user to define ROIs on the standard maze model that match the ROIs]
    """
    # Get all possible components name
    nplatf, nbridges = 6, 15
    platforms = ['p'+str(i) for i in range(1, nplatf + 1)]
    bridges = ['b'+str(i) for i in range(1, nbridges + 1)]
    components = ['s', 't']
    components.extend(platforms)
    components.extend(bridges)

    # Get maze model
    mmc = [m for m in ccm if m['uid'] == key['uid']]
    if not mmc:
        print('Could not find CommonCoordinateBehaviour Matrix for this entry: ', key)
        return
        # raise ValueError(
        #     'Could not find CommonCoordinateBehaviour Matrix for this entry: ', key)
    else:
        model = mmc[0]['maze_model']

    # Load yaml with rois coordinates
    paths = load_yaml('paths.yml')
    rois = load_yaml(paths['maze_model_templates'])

    # Only keep the rois relevant for each experiment
    sessions = pd.DataFrame(sessions().fetch())
    experiment = sessions.loc[sessions['uid'] == key['uid']].experiment_name.values[0]
    rois_per_exp = load_yaml(paths['maze_templates_per_experiment'])
    rois_per_exp = rois_per_exp[experiment]
    selected_rois = {k:(p if k in rois_per_exp else -1)  for k,p in rois.items()}

    # return new key
    return {**key, **selected_rois}


def make_recording_table(table, key):
    def behaviour(table, key, software):
        tb = ToolBox()
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
                raise ValueError(
                    'Something went wrong while getting recording number within the session')

            rec_name = key['session_name']+'_'+str(recnum)
            
            # Insert into table
            rec_key = key.copy()
            rec_key['recording_uid'] = rec_name
            rec_key['ai_file_path'] = os.path.join(tb.raw_metadata_folder, met)
            rec_key['software'] = software
            table.insert1(rec_key)

    def mantis(table, key, software):
        # Get AI file and insert in Recordings table
        tb = ToolBox()
        rec_name = key['session_name']
        aifile = [os.path.join(tb.analog_input_folder, f) for f 
                    in os.listdir(tb.analog_input_folder) 
                    if rec_name in f]
        if not aifile:
            print('aifile not found for session: ', key, '\n\n')
            return
        else:
            aifile = aifile[0]
        
        key_copy = key.copy()
        key_copy['recording_uid'] = rec_name
        key_copy['software'] = software
        key_copy['ai_file_path'] = aifile
        table.insert1(key_copy)

        print('Succesfully inserted into mantis table')

    # See which software was used and call corresponding function
    print(' Processing: ', key)
    if key['uid'] < 184:
        software = 'behaviour'
        behaviour(table, key, software)
    else:
        software = 'mantis'
        mantis(table, key, software)


def make_videofiles_table(table, key, recordings,):
    def make_videometadata_table(filepath, key):
        # Get videometadata
        cap = cv2.VideoCapture(filepath)
        key['tot_frames'], fps, key['frame_height'], key['fps'] = Editor.get_video_params(
            cap)
        key['frame_width'] = np.int(fps)
        key['frame_size'] =  key['frame_width']* key['frame_height']
        key['camera_offset_x'], key['camera_offset_y'] = -1, -1

        # if key['fps'] < 10: raise ValueError('Couldnt get metadata for ', filepath, key)

        return key

    def behaviour(table, key):
        tb  = ToolBox()
        videos, metadatas = tb.get_behaviour_recording_files(key)
        
        if key['recording_uid'].count('_') == 1:
            recnum = 1
        else:
            rec_num = int(key['recording_uid'].split('_')[-1])
        rec_name = key['recording_uid']
        try:
            vid, met = videos[rec_num-1], metadatas[rec_num-1]
            vid, met = os.path.join(tb.raw_video_folder, vid), os.path.join(tb.raw_metadata_folder, vid)
        except:
            raise ValueError('Could not collect video and metadata files:' , rec_num-1, rec_name, videos)
        # Get deeplabcut data
        posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                    if rec_name == os.path.splitext(f)[0].split('_pose')[0] and '.pickle' not in f]
        
        if not posefile:
            new_rec_name = rec_name[:-2]
            posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                        if new_rec_name == os.path.splitext(f)[0].split('_pose')[0] and '.pickle' not in f]

        if not posefile:
            # ! pose file was not found, create entry in incompletevideos table to mark we need dlc analysis on this
            incomplete_key = key.copy()
            incomplete_key['camera_name'] = 'overview'
            incomplete_key['conversion_needed'] = 'false'
            incomplete_key['dlc_needed'] = 'true'
            # try:
            #     videosincomplete.insert1(incomplete_key)
            # except:
            #     print(videosincomplete.describe())
            #     raise ValueError('Could not insert: ', incomplete_key )

            # ? Create dummy posefile name which will be replaced with real one in the future
            vid_name, ext = vid.split('.')
            posefile = vid_name+'_pose'+ '.h5'

        elif len(posefile) > 1:
            raise FileNotFoundError('Found too many pose files: ', posefile)
        else:
            posefile = posefile[0]

        # Insert into Recordings.VideoFiles 
        new_key = key.copy()
        new_key['camera_name'] = 'overview'
        new_key['video_filepath'] = vid
        new_key['converted_filepath'] = 'nan'
        new_key['metadata_filepath'] = 'nan'
        new_key['pose_filepath'] = posefile
        table.insert1(new_key)

        return vid

    def mantis(table, key):
        def insert_for_mantis(table, key, camera, vid, conv, met, pose):
            to_remove = ['tot_frames', 'frame_height', 'frame_width', 'frame_size',
                        'camera_offset_x', 'camera_offset_y', 'fps']
            
            video_key = key.copy()
            video_key['camera_name'] = camera
            video_key['video_filepath'] = vid
            video_key['converted_filepath'] = conv
            video_key['metadata_filepath'] = met
            video_key['pose_filepath'] = pose
            if 'conversion_needed' in video_key.keys():
                del video_key['conversion_needed'], video_key['dlc_needed']
            try:
                kk = tuple(video_key.keys())
                for n in to_remove:
                    if n in kk: del video_key[n]
                table.insert1(video_key)
            except:
                raise ValueError('Could not isnert ', video_key)
            
            metadata_key = make_videometadata_table(video_key['converted_filepath'], key)
            if 'conversion_needed' in metadata_key.keys():
                del metadata_key['conversion_needed'], metadata_key['dlc_needed']
            try:
                table.Metadata.insert1(metadata_key)
            except:
                return
            
        def check_files_correct(ls, name):
            """check_files_correct [check if we found the expected number of files]
            
            Arguments:
                ls {[list]} -- [list of file names]
                name {[str]} -- [name of the type of file we are looking for ]
            
            Raises:
                FileNotFoundError -- [description]
            
            Returns:
                [bool] -- [return true if everuything is fine else is false]
            """

            if not ls:
                print('Did not find ', name)
                return False
            elif len(ls)>1:
                raise FileNotFoundError('Found too many ', name, ls)
            else:
                return True


        #############################################################################################

        tb = ToolBox()  # toolbox

        # Get video Files
        videos = [f for f in os.listdir(tb.raw_video_folder)
                        if 'tdms' in f and key['session_name'] in f]
        
        # Loop and get matching files
        for vid in videos:
            # Get videos
            videoname, ext = vid.split('.')
            converted = [f for f in os.listdir(tb.raw_video_folder)
                        if videoname in f and '.mp4' in f]
            converted_check = check_files_correct(converted, 'converted')
            if converted_check: converted = converted[0]

            metadata = [f for f in os.listdir(tb.raw_metadata_folder)
                        if videoname in f and 'tdms' in f]
            metadata_check = check_files_correct(metadata, 'metadata')
            if not metadata_check: raise FileNotFoundError('Could not find metadata file!!')
            else: metadata = metadata[0]

            posedata = [os.path.splitext(f)[0].split('_pose')[0]+'.h5' 
                        for f in os.listdir(tb.pose_folder)
                        if videoname in f and 'h5' in f]
            pose_check = check_files_correct(posedata, 'pose data')
            if pose_check: posedata = posedata[0]
            
            # Check if anything is missing            
            if not converted_check or not pose_check:
                # ? add dummy files names which will be replaced with real ones in the future
                if not converted_check:
                    converted = videoname+'.mp4'
                if not pose_check:
                    posedata = videoname+'_pose.h5'

            # Get Camera Name and views videos
            if 'Overview' in vid:
                camera = 'overview'
            elif 'Threat' in vid:
                camera = 'threat'
            else:
                raise ValueError('Unexpected videoname ', vid)

            # Insert Main Video (threat or overview) in table
            insert_for_mantis(table, key, camera, os.path.join(tb.raw_video_folder, vid),
                                os.path.join(tb.raw_video_folder, converted),
                                os.path.join(tb.raw_metadata_folder, metadata),
                                os.path.join(tb.pose_folder, posedata))

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################

    print('Processing:  ', key)
    # Call functions to handle insert in main table
    software = [r['software'] for r in recordings.fetch(as_dict=True) if r['recording_uid']==key['recording_uid']][0]
    if not software:
        raise ValueError()
    if software == 'behaviour':
        videopath = behaviour(table, key)
        # Insert into part table
        metadata_key = make_videometadata_table(videopath, key)
        metadata_key['camera_name'] = 'overview'
        table.Metadata.insert1(metadata_key)
    else:
        videopath = mantis(table, key)

    
def make_behaviourstimuli_table(table, key, recordings, videofiles):
    if key['uid'] > 184:
        print(key['recording_uid'], '  was not recorded with behaviour software')
        return
    else:
        print('Extracting stimuli info for recording: ', key['recording_uid'])

    # Get file paths    
    rec = [r for r in recordings.fetch(as_dict=True) if r['recording_uid']==key['recording_uid']][0]
    tdms_path = rec['ai_file_path']
    vid = [v for v in videofiles.fetch(as_dict=True) if v['recording_uid']==key['recording_uid']][0]
    videopath = vid['video_filepath']

    # Get stimuli
    tb = ToolBox()
    stimuli = tb.extract_behaviour_stimuli(tdms_path)

    # If no sti add empty entry to table to avoid re-loading everyt time pop method called
    if not stimuli:
        print('Adding fake placeholder entry')
        stim_key = key.copy()
        stim_key['stimulus_uid'] = key['recording_uid']+'_{}'.format(0)
        stim_key['stim_duration']  = -1
        stim_key['video'] = videopath
        stim_key['stim_type'] = 'nan'
        stim_key['stim_start'] = -1
        stim_key['stim_name'] = -1
        table.insert1(stim_key)
    else:
        # Add in table
        for i, stim in enumerate(stimuli):
            stim_key = key.copy()
            stim_key['stimulus_uid'] = key['recording_uid']+'_{}'.format(i)

            if 'audio' in stim.name: stim_key['stim_duration'] = 9 # ! hardcoded
            else: stim_key['stim_duration']  = 5
            
            stim_key['video'] = videopath
            stim_key['stim_type'] = stim.type
            stim_key['stim_start'] = stim.frame
            stim_key['stim_name'] = stim.name
            table.insert1(stim_key)


def make_mantistimuli_table(table, key, recordings, videofiles):
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

    if key['uid'] <= 184:
        return
    else:
        print('Populating mantis stimuli for: ', key['recording_uid'])

    # ! key param
    sampling_rate = 25000

    # ? Load any AUDIO stim
    # Get the feather file with the AI data and the .yml file with all the AI .tdms group names
    rec = [r for r in recordings if r['recording_uid']==key['recording_uid']][0]

    vids_fps = get_videometadata_given_recuid(key['recording_uid'], just_fps=False)     # Get video metadata for the recording being processed
    fps_overview = vids_fps.loc[vids_fps['camera_name']=='overview'].fps[0]

    aifile = rec['ai_file_path']
    fld, ainame = os.path.split(aifile)
    ainame = ainame.split(".")[0]
    
    feather_file = os.path.join(fld, "as_pandas", ainame+".ft")
    groups_file = os.path.join(fld, "as_pandas",  ainame+"_groups.yml")
    visual_log_file = os.path.join(fld, ainame + "visual_stimuli_log.yml")

    if not os.path.isfile(feather_file) or not os.path.isfile(groups_file):
        print("     Could't file feather or group file for ", ainame)
        # ? load the AI file directly
        # Get stimuli names from the ai file
        tb = ToolBox()
        tdms_df, cols = tb.open_temp_tdms_as_df(aifile, move=True, skip_df=True)

        # Get stimuli
        groups = tdms_df.groups()
    else:
        print(" ... loading feather")
        tdms_df = load_feather(feather_file)
        groups = [g.split("'/'")[0][2:] for g in load_yaml(groups_file) if "'/'" in g]

    if 'WAVplayer' in groups:
        stimuli_groups = tdms_df.group_channels('WAVplayer')
    elif 'AudioIRLED_analog' in groups:
        stimuli_groups = tdms_df.group_channels('AudioIRLED_analog')
    else:
        stimuli_groups = []
    stimuli = {s.path:s.data[0] for s in stimuli_groups}

    if "LDR_signal_AI" in groups: visuals_check = True
    else: visuals_check = False

    # ? If there is no stimuli of any sorts insert a fake place holder to speed up future analysis
    if not len(stimuli.keys()) and not visuals_check:
        # There were no stimuli, let's insert a fake one to avoid loading the same files over and over again
        print('     No stim detected, inserting fake place holder')
        stim_key = key.copy()
        stim_key['stimulus_uid'] = stim_key['recording_uid']+'_{}'.format(0)
        stim_key['overview_frame'] = -1
        stim_key['duration'] = -1 
        stim_key['overview_frame_off'] =  -1
        stim_key['stim_name'] = 'nan'
        stim_key['stim_type'] = 'nan' 

        table.insert1(stim_key)
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
        stim_start_times = find_audio_stimuli(audio_channel_data, th, sampling_rate)

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
                np.delete(stim_start_times, int(sel))
            else:
                del stimuli[list(stimuli.keys())[sel]]

        if not len(stimuli) == len(stim_start_times):
            raise ValueError("oopsies")

        # Go from stim time in number of samples to number of frames
        overview_stimuli_frames = np.round(np.multiply(np.divide(stim_start_times, sampling_rate), fps_overview))

        
        # Instert these stimuli into the table
        for i, (stimname, stim_protocol) in enumerate(stimuli.items()):
            stim_key = key.copy()
            stim_key['stimulus_uid'] = stim_key['recording_uid']+'_{}'.format(i)
            stim_key['overview_frame'] = int(overview_stimuli_frames[i])
            stim_key['duration'] = 9 # ! hardcoded
            stim_key['overview_frame_off'] =    int(overview_stimuli_frames[i]) + fps_overview*stim_key['duration'] 
            stim_key['stim_name'] = stimname
            stim_key['stim_type'] = 'audio' 
            
            try:
                table.insert1(stim_key)
            except:
                raise ValueError('Cold not insert {} into {}'.format(stim_key, table.heading))

    # ? if there are any visual stimuli, process them
    if visuals_check:
        # Check how many audio stim were inserted to make sure that "stimulus_uid" table key is correct
        n_audio_stimuli = len(stimuli)

        # Get the stimuli start and ends from the LDR AI signal
        ldr_signal = tdms_df["/'LDR_signal_AI'/'0'"].values
        ldr_stimuli = find_visual_stimuli(ldr_signal, 0.24, sampling_rate)
        
        # Get the metadata about the stimuli from the log.yml file
        log_stimuli = load_visual_stim_log(visual_log_file)

        if len(ldr_stimuli) != len(log_stimuli): 
            if len(ldr_stimuli) < len(log_stimuli):
                warnings.warn("Something weird going on, ignoring some of the stims on the visual stimuli log file")
                log_stimuli = log_stimuli.iloc[np.arange(0, len(ldr_stimuli))]
            else:
                raise ValueError("Something went wrong with stimuli detection")

        # Add the start time (in seconds) and end time of each stim to log_stimuli df
        log_stimuli['start_time'] = [s.start/sampling_rate for s in ldr_stimuli]
        log_stimuli['end_time'] = [s.end/sampling_rate for s in ldr_stimuli]
        log_stimuli['duration'] = log_stimuli['end_time'] - log_stimuli['start_time']



        # Insert the stimuli into the table, these will be used to populate the metadata table separately
        for stim_n, stim in log_stimuli.iterrows():
            stim_key = key.copy()
            stim_key['stimulus_uid'] =          stim_key['recording_uid']+'_{}'.format(stim_n + n_audio_stimuli)  # ? -> use this to collect from metadata table
            stim_key['overview_frame'] =        int(np.round(np.multiply(stim.start_time, fps_overview)))
            stim_key['duration'] =              stim.duration
            stim_key['overview_frame_off'] =    int(stim_key['overview_frame'] + fps_overview*stim_key['duration'])
            stim_key['stim_name'] =             stim.stim_name
            stim_key['stim_type'] =             'visual' 

            try:
                table.insert1(stim_key)
            except:
                raise ValueError('Cold not insert {} into {}'.format(stim_key, table.heading))

            # Keep record of the path to the log file in the part table 
            try:
                part_key = key.copy()
                part_key['filepath'] =       visual_log_file
                part_key['stimulus_uid'] =   part_key['recording_uid']+'_{}'.format(stim_n + n_audio_stimuli)  # ? -> use this to collect from metadata table
                table.VisualStimuliLogFile2.insert1(part_key)
            except:
                raise ValueError('Cold not insert {} into {}'.format(stim_key, table.VisualStimuliLogFile2.heading))



def make_visual_stimuli_metadata_table(table, key, MantisStimuli):
    stim_data = get_mantisstim_given_stimuid(key['stimulus_uid']).iloc[0]

    if stim_data.stim_type == "audio": return # this is only for visualz
    print("Populating metadata for: ", key['stimulus_uid'])

    # Load the metadata
    try:
        stim_path_data = get_mantisstim_logfilepath_given_stimud(key['stimulus_uid']).iloc[0]
    except:
        print("     couldnt fine a stimulus log file for entry")
        return

    # Get the stim calculator
    contrast_calculator = ContrastCalc(measurement_file="C:\\Users\\Federico\\Documents\\GitHub\\VisualStimuli\\Utils\\measurements.xlsx")

    if not os.path.isfile(stim_path_data.filepath): return
    metadata = load_yaml(stim_path_data.filepath)
    
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
    key['params_file']           = "v" # ? useless
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

    try:
        table.insert1(key, allow_direct_insert=True)
        print("     ... succesfully inserted: ", key['stimulus_uid'])

    except:
        raise ValueError("could not insert key: ", key)
    
    
    


#####################################################################################################################
#####################################################################################################################


def make_trackingdata_table(table, key, videofiles, ccm_table, templates, sessions, fast_mode=False):
    if key['camera_name'] != 'overview': 
        # ! threat video
        # TODO process threat vids
        # print("     WE ARE NOT PROCESSING THREAT VIDEOS HERE")
        return

    # Get the name of the experiment the video belongs to
    fetched_sessions = sessions.fetch(as_dict=True)
    session = [s for s in fetched_sessions if s['uid']==key['uid']][0]
    experiment = session['experiment_name']

    if 'lambda' in experiment.lower(): return  # ? skip this useless experiment :)

    to_include = dict(bodyparts=['snout', 'neck', 'body', 'tail_base',])

    # Check if we have all the data necessary to continue 
    vid = get_videos_given_recuid(key['recording_uid'])
    ccm = get_ccm_given_sessuid(key['uid'])
    if not np.any(vid):
        print('Could not find videofile for ', key['recording_uid']) 
        return
    elif not np.any(ccm):  
        print('Could not find common coordinate matrix for ', key['recording_uid']) 
        return
    else:
        print('\n\nProcessing tracking data for : ', key['recording_uid'])
    
    
    # Load the .h5 file with the tracking data 
    try:
        if os.path.isfile(vid['pose_filepath'][0]):
            posefile = vid['pose_filepath'][0]
        else:
            posefile = vid['pose_filepath'][0].split(".")[0]+"_pose.h5"
        posedata = pd.read_hdf(posefile)
    except:
        print('Could not load pose data:', vid['pose_filepath'][0])
        a =1 
        return

    # Insert entry into MAIN CLASS for this videofile
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
        if fast_mode and  bp != 'body': continue
        elif not fast_mode and bp not in to_include['bodyparts']: continue
        print('     ... body part: ', bp)

        # Get XY pose and correct with CCM matrix
        xy = posedata[scorer[0], bp].values[:, :2]
        corrected_data = correct_tracking_data(xy, ccm['correction_matrix'][0], ccm['top_pad'][0], ccm['side_pad'][0], experiment, session['uid'])
        corrected_data = pd.DataFrame.from_dict({'x':corrected_data[:, 0], 'y':corrected_data[:, 1]})

        # get velocity
        vel = calc_distance_between_points_in_a_vector_2d(corrected_data.values)

        # Add new vals
        corrected_data['velocity'] = vel

        # If bp is body get the position on the maze
        if 'body' in bp:
            # Get position of maze templates - and shelter
            templates_idx = [i for i, t in enumerate(templates.fetch()) if t['uid'] == key['uid']][0]
            rois = pd.DataFrame(templates.fetch()).iloc[templates_idx]

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
        try:
            table.BodyPartData.insert1(bpkey)
        except:
            pass





