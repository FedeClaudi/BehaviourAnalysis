import sys
sys.path.append('./')  

import warnings as warn
try: import cv2
except: pass
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from moviepy.video.fx import crop
import os
from tempfile import mkdtemp
from tqdm import tqdm
from collections import namedtuple
from nptdms import TdmsFile
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import shutil
import matplotlib.pyplot as plt
import time

from Utilities.file_io.files_load_save import *


paths_file = 'paths.yml'

class VideoConverter:
    def __init__(self, filepath, output='.mp4', output_folder=None, extract_framesize=True, n_processes=1):
        if filepath is None:
            return

        self.tdms_converter_parallel_processes = n_processes

        self.editor = Editor()
        if filepath is not None:
            if not isinstance(filepath, list):
                filepath = [filepath]

            # Loop over each file in list of paths
            for fpath in filepath:
                self.filep = fpath
                self.output = output
                self.extract_framesize = extract_framesize

                self.folder, self.filename = os.path.split(self.filep)
                self.filename, self.extention = os.path.splitext(self.filename)

                if output_folder is not None:
                    self.folder = output_folder
                self.codecs = dict(avi='png', mp4='mpeg4')

                # check if converted video exists in the target folders
                files_with_same_name = [f for f in os.listdir(self.folder) if self.filename in f and self.extention not in f]
                if files_with_same_name:
                    print('Fle {} was already converted. We found files like {} \n\n '.format(self.filename, files_with_same_name[0]))
                    continue
                if output in self.filep:
                    warn.warn('The file is already in the desired format {}'.format(output))
                else:
                    # Check format of original file and call appropriate converter
                    if self.extention in ['.mp4', '.mp4', '.mov']: self.videotovideo_converter()
                    elif self.extention == '.tdms':
                        if not self.output == '.mp4':
                            raise ValueError('TDMS --> Video conversion only supports .mp4 format for output video')
                        self.tdmstovideo_converter()
                    else:
                        raise ValueError('Unrecognised file format {}'.format(self.extention))

    def videotovideo_converter(self):
        clip = VideoFileClip(self.filep)
        fps = clip.fps
        self.editor.save_clip(clip, self.folder, self.filename, self.output, fps)

    @staticmethod
    def opencv_mp4_to_avi_converter(videopath, savepath):
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            print('Could not process this one')
            # raise ValueError('Could not load video file')

        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        videowriter = cv2.VideoWriter(savepath, fourcc, fps, (width , height ), False)
        print('Converting video: ', videopath)
        framen = 0
        while True:
            if framen % 1000 == 0: print('Frame: ', framen)
            framen += 1
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            videowriter.write(gray)
        videowriter.release()

    @staticmethod
    def extract_framesize_from_metadata(videotdms):
        """extract_framesize_from_metadata [takes the path to the video to be connverted and 
            uses it to extract metadata about the acquired video (frame widht and height...)]
        
        Arguments:
            videotdms {[str]} -- [path to video.tdms]
        Returns:
            frame width, height and number of frames in the video to be converted
        """
        print('Extracting metadata...', videotdms)
        # Find the tdms video
        paths = load_yaml(paths_file)
        
        metadata_fld = os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder'])

        videoname = os.path.split(videotdms)[-1]
        try:
            metadata_file = [os.path.join(metadata_fld, f) for f in os.listdir(metadata_fld)
                                if videoname in f][0]
        except:
            print("Could not load metadata for {} in folder {}".format(videoname, metadata_fld))
            raise ValueError("Could not load metadata for {} in folder {}".format(videoname, metadata_fld))
        # Get info about the video data
        # video = TdmsFile(videotdms)
        # video_bytes = video.object('cam0', 'data').data
        video_size = os.path.getsize(videotdms)
        #plt.plot(video_bytes[:10000])
        #plt.show()

        # Get info about the metadata
        metadata = TdmsFile(metadata_file)

        # Get values to return
        metadata_object = metadata.object()
        props = {n:v for n,v in metadata_object.properties.items()} # fps, width, ...  code below is to print props out

        # for name, value in metadata_object.properties.items():
        #     print("{0}: {1}".format(name, value))
        # return
        if props['width'] > 0:
            tot = np.int(round(video_size/(props['width']*props['height'])))  # tot number of frames 
            if tot != props['last']:
                raise ValueError('Calculated number of frames doesnt match what is stored in the metadata: {} vs {}'.format(tot, props['last']))
        else:
            tot = 0

        return props, tot

    def tdmstovideo_converter(self):
        def write_clip(arguments, limits=None, clean_name=False):
            """ create a .cv2 videowriter and start writing """
            vidname, w, h, framerate, data = arguments
            if clean_name:
                vidname = '{}.mp4'.format(vidname)
            else:
                vidname = '{}__{}.mp4'.format(vidname, limits[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videowriter = cv2.VideoWriter(os.path.join(self.folder, vidname), fourcc,
                                            framerate, (w, h), iscolor)

            for framen in tqdm(range(limits[0], limits[1]+1)):
                videowriter.write(data[framen])
            videowriter.release()

        ###################################################################################################################################

        start = time.time()
        
        print("Ready to convert: ", self.filep)

        props, tot_frames = self.extract_framesize_from_metadata(self.filep)

        print("Got metadata\n")
        # ? set up options
        # Number of parallel processes for faster writing to video
        num_processes = self.tdms_converter_parallel_processes
        iscolor = False  # is the video RGB or greyscale
        print('Preparing to convert video, saving .mp4 at {}fps using {} parallel processes'.format(props['fps'], num_processes))
        print('Total number of frames {}'.format(tot_frames))

        # * Get file from winstore
        # temp_file = load_tdms_from_winstore(self.filep)
        # tempdir = os.path.split(temp_file)[0]

        # ? alternatively just create a temp folder where to store the memmapped tdms
        # Open video TDMS 
        try:    # Make a temporary directory where to store memmapped tdms
            os.mkdir(os.path.join("D:\\", 'Temp'))
        except:
            pass
        tempdir = os.path.join("D:\\", 'Temp')

        print('Opening TDMS: ', self.filename + self.extention)
        print("Opening binary at: ", self.filep)
        bfile = open(self.filep, 'rb')

        print('  ...binary opened, opening mmemmapped')
        openstart = time.time()
        tdms = TdmsFile(bfile, memmap_dir=tempdir)  # open tdms binary file as a memmapped object
        openingend = time.time()

        print('     ... memmap opening took: ', np.round(openingend-openstart, 2))
        print('Extracting data')
        tdms = tdms.__dict__['objects']["/'cam0'/'data'"].data.reshape((tot_frames, props['height'], props['width']), order='C')
        tdms = tdms[:, :, :(props['width']+props['padding'])]  # reshape

        # Write to Video
        print('Writing to Video {} - {} parallel processes'.format(self.filename, num_processes))
        params = (self.filename, props['width'], props['height'], props['fps'], tdms)  # To create a partial of the writer func

        if num_processes == 1:
            limits = [0, tot_frames-1]
            write_clip(params, limits, clean_name=True)
            clip_names = ['{}.mp4'.format(self.filename)]
        else:
            # Get frames range for each video writer that will run in parallel
            # vid 1 will do A->B, vid2 B+1->C ...
            frame_numbers = [i for i in range(int(tot_frames))]
            splitted = np.array_split(frame_numbers, num_processes)
            limits = [(int(x[0]), int(x[-1])) for x in splitted]
            print(limits)
            
            # Create partial function
            partial_writer = partial(write_clip, params)

            # start writing
            pool = ThreadPool(num_processes)
            _ = pool.map(partial_writer, limits)
            clip_names = ['{}__{}.mp4'.format(self.filename, lim[0]) for lim in limits]

        # Check if all the frames have been converted
        readers = {}
        print('\n\n\nSaved clips: ', clip_names)
        frames_counter = 0
        for clip in clip_names:  
            # Open each clip and get number of frames
            cap = cv2.VideoCapture(os.path.join(self.folder, clip))
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_counter += nframes
            print(clip, ' has frames: ', nframes)
        print('Converted clips have {} frames, original clip had: {}'.format(frames_counter, tot_frames))
        if not tot_frames == frames_counter:
            raise ValueError('Number of frames in converted clip doesnt match that of original clip')

        # Remove temp file
        ##os.remove(temp_file)

        # fin
        end = time.time()
        print('Video writing and extra handling tooK : ', np.round(end-openingend, 2))
        print('Converted {} frames in {}s\n\n'.format(tot_frames, round(end-start)))

class Editor:
    def __init__(self):
        self.paths = load_yaml('./paths.yml')

    @staticmethod
    def save_clip(clip, folder, name, format, fps):
        codecs = dict(avi='png', mp4='mpeg4')
        outputname = os.path.join(folder, name + format)
        codec = codecs[format.split('.')[1]]

        print("""
            Writing {} to:
            {}
            """.format(name + format, outputname))
        clip.write_videofile(outputname, codec=codec, fps=fps)

    def trim_clip(self, videopath, savepath, frame_mode=False, start=0.0, stop=0.0,
                    start_frame=None, stop_frame=None, sel_fps=None, lighten=False):
        """trim_clip [take a videopath, open it and save a trimmed version between start and stop. Either 
        looking at a proportion of video (e.g. second half) or at start and stop frames]
        
        Arguments:
            videopath {[str]} -- [video to process]
            savepath {[str]} -- [where to save]
        
        Keyword Arguments:
            frame_mode {bool} -- [define start and stop time as frame numbers] (default: {False})
            start {float} -- [video proportion to start at ] (default: {0.0})
            end {float} -- [video proportion to stop at ] (default: {0.0})
            start_frame {[type]} -- [video frame to stat at ] (default: {None})
            end_frame {[type]} -- [videoframe to stop at ] (default: {None})
            selfpd {[int]}(default, None) -- [specify the fps of the output]
            lighten --> make the video a bit brighter
        """

        # Open reader and writer
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps  = self.get_video_params(cap)
    
        if sel_fps is not None:
            fps = sel_fps
        writer = self.open_cvwriter(savepath, w=width, h=height, framerate=int(fps), format='.mp4', iscolor=False)

        # if in proportion mode get start and stop mode
        if not frame_mode:
            start_frame = int(round(nframes*start))
            stop_frame = int(round(nframes*stop))
        
        # Loop over frames and save the ones that matter
        print('Processing: ', videopath)
        cur_frame = 0
        cap.set(1,start_frame)
        while True:
            cur_frame += 1
            if cur_frame % 100 == 0: print('Current frame: ', cur_frame)
            if cur_frame <= start_frame: continue
            elif cur_frame >= stop_frame: break
            else:
                
                ret, frame = cap.read()
                if not ret: break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if lighten:
                        a = 1
                    writer.write(frame)
        writer.release()


    def split_clip(self, clip, number_of_clips=4, dest_fld=None):
        """[Takes a video and splits into clips of equal length]
        
        Arguments:
            clip {[str]} -- [path to video to be split]
        
        Keyword Arguments:
            number_of_clips {int} -- [number of subclips] (default: {4})
            dest_fld {[srt]} -- [path to folder where clips will be saved. If None, clips will be saved in same folder as original clip] (default: {None})
        """

        fld, name = os.path.split(clip)
        name, ext = name.split('.')
        if dest_fld is None: dest_fld = fld

        cap = cv2.VideoCapture(clip)
        nframes, width, height, fps  = self.get_video_params(cap)

        frames_array = np.linspace(0, nframes, nframes+1)
        clips_frames = np.array_split(frames_array, number_of_clips)

        for i, clip in enumerate(clips_frames):
            
            start, end = clip[0], clip[-1]
            print('Clip {} of {}, frame range: {}-{}'.format(i, number_of_clips, start, end))
            if i == 0: 
                print(' ... skipping the first clip')
                continue
            cap.set(1,start)
            
            savename = os.path.join(dest_fld, name+'_clip{}.'.format(i)+ext)
            writer = self.open_cvwriter(savename, w=width, h=height, framerate=fps, iscolor=False)

            counter = start
            while counter <= end:
                counter += 1
                ret, frame = cap.read()
                if not ret: 
                    writer.release()
                    return
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(gray)
            writer.release()
        cap.release()

    def tile_clips(self, clips_l, savepath):
        """[Tiles multiple videos horizzontally. It assumes that all videos have the same width and height]
        
        Arguments:
            clips_l {[list]} -- [list of paths to the videos to be tiled]
            savepath {[type]} -- [complete filepath of the video to be saved]
        """
        caps = [ cv2.VideoCapture(videofilepath) for videofilepath in clips_l]

        nframes, width, height, fps = self.get_video_params(caps[0])
        width *= len(caps)
        writer = self.open_cvwriter(savepath, w=width, h=height, framerate=fps, iscolor=True)

        while True:
            try:
                frames = [cap.read()[1] for cap in caps]
            except:
                break
            else:
                tot_frame = np.hstack(frames)
                writer.write(tot_frame)
        writer.release()

    @staticmethod
    def opencv_write_clip(videopath, frames_data, w=None, h=None, framerate=None, start=None, stop=None,
                            format='.mp4', iscolor=False):
        """ create a .cv2 videowriter and  write clip to file """
        if format != '.mp4':
            raise ValueError('Fileformat not yet supported by this function: {}'.format(format))

        if start is None: start = 0
        if stop is None: stop = frames_data.shape[-1]
        start, stop = int(start), int(stop)
        if w is None: w = frames_data.shape[0]
        if h is None: h = frames_data.shape[1]
        if framerate is None: raise ValueError('No frame rate parameter was given as an input')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(videopath, fourcc, framerate, (w, h), iscolor)

        for framen in tqdm(range(start, stop)):
            frame = np.array(frames_data[:, :, framen], dtype=np.uint8).T
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            videowriter.write(frame)
        videowriter.release()

    @staticmethod
    def open_cvwriter(filepath, w=None, h=None, framerate=None, format='.mp4', iscolor=False):
        try:
            if 'avi' in format:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # (*'MP4V')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
        except:
            raise ValueError('Could not create videowriter')
        else:
            return videowriter

    @staticmethod
    def get_video_params(cap):
        if isinstance(cap, str):
            cap = cv2.VideoCapture(cap)
            
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        return nframes, width, height, fps 

    def concated_tdms_to_mp4_clips(self, fld):
        """[Concatenates the clips create from the tdms video converter in the class above]
        """
        
        def joiner(arguments):
            clipname, matches, writer = arguments
            print('Joiner working on : ', clipname)
            # Loop over videos and write them to joined
            all_frames_count = 0
            for i, vid in enumerate(matches):
                print('     ... joining: ', vid, ' {} of {}'.format(i, len(matches)-1))
                cap = cv2.VideoCapture(os.path.join(fld, vid))
                nframes, width, height, fps = self.get_video_params(cap)
                all_frames_count += nframes
                framecounter = 0
                while True:
                    framecounter += 1
                    ret, frame = cap.read()
                    if not ret: break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    writer.write(gray)
            cap.release()
            writer.release()

            # Re open joined clip and check if total number of frames is correct
            cap = cv2.VideoCapture(dest)
            nframesjoined, width, height, fps = self.get_video_params(cap)
            if nframesjoined != all_frames_count:
                raise ValueError('{} - Joined clip number of frames doesnt match total of individual clips: {} vs {}'.format( clipname, nframesjoined, all_frames_count))
            else:
                print('Clip ', clipname, ' was saved succesfully')

        # Get list of .tdms files that might have been converted
        tdms_names = [f.split('.')[0] for f in os.listdir(fld) if 'tdms' in f]
        # Get names of converted clips
        tdms_videos_names = [f for f in os.listdir(fld) if f.split('__')[0] in tdms_names]

        print('Collecting data on videos to join ')
        writers_store = {}
        # For each tdms create joined clip
        for tdmsname in tdms_names:
            # Check if a "joined" clip already exists and skip if so
            matches = sorted([v for v in tdms_videos_names if tdmsname in v])
            if not matches: continue
            joined = [m for m in matches if '__joined' in m]
            if joined: 
                joined = os.path.join(fld, joined[0])
                jsize = os.path.getsize(joined)
                if jsize > 258:
                    print(tdmsname, ' already joined', jsize, 'bytes')
                    continue
                else:
                    matches = [m for m in matches if '__joined' not in m]
            
            # Sort matches
            sortidx = np.argsort(np.array(([int(n.split('__')[-1].split('.')[0]) for n in matches])))
            mathches = [np.array(matches)[sortidx]]

            # Get video params and open writer
            cap = cv2.VideoCapture(os.path.join(fld, matches[0]))
            nframes, width, height, fps = self.get_video_params(cap)
            dest = os.path.join(fld, tdmsname+'__joined.mp4')
            writer = self.open_cvwriter(dest, w=width, h=height, framerate=int(fps), format='.mp4', iscolor=False)

            # Add to writers store
            writers_store[tdmsname] = (dest, matches, writer)

        # Write in parallel
        num_processes = 1  
        if num_processes>1:
            print('Ready to joint {} videos in parallel'.format(num_processes))
            pool = ThreadPool(num_processes)
            args_to_write = []
            for i in range(num_processes):
                key = list(writers_store.keys())[i]
                args = writers_store[key]
                args_to_write.append(args)

            print('Writing...')
            [print(a[0]) for a in args_to_write]

            _ = pool.map(joiner, args_to_write)
        else:
            for args in writers_store.values():
                try:
                    joiner(args)
                except:
                    print('Joining Failed... removing incopmlete file: ', args[0])
                    os.remove(args[0])

    def compress_clip(self, videopath, compress_factor, save_path=None, start_frame=0, stop_frame=None):
        '''
            takes the path to a video, opens it as opecv Cap and resizes to compress factor [0-1] and saves it
        '''
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        resized_width = int(np.ceil(width*compress_factor))
        resized_height = int(np.ceil(height*compress_factor))

        if save_path is None:
            save_name = os.path.split(videopath)[-1].split('.')[0] + '_compressed' + '.mp4'
            save_path = os.path.split(videopath)
            save_path = os.path.join(list(save_path))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = self.open_cvwriter(save_path, w=resized_width, h=resized_height, framerate=fps, format='.mp4', iscolor=False)
        framen = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while True:       
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (resized_width, resized_height)) 
            videowriter.write(resized)
            framen += 1

            if stop_frame is not None:
                if framen >= stop_frame: break

        videowriter.release()
    
    def mirros_cropper(self, v, fld):
        """mirros_cropper [takes a video and saves 3 cropped versions of it with different views]
        
        Arguments:
            v {[str]} -- [path to video file]
            fld {[str]} -- [path to destination folder]
        """

        # cropping params
        crop = namedtuple('coords', 'x0 x1 y0 y1')
        main = crop(320, 1125, 250, 550)
        side = crop(1445, 200, 250, 550)
        top = crop(675, 800, 75, 200)

        # get names of dest files
        main_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_catwalk')))
        side_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_side')))
        top_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_top')))

        # Check if files already exists
        finfld = os.listdir(fld)
        names = [os.path.split(main_name)[-1], os.path.split(side_name)[-1], os.path.split(top_name)[-1]]
        matched = [n for n in names if n in finfld]
        if not matched:
            # Open opencv reader and writers
            cap = cv2.VideoCapture(v)
            main_writer = self.open_cvwriter(filepath=main_name, w=main.x1, h=main.y1, framerate=30)
            side_writer = self.open_cvwriter(filepath=side_name, h=side.x1, w=side.y1, framerate=30)
            top_writer = self.open_cvwriter(filepath=top_name, w=top.x1, h=top.y1, framerate=30)
            writers = [main_writer, side_writer, top_writer]
        elif len(matched) != len(names):
            raise FileNotFoundError('Found these videos in destination folder which would be overwritten: ', matched)
        else:
            # all videos already exist in destination folder, no need to do anything
            print('  cropped videos alrady exist in destination folder: ', names)
            return main_name, side_name, top_name

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            main_frame = frame[main.y0:main.y0+main.y1, main.x0:main.x0+main.x1]
            side_frame = frame[side.y0:side.y0+side.y1, side.x0:side.x0+side.x1]

            print(side_frame.shape)
            side_frame = np.rot90(side_frame, 1)
            print(side_frame.shape)

            top_frame = frame[top.y0:top.y0+top.y1, top.x0:top.x0+top.x1]

            cv2.imshow('main', main_frame)
            cv2.imshow('side', side_frame)
            cv2.imshow('top', top_frame)

            main_writer.write(main_frame)
            side_writer.write(side_frame)
            top_writer.write(top_frame)

            cv2.waitKey(1)
        
        for wr in writers:
            wr.release()
        
        return main_name, side_name, top_name

    @staticmethod
    def play_video(videofilepath, faster=False, play_from=None,  stop_after=None):
        import cv2
        cap = cv2.VideoCapture(videofilepath)

        if play_from is not None:
            cap.set(1, play_from)

        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow('frame', frame)
            if faster: wait = 5
            else: wait = 15
            k = cv2.waitKey(wait)
            if k == ord('q'): break

            frame_counter += 1
            if stop_after is not None:
                if frame_counter >= stop_after: break

    @staticmethod
    def manual_video_inspect(videofilepath):
        """[loads a video and lets the user select manually which frames to show]

                Arguments:
                        videofilepath {[str]} -- [path to video to be opened]

                key bindings:
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frame
                        - f: save frame
        """        
        def get_selected_frame(cap, show_frame):
                cap.set(1, show_frame)
                ret, frame = cap.read() # read the first frame
                return frame

        import cv2   # import opencv
        
        # Open text file to save selected frames
        fold, name = os.path.split(videofilepath)

        frames_file = open(os.path.join(fold, name.split('.')[0])+".txt","w+")


        cap = cv2.VideoCapture(videofilepath)
        if not cap.isOpened():
                raise FileNotFoundError('Couldnt load the file')
        
        print(""" Instructions
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frame
                        - f: save frame number
                        - q: quit
        """)

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialise showing the first frame
        show_frame = 0
        frame = get_selected_frame(cap, show_frame)

        while True:
                cv2.imshow('frame', frame)

                k = cv2.waitKey(25)

                if k == ord('d'):
                        # Display next frame
                        if show_frame < number_of_frames:
                                show_frame += 1
                elif k == ord('a'):
                        # Display the previous frame
                        if show_frame > 1:
                                show_frame -= 1
                elif k ==ord('s'):
                        selected_frame = int(input('Enter frame number: '))
                        if selected_frame > number_of_frames or selected_frame < 0:
                                print(selected_frame, ' is an invalid option')
                        show_frame = int(selected_frame)
                elif k == ord('f'): 
                    print('Saving frame to text')
                    frames_file.write('\n'+str(show_frame))
                elif k == ord('q'):
                    frames_file.close()
                    sys.exit()

                try:
                        frame = get_selected_frame(cap, show_frame)
                        print('Showing frame {} of {}'.format(show_frame, number_of_frames))
                except:
                        raise ValueError('Could not display frame ', show_frame)

    def crop_video(self, videopath, x, y):
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        path, name = os.path.split(videopath)
        name, ext = name.split(".")
        savename = os.path.join(path, name +"_cropped.mp4")

        writer = self.open_cvwriter(savename, w=x, h=y, framerate=fps, format='.mp4', iscolor=True)


        while True:
            ret, frame = cap.read()
            if not ret: break

            cropped = frame[:y, :x, :]

            # cv2.imshow("frame", cropped)
            # cv2.waitKey(1)

            writer.write(cropped)
        writer.release()

    def concatenate_videos(self, videos):
        """[takes a list of paths as argument]
        """
        tot_frames = 0
        for video in videos:
            cap = cv2.VideoCapture(video)
            nframes, width, height, fps = self.get_video_params(cap)
            tot_frames += nframes

        path, name = os.path.split(video)
        name, ext = name.split(".")
        savename = os.path.join(path, name +"_concatenated.mp4")

        writer = self.open_cvwriter(savename, w=width, h=height, framerate=fps, format='.mp4', iscolor=True)

        for video in videos:
            cap = cv2.VideoCapture(video)
            while True:
                ret, frame = cap.read()
                if not ret: break

                writer.write(frame)
        writer.release()

    def brighten_video(self, videopath, save_path, add_value=100):
        cap = cv2.VideoCapture(videopath)
        nframes, width, height, fps = self.get_video_params(cap)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = self.open_cvwriter(save_path, w=width, h=height, framerate=fps, format='.mp4', iscolor=False)

        while True:       
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = np.add(gray, add_value)
            gray[gray>255] + 255
            videowriter.write(gray)
        videowriter.release()

    @staticmethod
    def get_selected_frame(cap, show_frame):
            cap.set(1, show_frame)
            ret, frame = cap.read() # read the first frame
            
            if not ret: return None
            else: return frame

if __name__ == '__main__':
    # ! Preapre Threat Clips for DLC training
    editor = Editor()
    fld = "Z:\\branco\\Federico\\raw_behaviour\\maze\\_threat_training_clips\\originals"
    fld2 = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\_threat_training_clips_cut"

    # ? trim em and move em
    # for v in os.listdir(fld):
    #     editor.trim_clip(os.path.join(fld,v), os.path.join(fld2, v), frame_mode=True, start_frame=0, stop_frame=450)

    # ? squeeze em
    # for v in tqdm(os.listdir(fld2)):
    #     vv = v.split(".")[0]+"_compressed.mp4"
    #     editor.compress_clip(os.path.join(fld2, v), .5, save_path=os.path.join(fld2, vv))

    # ? add some light to em
        # ? squeeze em
    # for v in tqdm(os.listdir(fld2)):
        # vv = v.split(".")[0]+"_light.mp4"
        # editor.brighten_video(os.path.join(fld2, v), os.path.join(fld2, vv), 25)



    Editor().manual_video_inspect(r"C:\Users\Federico\Desktop\test\TOSHDNX120_A_011.mov")

    # Editor().compress_clip(r"O:\M_1L.avi", .6, save_path=r"O:\test.mp4")