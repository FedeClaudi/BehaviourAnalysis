import sys
sys.path.append('./')  

import warnings as warn
try: import cv2
except: pass
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import crop
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

from Utilities.file_io.files_load_save import load_yaml, load_tdms_from_winstore


class VideoConverter:
    def __init__(self, filepath, output='.mp4', output_folder=None, extract_framesize=True):
        if filepath is None:
            return

        self.tdms_converter_parallel_processes = 6

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

    def __repr__(self):
        functions = [
            'def videotovideo_converter(self)', 'def opencv_mp4_to_avi_converter(videopath, savepath)', 'def tdmstovideo_converter(self)']

        print('\n\n\nThe VideoConverter class has functions:\n')
        [print('FUNCTION: ', f, '\n\n') for f in functions]
        return ''

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
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

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

    def tdmstovideo_converter(self):
        def write_clip(arguments, limits=None):
            """ create a .cv2 videowriter and start writing """
            vidname, w, h, framerate, data = arguments
            vidname = '{}__{}.mp4'.format(vidname, limits[0])
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            videowriter = cv2.VideoWriter(os.path.join(self.folder, vidname), fourcc,
                                            framerate, (w, h), iscolor)

            for framen in tqdm(range(limits[0], limits[1]+1)):
                videowriter.write(data[framen])
            videowriter.release()

        def extract_framesize_from_metadata(videotdms):
            """extract_framesize_from_metadata [takes the path to the video to be connverted and 
                uses it to extract metadata about the acquired video (frame widht and height...)]
            
            Arguments:
                videotdms {[str]} -- [path to video.tdms]
            Returns:
                frame width, height and number of frames in the video to be converted
            """
            # Find the tdms video
            paths = load_yaml('paths.yml')
            
            metadata_fld = os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder'])

            videoname = os.path.split(videotdms)[-1]
            metadata_file = [os.path.join(metadata_fld, f) for f in os.listdir(metadata_fld)
                                if videoname in f][0]

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

            tot = np.int(round(video_size/(props['width']*props['height'])))  # tot number of frames 

            if tot != props['last']:
                raise ValueError('Calculated number of frames doesnt match what is stored in the metadata: {} vs {}'.format(tot, props['last']))

            return props, tot

        ###################################################################################################################################

        start = time.time()

        if not self.extract_framesize:
            warn.warn('\nCurrently TDMS conversion depends on hardcoded variables !!')
            paddig = 0
            # ! HARDCODED variables about the video recorded
            skip_data_points = 4094
            real_width = 1936
            paddig = 48
            width = real_width + padding
            height = 1216
            frame_size = width * height
            real_frame_size = real_width * height
            f_size = os.path.getsize(self.filep)  # size in bytes
            tot_frames = int((f_size - skip_data_points) / frame_size)  # num frames
            fps = 100
            raise ValueError('This code is updated, it WILL give errors, needs to be checked first')
        else:
            props, tot_frames = extract_framesize_from_metadata(self.filep)


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
            os.mkdir(os.path.join("M:\\", 'Temp'))
        except:
            pass
        tempdir = os.path.join("M:\\", 'Temp')

        print('Opening TDMS: ', self.filename + self.extention)
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
        print('Writing to Video - {} parallel processes'.format(num_processes))
        params = (self.filename, props['width'], props['height'], props['fps'], tdms)  # To create a partial of the writer func

        if num_processes == 1:
            limits = [0, tot_frames-1]
            write_clip(params, limits)
            clip_names = ['{}__{}.mp4'.format(self.filename, limits[0])]
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
    def __repr__(self):
        functions = [
            'def concatenate_clips(paths_tuple)', 'def save_clip(clip, folder, name, format, fps)',
            'def split_clip(clip, number_of_clips=4, ispath=False)', 'def opencv_write_clip(videopath, frames_data, ...)',
            'def open_cvwriter(filepath...)', 'def compress_clip(videopath, compress_factor...)', 'def mirros_cropper()']

        print('The Editor class has functions:\n')
        [print('FUNCTION: ', f, '\n\n') for f in functions]
        return ''

    @staticmethod
    def concatenate_clips(paths_tuple):
        clips = [VideoFileClip(p) for p in paths_tuple]
        concatenated = concatenate_videoclips(clips)
        return concatenated

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
                    start_frame=None, stop_frame=None, sel_fps=None, use_moviepy=False):
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
        """

        if not use_moviepy:
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
            while True:
                cur_frame += 1
                if cur_frame % 100 == 0: print('Current frame: ', cur_frame)
                if cur_frame <= start_frame: continue
                elif cur_frame >= stop_frame: break
                else:
                    ret, frame = cap.read()
                    if not ret: break
                    else:
                        writer.write(frame)
            writer.release()
        else:
            clip = VideoFileClip(videopath)
            duration = clip.duration
            trimmed = clip.subclip(duration*start, duration*stop)
            fld, name = os.path.split(savepath)
            name, ext = name.split('.')
            self.save_clip(trimmed, fld, name, '.mp4', 120)


    @staticmethod
    def split_clip(clip, number_of_clips=4, ispath=False):
            if ispath: clip = VideoFileClip(clip)

            duration = clip.duration
            step = duration / number_of_clips
            subclips = []
            for n in range(number_of_clips):
                subclips.append(clip.subclip(step*n, step*(n+1)))

            return subclips

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

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
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
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
        except:
            raise ValueError('Could not create videowriter')
        else:
            return videowriter

    @staticmethod
    def get_video_params(cap):
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        return nframes, width, height, fps 

    def concated_tdms_to_mp4_clips(self, fld):
        """[Concatenates the clips create from the tdms video converter in the class above]
        """
        # Get list of .tdms files that might have been converted
        tdms_names = [f.split('.')[0] for f in os.listdir(fld) if 'tdms' in f]
        # Get names of converted clips
        tdms_videos_names = [f for f in os.listdir(fld) if f.split('__')[0] in tdms_names]

        # For each tdms create joined clip
        for tdmsname in tdms_names:
            # Check if a "joined" clip already exists and skip if so
            matches = sorted([v for v in tdms_videos_names if tdmsname in v])
            joined = [m for m in matches if '__joined' in m]
            if joined: 
                print(tdmsname, ' already joined')
                continue
            print('Joining clips for {} - {}'.format(tdmsname, matches))

            # Get video params and open writer
            cap = cv2.VideoCapture(os.path.join(fld, matches[0]))
            nframes, width, height, fps = self.get_video_params(cap)
            dest = os.path.join(fld, tdmsname+'__joined.mp4')

            writer = self.open_cvwriter(dest, w=width, h=height, framerate=fps, format='.mp4', iscolor=False)

            # Loop over videos and write them to joined
            all_frames_count = 0
            for vid in matches:
                print('     ... joining: ', vid)
                cap = cv2.VideoCapture(os.path.join(fld, vid))
                nframes, width, height, fps = self.get_video_params(cap)
                all_frames_count += nframes
                framecounter = 0
                while True:
                    if framecounter % 5000 == 0:
                        print('             ... frame ', framecounter)
                    framecounter += 1
                    ret, frame = cap.read()
                    if not ret: break
                    writer.write(frame)
            cap.release()
            writer.release()

            # Re open joined clip and check if total number of frames is correct
            cap = cv2.VideoCapture(dest)
            nframesjoined, width, height, fps = self.get_video_params(cap)
            if nframesjoined != all_frames_count:
                raise ValueError('Joined clip number of frames doesnt match total of individual clips: {} vs {}'.format(nframesjoined, all_frames_count))

    @staticmethod
    def compress_clip(videopath, compress_factor, save_path=None, start_frame=0, stop_frame=None):
        '''
            takes the path to a video, opens it as opecv Cap and resizes to compress factor [0-1] and saves it
        '''
        cap = cv2.VideoCapture(videopath)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)

        resized_width = int(np.ceil(width*compress_factor))
        resized_height = int(np.ceil(height*compress_factor))

        if save_path is None:
            save_name = os.path.split(videopath)[-1].split('.')[0] + '_compressed' + '.mp4'
            save_path = os.path.split(videopath)
            save_path = os.path.join(list(save_path))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        videowriter = cv2.VideoWriter(save_path, fourcc, fps, (resized_width, resized_height), False)
        framen = 0
        while True:
            if framen % 100 == 0:
                print('Processing frame ', framen)
            
            if framen >= start_frame:
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
    def manual_video_inspect(videofilepath):
        """[loads a video and lets the user select manually which frames to show]

                Arguments:
                        videofilepath {[str]} -- [path to video to be opened]

                key bindings:
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frmae
        """        
        def get_selected_frame(cap, show_frame):
                cap.set(1, show_frame)
                ret, frame = cap.read() # read the first frame
                return frame

        import cv2   # import opencv
        
        cap = cv2.VideoCapture(videofilepath)
        if not cap.isOpened():
                raise FileNotFoundError('Couldnt load the file')
        
        print(""" Instructions
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frmae
        """)

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialise showing the first frame
        show_frame = 0
        frame = get_selected_frame(cap, show_frame)

        while True:
                cv2.imshow('frame', frame)

                k = cv2.waitKey(10)

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

                try:
                        frame = get_selected_frame(cap, show_frame)
                except:
                        raise ValueError('Could not display frame ', show_frame)

if __name__ == '__main__':
    
    converter = VideoConverter(None, None)
    editor = Editor()
    print(converter, '\n\n', editor)

    ###############

    fld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
    # toconvert = [f for f in os.listdir(fld) if '.tdms' in f]
    # print(toconvert)
    # for f in toconvert:
    #     converter = VideoConverter(os.path.join(fld, f), extract_framesize=True)
        
    editor.concated_tdms_to_mp4_clips(fld)

    vid = '/Users/federicoclaudi/Desktop/181211_CA3693_1Threat__0.mp4'
    savename = '/Users/federicoclaudi/Desktop/clip2.mp4'
    editor.trim_clip(vid, savename, frame_mode=False, start=0.5, stop=0.75, use_moviepy=True)
    # editor.manual_video_inspect(savename)
























