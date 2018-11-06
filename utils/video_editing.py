import warnings as warn
try: import cv2
except: pass
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import crop
import os
from tempfile import mkdtemp
from tqdm import tqdm
from nptdms import TdmsFile
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import shutil

# TODO stitch videos together after tdms conversion
# TODO add video cropper to Misc

class VideoConverter:
    def __init__(self, filepath, output='.mp4', output_folder=None):
        self.editor = Editor()

        self.filep = filepath
        self.output = output

        self.folder, self.filename = os.path.split(self.filep)
        self.filename, self.extention = os.path.splitext(self.filename)

        if output_folder is not None:
            self.folder = output_folder
        self.codecs = dict(avi='png', mp4='mpeg4')

        if output in self.filep:
            warn.warn('The file is already in the desired format {}'.format(output))
        else:
            # Check format of original file and call appropriate converter
            if self.extention in ['.avi', '.mp4']: self.videotovideo_converter()
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

    def tdmstovideo_converter(self):
        def write_clip(arguments, limits=None):
            """ create a .cv2 videowriter and start writing """
            vidname, w, h, framerate, data = arguments
            vidname = '{}__{}.mp4'.format(vidname, limits[0])
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            videowriter = cv2.VideoWriter(os.path.join(self.folder, vidname), fourcc,
                                          framerate, (w, h), iscolor)

            for framen in tqdm(range(limits[0], limits[1])):
                videowriter.write(data[framen])
            videowriter.release()

        warn.warn('\nCurrently TDMS conversion depends on hardcoded variables !!')
        tempdir = os.path.join(self.folder, 'Temp')

        # HARDCODED variables about the video recorded
        skip_data_points = 4094
        real_width = 1936
        width = real_width + 48
        height = 1216
        frame_size = width * height
        real_frame_size = real_width * height
        f_size = os.path.getsize(self.filep)  # size in bytes
        tot_frames = int((f_size - skip_data_points) / frame_size)  # num frames
        fps = 100

        iscolor = False  # is the video RGB or greyscale
        print('Total number of frames {}'.format(tot_frames))

        # Number of parallel processes for faster writing to video
        num_processes = 3

        # Open TDMS
        print('Opening TDMS: ', self.filename + self.extention)
        bfile = open(self.filep, 'rb')
        print('  ...binary opened, opening mmemmapped')
        tdms = TdmsFile(bfile, memmap_dir=tempdir)  # open tdms binary file as a memmapped object

        print('Extracting data')
        tdms = tdms.__dict__['objects']["/'cam0'/'data'"].data.reshape((tot_frames, height, width), order='C')
        tdms = tdms[:, :, :real_width]  # reshape

        # Write to Video
        print('Writing to Video - {} parallel processes'.format(num_processes))
        params = (self.filename, real_width, height, fps, tdms)

        if num_processes == 1:
            write_clip(params, [0, tot_frames])
        else:
            # Get frames range for each video writer that will run in parallel
            steps = np.linspace(0, tot_frames, num_processes + 1).astype(int)
            step = steps[1]
            steps2 = np.asarray([x + step for x in steps])
            limits = [s for s in zip(steps, steps2)][:-1]
            partial_writer = partial(write_clip, params)
            # vidname, w, h, framerate, data, limits
            # start writing
            pool = ThreadPool(num_processes)
            _ = pool.map(partial_writer, limits)

        # shutil.rmtree(tempdir)

        # TODO fetch clips names and concatenate them


class Editor:
    @staticmethod
    def concatenate_clips(paths_tuple):
        clips = [VideoFileClip(p) for p in paths_tuple]
        concatenated = concatenate_videoclips(clips)
        return concatenated

    @staticmethod
    def save_clip(clip, folder, name, format, fps):
        codecs = dict(avi='png', mp4='mpeg4')
        outputname = os.path.join(folder, name + format)
        codec = codecs[format.split('.')[0]]

        print("""
            Writing {} to:
            {}
            """.format(name + format, outputname))
        clip.write_videofile(outputname, codec=codec, fps=fps)

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
            videowriter.write(frames_data[:, :, framen])
        videowriter.release()



























