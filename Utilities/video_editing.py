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

# TODO stitch videos together after tdms conversion
# TODO add video cropper to Misc

class VideoConverter:
    def __init__(self, filepath, output='.mp4', output_folder=None, extract_framesize=True):

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

            for framen in tqdm(range(limits[0], limits[1])):
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

            pth = os.path.split(videotdms)[0]
            metadata_name = videotdms.split('.')[0] + 'meta.tdms'
            
            metadata = TdmsFile(os.path.join(pth, metadata_name))
            print(metadata)

        if not self.extract_framesize:
            warn.warn('\nCurrently TDMS conversion depends on hardcoded variables !!')
    
            # ! HARDCODED variables about the video recorded
            skip_data_points = 4094
            real_width = 1936
            width = real_width + 48
            height = 1216
            frame_size = width * height
            real_frame_size = real_width * height
            f_size = os.path.getsize(self.filep)  # size in bytes
            tot_frames = int((f_size - skip_data_points) / frame_size)  # num frames
        else:
            extract_framesize_from_metadata(self.filep)

        iscolor = False  # is the video RGB or greyscale
        print('Total number of frames {}'.format(tot_frames))

        # Number of parallel processes for faster writing to video
        tempdir = os.path.join(self.folder, 'Temp')
        fps = 100
        num_processes = 3

        print('Preparing to convert video, saving .mp4 at {}fps using {} parallel processes'.format(fps, num_processes))

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
        codec = codecs[format.split('.')[1]]

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
            frame = np.array(frames_data[:, :, framen], dtype=np.uint8).T
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            videowriter.write(frame)
        videowriter.release()

    def open_cvwriter(filepath, w=None, h=None, framerate=None, format='.mp4', iscolor=False):
        if format != '.mp4':
            raise ValueError('Fileformat not yet supported by this function: {}'.format(format))
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
        except:
            raise ValueError('Could not create videowriter')
        else:
            return videowriter

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

    def mirros_cropper(self):
        fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\3dcam_test'


        # cropping params
        crop = namedtuple('coords', 'x0 x1 y0 y1')
        main = crop(320, 1125, 250, 550)
        side = crop(1445, 200, 250, 550)
        top = crop(675, 800, 75, 200)

        edit = sekf

        videos = sorted(os.listdir(fld))
        for v in videos:
            if '.mp4' in v: continue
            orig = os.path.join(fld, v)
            cap = cv2.VideoCapture(orig)

            main_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_main')))
            main_writer = edit.open_cvwriter(filepath=main_name, w=main.x1, h=main.y1, framerate=30)
            side_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_side')))
            side_writer = edit.open_cvwriter(filepath=side_name, h=side.x1, w=side.y1, framerate=30)
            top_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_top')))
            top_writer = edit.open_cvwriter(filepath=top_name, w=top.x1, h=top.y1, framerate=30)

            writers = [main_writer, side_writer, top_writer]

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


if __name__ == '__main__':
    origin = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\video'
    destination = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\video_mp4'

    converter = VideoConverter(None)
    
    for v in os.listdir(origin):
        if not '.mp4' in v: continue
        
        ori = os.path.join(origin, v)
        new_name = v.split('.')[0]+'.mp4'
        dest =  os.path.join(destination, new_name)

        if not new_name in os.listdir(destination):
            converter.opencv_mp4_to_avi_converter(ori, dest)

        



























