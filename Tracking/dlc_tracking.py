# Standard dependencies
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import pandas as pd
import numpy as np
import pprint
import logging
import os
import yaml
import sys
from easydict import EasyDict as edict
import time
from tqdm import tqdm

from video_editing import VideoConverter

def load_yaml(fpath):
    """ load settings from a yaml file and return them as a dictionary """
    with open(fpath, 'r') as f:
        settings = yaml.load(f)
    return settings
# add parent directory: (where nnet & config are!)
tf_settings = load_yaml('.\cfg_dlc.yml')
sys.path.append(os.path.join(tf_settings['DLC folder'], "pose-tensorflow"))
sys.path.append(os.path.join(tf_settings['DLC folder'], "Generating_a_Training_Set"))

# Deep-cut dependencies
from nnet import predict
from dataset.pose_dataset import data_to_input
import default_config  # FROM DLC original scripts


class DLCmanager:
    def __init__(self, rawfolder, processedfolder=None, given_list=None):
        if processedfolder is None: processedfolder = rawfolder

        self.raw = rawfolder
        self.proc = processedfolder

        # Select files to process
        log = open(os.path.join(self.raw, "Log.txt"), "a+")

        if given_list is None:  # Process all new files in the folder
            self.analysed_files = log.read()

            # Get files to process
            self.to_process = [f for f in os.listdir(self.raw)
                               if f not in self.analysed_files and f.split('.')[-1] in ['avi', 'mp4', 'tdms']]

        else:
            self.to_process = given_list

        for video in self.to_process:
            DLCtracking(os.path.join(self.raw, video), self.proc)
            log.write(video)


class DLCtracking:
    def __init__(self, file, destfld=None, resize=False, resize_fact=2.5, batch_size=4):
        self.batch_size = batch_size

        self.filep = file
        self.folder, self.filename = os.path.split(self.filep)
        self.filename, self.extention = os.path.splitext(self.filename)

        self.resize, self.resize_f = resize, resize_fact

        if destfld is not None:
            self.destfld = destfld
        else:
            self.destfld = self.folder

        analysed = self.check_if_analysed()

        if not analysed:
            self.configure()

            if self.extention == '.tdms':  # Convert to mp4
                convert = VideoConverter(self.filep, output_folder=self.destfld)
                raise ValueError('Functionality not yet implemented: DLC on .tdms videos')
            else:
                print('Processing: ', self.filename)
                self.analyze()

    # SET UP FUNCTIONS
    def check_if_analysed(self):
        processed_data_files = [f.split('.')[0] for f in os.listdir(self.destfld) if '.h5' in f]
        if self.filename in processed_data_files:
            print("            ... video already analyzed!")
            return True
        else:
            return False

    def configure(self):
        def _merge_a_into_b(a, b):
            """Merge config dictionary a into config dictionary b, clobbering the
            options in b whenever they are also specified in a.
            """
            if type(a) is not edict:
                return

            for k, v in a.items():
                # a must specify keys that are in b
                # if k not in b:
                #    raise KeyError('{} is not a valid config key'.format(k))

                # recursively merge dicts
                if type(v) is edict:
                    try:
                        _merge_a_into_b(a[k], b[k])
                    except:
                        print('Error under config key: {}'.format(k))
                        raise
                else:
                    b[k] = v

        def cfg_from_file(filename):
            """Load a config from file filename and merge it into the default options.
            """
            cfg = default_config.cfg
            with open(filename, 'r') as f:
                yaml_cfg = edict(yaml.load(f))
            _merge_a_into_b(yaml_cfg, cfg)
            logging.info("Config:\n" + pprint.pformat(cfg))
            return cfg

        def load_config(path, filename="pose_cfg.yaml"):
            # filename = os.path.join(path, filename)
            return cfg_from_file(path)

        dlc_params = load_yaml('.\cfg_dlc.yml')
        self.cfg = load_config(dlc_params['dlc_network_posecfg'])

        basefoler = os.path.split(os.path.split(dlc_params['dlc_network_posecfg'])[0])[0]
        model_folder = os.path.join(basefoler, 'train')

        Snapshots = np.array([
            fn.split('.')[0]
            for fn in os.listdir(model_folder)
            if "index" in fn
        ])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        self.cfg['init_weights'] = os.path.join(model_folder ,
                                                Snapshots[int(dlc_params['dlc_network_snapshot'])])
        self.cfg['batch_size'] = self.batch_size
        print(dlc_params['dlc_network_posecfg'])
        print(Snapshots)

        self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.cfg)
        a =1



    # ANALYZE FUNCTIONS
    @staticmethod
    def getpose(sess, inputs, image, cfg, outputs, outall=False):
        ''' Adapted from DeeperCut, see pose-tensorflow folder'''
        image_batch = data_to_input(skimage.color.gray2rgb(image))
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        if outall:
            return scmap, locref, pose
        else:
            return pose

    def analyze(self):
        dataname = os.path.join(self.destfld, self.filename+'.h5')

        print("                 ... loading ", self.filename+self.extention)

        # Get and resize clip
        if not self.resize: factor = 1
        else:
            raise ValueError('Resize functionality not yet implemented')
            # factor = 1/self.resize_f

        try: clip = VideoFileClip(os.path.join(self.folder, self.filename+self.extention)).resize(factor)
        except:
            raise ValueError('Could not open clip at: {}'.format(os.path.join(self.folder, self.filename+self.extention)))

        # prepare data array
        frame_buffer = 10
        ny, nx = clip.size  # dimensions of frame (height, width)
        fps = clip.fps
        nframes_approx = int(np.ceil(clip.duration * clip.fps) + 10)

        nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)
        print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
              "fps!")
        print("Overall # of frames: ", nframes_approx,"with cropped frame dimensions: ", clip.size)

        start = time.time()
        clip.reader.initialize()
        print("Starting to extract posture")

        # POSE ESTIMATION <---
        if self.batch_size <2:
            PredicteData = np.zeros((nframes_approx, 3 * len(self.cfg['all_joints_names'])))
            for frame_n, frame in enumerate(clip.iter_frames()):
                if frame_n % 100 == 0:
                    print('Processed {} frames of {} total'.format(frame_n, nframes_approx))

                image = img_as_ubyte(frame)
                pose = self.getpose(self.sess, self.inputs, image, self.cfg, self.outputs, outall=False)
                PredicteData[frame_n, :] = pose.flatten()
        else:
            PredicteData = np.zeros((nframes_approx, 3 * len(self.cfg['all_joints_names'])))
            batch_ind = 0  # keeps track of which image within a batch should be written to
            batch_num = 0  # keeps track of which batch you are at
            ny, nx = clip.size  # dimensions of frame (height, width)
            frames = np.empty((self.batch_size, nx, ny, 3), dtype='ubyte')  # this keeps all frames in a batch
            for index in tqdm(range(nframes_approx)):
                image = img_as_ubyte(clip.reader.read_frame())
                if index == int(nframes_approx - frame_buffer * 2):
                    last_image = image
                elif index > int(nframes_approx - frame_buffer * 2):
                    if (image == last_image).all():
                        nframes = index
                        print("Detected frames: ", nframes)
                        if batch_ind > 0:
                            pose = predict.getposeNP(frames, self.cfg, self.sess, self.inputs, self.outputs)
                            PredicteData[batch_num * self.batch_size:batch_num * self.batch_size + batch_ind, :] = pose[:batch_ind, :]

                        break
                    else:
                        last_image = image

                frames[batch_ind] = image
                if batch_ind == self.batch_size - 1:
                    pose = predict.getposeNP(frames, self.cfg, self.sess, self.inputs, self.outputs)
                    PredicteData[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                    batch_ind += 1

        # SAVE results
        stop = time.time()
        print("Saving results...")
        pdindex = pd.MultiIndex.from_product(
            [self.cfg['all_joints_names'], ['x', 'y', 'likelihood']],
            names=['bodyparts', 'coords'])

        DataMachine = pd.DataFrame(PredicteData[:frame_n, :], columns=pdindex,
                                   index=range(frame_n))  # slice pose data to have same # as # of frames.
        DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')

        print("Processing took", round(stop-start,2), "seconds.")


if __name__ == "__main__":
    datafolder = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\video'
    processedfolder = 'D:\\Dropbox (UCL - SWC)\Dropbox (UCL - SWC)\\Rotation_vte\\processed'

    DLCmanager(datafolder, processedfolder)















