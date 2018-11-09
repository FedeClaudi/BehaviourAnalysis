"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

def extract_frames(config,mode,algo='uniform',crop=False,checkcropping=False):
    """
    Extracts frames from the videos in the config.yaml file. Only the videos in the config.yaml will be used to select the frames.\n
    Use the function ``add_new_video`` at any stage of the project to add new videos to the config file and extract their frames.
    
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.
        
    mode : string
        String containing the mode of extraction. It must be either ``automatic`` or ``manual``.
        
    algo : string 
        String specifying the algorithm to use for selecting the frames. Currently, deeplabcut supports either ``kmeans`` or ``uniform`` based selection. This flag is
        only required for ``automatic`` mode and the default is ``uniform``.
        
    crop : bool, optional
        If this is set to True, the selected frames are cropped based on the ``crop`` parameters in the config.yaml file. 
        The default is ``False``; if provided it must be either ``True`` or ``False``.
        
    checkcropping: bool, optional
        If this is set to True, the cropping parameters are overlayed in a plot of the first frame to check and the user can decide if the program should proceed 
        with those parameters, or perhaps edit them. The default is ``False``; if provided it must be either ``True`` or ``False``.
        
    Examples
    --------
    for selecting frames automatically with 'kmeans' and want to crop the frames based on the ``crop`` parameters in config.yaml
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic','kmeans',True)
    --------
    for selecting frames automatically with 'uniform' and want to crop the frames based on the ``crop`` parameters in config.yaml
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic',crop=True)
    --------
    for selecting frames automatically with 'uniform', want to crop the frames based on the ``crop`` parameters in config.yaml and check for cropping
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','automatic',crop=True,checkcropping=True)
    --------
    for selecting frames manually,
    >>> deeplabcut.extract_frames('/analysis/project/reaching-task/config.yaml','manual')
    While selecting the frames manually, you do not need to specify the ``crop`` parameter in the command. Rather, you will get a prompt in the graphic user interface to choose 
    if you need to crop or not.
    --------
    
    """
    import os
    import sys
    import yaml
    import numpy as np
    from moviepy.editor import VideoFileClip
    from pathlib import Path
    from skimage import io
    from skimage.util import img_as_ubyte
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    try:
        import cv2
        use_opencv = True
    except:
        use_opencv = False
    
    from deeplabcut.generate_training_dataset import frameselectiontools

    if mode == "manual":
        wd = Path(config).resolve().parents[0]
        os.chdir(str(wd))
        from deeplabcut.generate_training_dataset import frame_extraction_toolbox 
        frame_extraction_toolbox.show(config)
        
    elif mode == "automatic":
        config_file = Path(config).resolve()
        with open(str(config_file), 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        print("Reading config file successfully...")
        
        numframes2pick = cfg['numframes2pick']
        start = cfg['start']
        stop = cfg['stop']
        
        # Check for variable correctness
        if start>1 or stop>1 or start<0 or stop<0 or start>=stop:
            raise Exception("Erroneous start or stop values. Please correct it in the config file.")
        if numframes2pick<1 and not int(numframes2pick):
            raise Exception("Perhaps consider extracting more, or a natural number of frames.")
            
        videos = cfg['video_sets'].keys()
        for vindex,video in enumerate(videos):
            
            if not use_opencv: # use moviepy
                clip = VideoFileClip(video)
                indexlength = int(np.ceil(np.log10(clip.duration * clip.fps)))
            else:  # updated frame extraction with openCV
                cap = cv2.VideoCapture(video)
                if not cap.isOpened():
                    raise ValueError('Could not load video file for frame extraction')
                fps = cap.get(cv2.CAP_PROP_FPS)
                indexlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # tot number of frames in videofile

            if crop==True:
                print("Make sure you change the crop parameters in the config.yaml file. The default parameters are set to the video dimensions.")
                coords = cfg['video_sets'][video]['crop'].split(',')

                if not use_opencv:
                    image = clip.get_frame(start*clip.duration) #frame is accessed by index *1./clip.fps (fps cancels)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.floor(indexlength*start)))
                    ret, image = cap.read()
                    if not ret:
                        raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                         format(int(np.floor(indexlength*start)), indexlength))

                fname = Path(video)
                output_path = Path(config).parents[0] / 'labeled-data' / fname.stem
                if output_path.exists() and checkcropping==True:
                    fig,ax = plt.subplots(1)
                    # Display the image
                    ax.imshow(image)
                    # Create a Rectangle patch
                    rect = patches.Rectangle((int(coords[0]),int(coords[2])),int(coords[1]),int(coords[3]),linewidth=3,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    plt.show()
                    
                    print("The red boundary indicates how the cropped image will look.")
                    # saveimg = str(Path(config).parents[0] / Path('labeled-data','IsCroppingOK_'+fname.stem +".png"))
                    # io.imsave(saveimg, image)
                    
                    msg = input("Is the cropping ok? (yes/no): ")
                    if msg == "yes" or msg == "y" or msg =="Yes" or msg == "Y":
                      if len(os.listdir(str(output_path)))==0: #check if empty
                            # store full frame (good for augmentation)
                            index = 0

                            if not use_opencv: image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
                            else:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                                ret, image = cap.read()
                                image = img_as_ubyte(image)
                            output_path = Path(config).parents[0] / 'labeled-data' / Path(video).stem
                            saveimg = str(output_path) +'/img'+ str(index).zfill(indexlength) + ".png"  
                            io.imsave(saveimg, image)
                            
                            # crop and move on with extraction of frames:
                            if not use_opencv:   # if using opencv the individual frames are cropped during extraction
                                clip=clip.crop(y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]))

                      else:
                          askuser=input ("The directory already contains some frames. Do you want to add to it?(yes/no): ")
                          if askuser=='y' or askuser=='yes' or askuser=='Y' or askuser=='Yes':
                              if not use_opencv: # if using opencv the individual frames are cropped during extraction
                                  clip=clip.crop(y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]))
                          else:
                              sys.exit("Delete the frames and try again later!")
                    else:
                      sys.exit("Correct the crop parameters in the config.yaml file and try again!")

            print("Extracting frames based on %s ..." %algo)
            if use_opencv:
                clip = VideoFileClip(video)  # need to open the clip to pass it to the frame selection tools
            if algo =='uniform': #extract n-1 frames (0 was already stored)
                frames2pick = frameselectiontools.UniformFrames(clip,numframes2pick-1,start,stop)
            elif algo =='kmeans':
                frames2pick = frameselectiontools.KmeansbasedFrameselection(clip,numframes2pick-1,start,stop)
            else:
                print("Please implement this method yourself and send us a pull request! Otherwise, choose 'uniform' or 'kmeans'.")
                frames2pick=[]

            if use_opencv: indexlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else: indexlength = int(np.ceil(np.log10(clip.duration * clip.fps)))
            for index in frames2pick:
                output_path = str(Path(config).parents[0] / 'labeled-data' / Path(video).stem)

                try:
                    # using opencv
                    if use_opencv:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ret, image = cap.read()
                        if crop:
                            image = image[int(coords[2]): int(coords[3]), int(coords[0]):int(coords[1])]

                        image = img_as_ubyte(image)  # this will fail if frame is not loaded correctly
                        img_name = str(output_path) + '/img' + str(index) + ".png"
                    else:
                        image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
                        img_name = str(output_path) + '/img' + str(index).zfill(indexlength) + ".png"
                    io.imsave(img_name,image)
                except FileNotFoundError:
                    print("Frame # ", index, " does not exist.")
            #close video. 
            if not use_opencv:
                clip.close()
                del clip
            else:
                del cap
    else:
        print("Invalid MODE. Choose either 'manual' or 'automatic'. Check ``help(deeplabcut.extract_frames)`` on python and ``deeplabcut.extract_frames?`` \
              for ipython/jupyter notebook for more details.")
    
    print("\nFrames are selected.\nYou can now label the frames using the function 'label_frames'.")

