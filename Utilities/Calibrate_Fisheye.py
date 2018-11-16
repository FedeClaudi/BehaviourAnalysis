import numpy as np; import os; import glob; import cv2

# TODO give credit to philip

class FisheyeCorrection:
    ''' 
        Collections of functions to correct fish-eye distorsions.
        Two steps:
            1) Identify distorsion matrices by taking several pictures of checkerboards with the camera being used
                -- optional: test calibration
            2) Use these matrices to correct images and videos


        Usage:
            for 1) call: self.get_calibration_matrices()
    '''

    def __init__(self, images_fld='', cameraname='', load_maps=False, maps_file=''):
        '''__init__ [initialises variables and loads maps]
        
        Keyword Arguments:
            images_fld {str} -- [path to folder containing images to use for calibration] (default: {''})
            cameraname {str} -- [name of the camera being calibrated] (default: {''})
            load_maps {bool} -- [load previously saved maps] (default: {False})
            maps_file {str} -- [path to maps file] (default: {''})
        
        Raises:
            ValueError -- [description]
        '''

        if load_maps:
            if not maps_file or not isinstance(maps_file, str) or not '.npy' in maps_file:
                raise ValueError('Plese provide valid path to maps file')
            self.load_maps(maps_file)
        else:
            if not images_fld or not cameraname or not isinstance(images_fld, str) or not isinstance(cameraname, str):
                raise ValueError('Either images fld or cameraname parameters not passed correctly')
            self.images_fld = images_fld  # folder storing checkerboard images to use for calibration

            # size of the checkerboard (# of vertices in each dimension, not including those on the edge)
            self.CHECKERBOARD = (28, 12)

            self.cameraname = cameraname  # name prepended to the saved rectification maps

            self.light_thresholds = np.flip(np.arange(30, 160, 14), axis=0)
            self.dark_threshold = 30

            self.maps = None

    def find_checkerboard(self, display=True):
        '''find_checkerboard [goes through calibration images and detectes checkerboard in them, returns identified points]
        
        Keyword Arguments:
            display {bool} -- [display images as computation takes over yes/no] (default: {True})
        
        Returns:
            last calibration image used
            object points - ?
            image points - ?
        '''

        # Find checkerboard corners -- set up for .pngs
        # //CHECKERFLIP = tuple(np.flip(self.CHECKERBOARD, 0))
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        objp = np.zeros((1, self.CHECKERBOARD[0]*self.CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        # Loop over images
        self._img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = [os.path.join(self.images_fld, i) for i in os.listdir(self.images_fld) if 'png' in i or 'jpg' in i]
        for fname in images:
            img = cv2.imread(fname)
            if self._img_shape == None:
                self._img_shape = img.shape[:2]
            else:
                assert self._img_shape == img.shape[:2], "All images must share the same size."
            

            calib_image_pre = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
            #increase contrast
            for light_threshold in self.light_thresholds:
                calib_image = calib_image_pre
                
                calib_image[calib_image<dark_threshold] = 0
                calib_image[calib_image>light_threshold] = 255

                if display:
                    cv2.imshow('calibration image',calib_image)  # <- display thresholded image
                    if cv2.waitKey(5) & 0xFF == ord('q'): break

                # Find the chess board corners, if succesful -> stop
                ret, corners = cv2.findChessboardCorners(
                    calib_image, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)
                if ret: break
            print(ret)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(calib_image,corners,(11,11),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                if display:
                    cv2.drawChessboardCorners(calib_image, self.CHECKERBOARD, corners2, ret)
                    cv2.imshow('calibration image', calib_image)
                    cv2.waitKey(500)
        return calib_image, objpoints, imgpoints
              
    def get_calibration_matrices(self):
        '''get_calibration_matrices [give the results of find checkerboard computes the matrices]
        '''

        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                    cv2.fisheye.CALIB_FIX_SKEW  # +cv2.fisheye.CALIB_CHECK_COND

        calib_image, objpoints, imgpoints = self.find_checkerboard()

        N_OK = len(objpoints)
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]


        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                calib_image.shape[::-1],
                self.K,
                self.D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(self._img_shape[::-1]))
        print("K=np.array(" + str(self.K.tolist()) + ")")
        print("D=np.array(" + str(self.D.tolist()) + ")")

    def test_calibration(self, display=True):
        # Display recalibrated images
        DIM = self._img_shape[::-1]
        K = np.array(self.K)
        D = np.array(self.D)

        if self.maps is None: self.compute_maps()

        images = [os.path.join(self.images_fld, i) for i in os.listdir(self.images_fld) if 'png' in i or 'jpg' in i]
        for img in images:
            img = cv2.imread(img)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if display:
                cv2.imshow("correction", img)
                if cv2.waitKey(500) & 0xFF == ord('q'): 
                    break
                cv2.imshow('correction', undistorted_img)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

        # save maps to use in analysis!
        # to be used like: undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.maps = np.zeros((self.calib_image.shape[0], self.calib_image.shape[1], 3)).astype(np.int16)
        self.maps[:, :, 0:2] = map1
        self.maps[:, :, 2] = map2
        np.save(os.path.join(self.images_fld, 'fisheye_maps_' + self.cameraname + '.npy', self.maps))

    def compute_maps(self):
        # Load an image and compute thee maps
        DIM = self._img_shape[::-1]
        K = np.array(self.K)
        D = np.array(self.D)

        if self.maps is None:
            self.load_maps()
        map1 = self.maps[:, :, 0:2]
        map2 = self.maps[:, :, 2]

        img_path = [os.path.join(self.images_fld, i) for i in os.listdir(
            self.images_fld) if 'png' in i or 'jpg' in i][0]
    
        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # dim1 is the dimension of input image to un-distort
        dim1 = img.shape[:2][::-1]
        assert dim1[0]/dim1[1] == DIM[0] / \
            DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        dim2 = dim1  # dim2 is the dimension of remapped image
        dim3 = dim2  # dim3 is the dimension of final output image

        # K, dim2 and balance are used to determine the final K used to un-distort image -- balance = 1 retains all pixels
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, dim2, np.eye(3), balance=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        self.maps = np.zeros((self.calib_image.shape[0], self.calib_image.shape[1], 3)).astype(np.int16)
        self.maps[:, :, 0:2] = map1
        self.maps[:, :, 2] = map2
        np.save(os.path.join(self.images_fld, 'fisheye_maps_' +
                             self.cameraname + '.npy', self.maps))


    def load_maps(self):
        '''load_maps [load maps for a camera that has been calibrated already]
        '''
        self.maps = np.load(os.path.join(self.images_fld, 'fisheye_maps_' + self.cameraname + '.npy'))

    def correct_video(self, video, save_path, is_path=False):
        '''correct_video [corrects fish eye aberrations from videofile]
        
        Arguments:
            video {[str/opencv videofilecapture]} -- [either path to video or cap object]
            save_path {[str]} -- [complete path of target file]
        
        Keyword Arguments:
            is_path {bool} -- [is the video argument a path to a file] (default: {False})
        '''

        if not 'mp4' in save_path: raise ValueError('Unrecognised format for file to save. Supported format: - .mp4 -')

        print('Setting up video correction')
        if is_path:
            if not isinstance(video, str): raise ValueError('is_path is set as True but the video argument is not a string')
            if not 'avi' in video and not 'mp4' in video: raise ValueError('unrecognised video format for video to correct: ', video)
            video = cv2.VideoFileCapture(video)

        # load correction maps
        if self.maps is None: self.load_maps()

        # set up video writer
        width = video.get(3)
        height = video.get(4)
        fps = video.get(5)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        videowriter = cv2.VideoWriter(save_path, fourcc, fps, (width, height), False)

        while True:
            ret, frame = video.read()

            if not ret: break

            gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            undistorted = cv2.remap(gray, self.maps[:, :, 0:2], self.maps[:, :, 2],
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            videowriter.write(undistorted)
        videowriter.release()
        print('Done')
            
    def correct_image(self, img, is_path=False):
        '''correct_image [removes distortion from a single image]
        
        Arguments:
            img {[str, np.ndarray]} -- [either path to an image or image ]
        
        Keyword Arguments:
            is_path {bool} -- [is the img argument a path to an image] (default: {False})

        Returns:
            [np.ndarray] -- [corrected image]
        '''

        if is_path:
            if not isinstance(img, str): raise ValueError('is_path is set as true but the img argument is not a string')
            if not 'png' in img and not 'jpg' in img: raise ValueError('Unrecognised image format')
            img = cv2.imread(img) # <- load image file
        else:
            if not isinstance(img, np.ndarray): raise ValueError('if is_path is False, img argument should be a numpy array.')

        if self.maps is None: self.load_maps()

        undistorted = cv2.remap(img, self.maps[:, :, 0:2], self.maps[:, :, 2],
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted

