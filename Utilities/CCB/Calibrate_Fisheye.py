'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------#                    Calibrate Fisheye Lens                            --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import os; import glob; import cv2


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                         Get Lens Parameters                   --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# Select calibration images folder location
# ------------------------------------------
file_loc = 'C:\Drive\Video Analysis\data\\'
date = ''
mouse_session = 'calibration_images\\'
camera_name = 'fede\\'

file_loc = file_loc + date + mouse_session + camera_name

#name prepended to the saved rectification maps
camera = 'upstairs'


#Set parameters
CHECKERBOARD = (28,12) #size of the checkerboard (# of vertices in each dimension, not including those on the edge)

#Go through and set pixels below the light_threshold to white and pixels above the dark_threshold to black

light_thresholds = np.flip(np.arange(30,160,14),axis=0)
dark_threshold = 30



# -------------------------
# Find checkerboard corners
# -------------------------

#Find checkerboard corners -- set up for .pngs
CHECKERFLIP = tuple(np.flip(CHECKERBOARD,0))
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW #+cv2.fisheye.CALIB_CHECK_COND
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(file_loc + '*.png')
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    
    calib_image_pre = cv2.cvtColor(img.astype(uint8),cv2.COLOR_BGR2GRAY)
    #increase contrast
    for light_threshold in light_thresholds:
        calib_image = calib_image_pre
        
        calib_image[calib_image<dark_threshold] = 0
        calib_image[calib_image>light_threshold] = 255

        cv2.imshow('calibration image',calib_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(calib_image, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)
        
        if ret:
            cv2.waitKey(5)
            break
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(calib_image,corners,(11,11),(-1,-1),subpix_criteria) #11,11?
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(calib_image, CHECKERBOARD, corners2, ret)
        cv2.imshow('calibration image', calib_image)
        cv2.waitKey(500)
        
        
        
# -----------------------------------------------------------------
# Use checkerboard corners to get the calibration matrices K and D
# -----------------------------------------------------------------
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        calib_image.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                Test Calibration and Save Remappings                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# Display recalibrated images
DIM=_img_shape[::-1]
K=np.array(K)
D=np.array(D)
for img_path in glob.glob(file_loc + '*.png'):
    img = cv2.imread(img_path)
    
    
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    dim2 = dim1 #dim2 is the dimension of remapped image
    dim3 = dim2 #dim3 is the dimension of final output image
    
    # K, dim2 and balance are used to determine the final K used to un-distort image -- balance = 1 retains all pixels
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    
    cv2.imshow("correction", img)
    if cv2.waitKey(500) & 0xFF == ord('q'):
       break 
    cv2.imshow('correction', undistorted_img)
    if cv2.waitKey(500) & 0xFF == ord('q'):
       break 
   
# save maps to use in analysis!
# to be used like: undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)       
maps = np.zeros((calib_image.shape[0],calib_image.shape[1],3)).astype(int16)
maps[:,:,0:2] = map1
maps[:,:,2] = map2
np.save(file_loc + 'fisheye_maps_' + camera + '.npy', maps)




        