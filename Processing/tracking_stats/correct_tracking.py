import sys
sys.path.append("./")

import time

from Utilities.imports import *


"""
	OLD - SLOW - FUNCTION FOR TRACKING DATA CORRECTINO
""" 

def correct_tracking_data(uncorrected, M, ypad, xpad, exp_name, sess_uid):

	"""[Corrects tracking data (as extracted by DLC) using a transform Matrix obtained via the CommonCoordinateBehaviour
		toolbox. ]

	Arguments:
		uncorrected {[np.ndarray]} -- [n-by-2 or n-by-3 array where n is number of frames and the columns have X,Y and Velocity ]
		M {[np.ndarray]} -- [2-by-3 transformation matrix: https://github.com/BrancoLab/Common-Coordinate-Behaviour]

	Returns:
		corrected {[np.ndarray]} -- [n-by-3 array with corrected X,Y tracking and Velocity data]
	"""     
	# Do the correction 
	m3d = np.append(M, np.zeros((1,3)),0)
	pads = np.vstack([[xpad, ypad] for i in range(len(uncorrected))])
	padded = np.add(uncorrected, pads)
	corrected = np.zeros_like(uncorrected)

	# ? SLOW
	"""
		x, y = np.add(uncorrected[:, 0], xpad), np.add(uncorrected[:, 1], ypad)  # Shift all traces correctly based on how the frame was padded during alignment 
		for framen in range(uncorrected.shape[0]): # Correct the X, Y for each frame
			xx,yy = x[framen], y[framen]
			corrected[framen, :2] = (np.matmul(m3d, [xx, yy, 1]))[:2]
	"""

	# ! FAST
	# affine transform to match model arena
	concat = np.ones((len(padded), 3))
	concat[:, :2] = padded
	corrected = np.matmul(m3d, concat.T).T[:, :2]

	# Flip the tracking on the Y axis to have the shelter on top
	midline_distance = np.subtract(corrected[:, 1], 490)
	corrected[:, 1] = np.subtract(490, midline_distance)
	return corrected

