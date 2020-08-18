"""
* Python script to determine the role of lamin within cell nuclei.
* usage: python3 lamin_role.py
*				<filename>					# Name of image
*				<cv2 image linger in ms>	# Delay (milliseconds) between CV2 images
*				<"hog" OR "input">			# Select "hog" for vectors to be overlayed on the HOG image, or "input" for vectors to be overlayed on the original image
"""

# Written by Alexander Becker and Justin de Vries

# Importing required libraries
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure
import sys
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from PIL import Image

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (45,45), maxLevel = 30, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture("Cropped_wild_type_film_low_res.mp4")

# Variable for color to draw optical flow track - (B,G,R)
color = (0, 0, 255)

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

#PADDING WOULD GO HERE

# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

# Frame dimensions
width = 454
height = 540

# Creates a video file for saving output at 20 fps (can change at will)
video = cv2.VideoWriter("Wild_type_result.mov",cv2.VideoWriter_fourcc(*'avc1'),20,(width,height))
video.write(first_frame)

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #PADDING WOULD GO HERE

    # Define coordinates where movement should be tracked - larger step size = more spaced out vectors
    step_size = 15
    coords = np.array([[x,y] for x in range(0,width,step_size) for y in range(0,height,step_size)],np.float32)
    
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, coords, None, **lk_params)
    
    # Selects good feature points for previous position
    coords = coords.reshape(-1,1,2)
    good_old = coords[status == 1]
    
    # Selects good feature points for next position
    next = next.reshape(-1,1,2)
    good_new = next[status == 1]
    
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        
        # Draws vector between new and old position
        mask = cv2.arrowedLine(mask, (c,d), (a,b), color, 1, tipLength = 0.2)
    
    # Overlays the optical flow tracks on the original frame
    output = cv2.add(frame,mask)
    
    # Updates previous frame
    prev_gray = gray.copy()
    
    # Updates previous good feature points
    coords = good_new.reshape(-1, 1, 2)
    
    # Opens a new window and displays the output frame
    cv2.imshow("Sparse Optical Flow",output)
    
    # Saves output frames to video
    video.write(output)
    
    # Frames are read by intervals of 100 milliseconds (can change at will depending on video frame rate) 
    # The program breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()