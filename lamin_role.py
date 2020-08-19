"""
* Python script to determine the role of lamin within cell nuclei.
* usage: python3 lamin_role.py
*				<.mp4 file name>			# Name of video
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
from lamin_fxns import orientation_analysis,find_avg_px_intensity,pad_img,force_3d

# The terminal will not skip output lines
np.set_printoptions(threshold=sys.maxsize)

# ---------- Possible Angles ----------
first_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])

second_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])

third_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,1,1,0,0],
                        [0,0,1,1,0,0,0,0],
                        [0,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])

fourth_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])

fifth_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0],
                        [0,0,1,1,0,0,0,0],
                        [0,0,0,0,1,1,0,0],
                        [0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0]])

sixth_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,0]])

seventh_angle = np.array([[0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,1,0,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0,0]])
# ---------- Possible Angles ----------

# Define relative coordinates for each possible gradient line
# All coordinates are based on a first-quadrant, bottom-left-corner origin
# All vectors run from left to right, given the (first,second) point format
first_angle_coords = ((3,6),(4,1))
second_angle_coords = ((2,6),(5,1))
third_angle_coords = ((1,5),(6,2))
fourth_angle_coords = ((1,4),(7,4))
fifth_angle_coords = ((1,2),(6,5))
sixth_angle_coords = ((2,1),(5,6))
seventh_angle_coords = ((3,1),(4,6))
angle_coords = (first_angle_coords,second_angle_coords,third_angle_coords,fourth_angle_coords,fifth_angle_coords,sixth_angle_coords,seventh_angle_coords)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (45,45), maxLevel = 30, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(sys.argv[1])

# Set global CV2 image delay
img_delay = int(sys.argv[2])

# Variable for color to draw optical flow track - (B,G,R)
color = (0, 0, 255)

# Define number of pixels per HOG image cell dimension (square)
px_per_cell = 8

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

# Display the first frame
cv2.imshow('Unaltered Frame',first_frame)
cv2.waitKey(img_delay)

# Force the frame to be RGB (three-dimensional)
first_frame,multi_channel = force_3d(first_frame)

# Display sliced image
cv2.imshow('Sliced to RGB',img)
cv2.waitKey(img_delay)

# Adapts the process to the final image size
cells_per_row_column = int(len(first_frame)/px_per_cell)

# Control for invalid image sizes (padding)
if ((len(first_frame)%px_per_cell != 0) and (len(first_frame[0])%px_per_cell != 0)) or (len(first_frame) != len(first_frame[0])):
    print("The frame will now be padded for compatibility purposes.")
    first_frame = pad_img(first_frame,px_per_cell)
    
    # Acknowledge padding of frame
    print("Padding success")

    # Display padded frame
    cv2.imshow('Padded',first_frame)
    cv2.waitKey(img_delay)

else:
    print("The frame is compatible with this script.")
    print("Padding is not necessary.")

# Failsafe double check
if ((len(first_frame)%px_per_cell != 0) and (len(first_frame[0])%px_per_cell != 0)) or (len(first_frame) != len(first_frame[0])):
    print("Double check: invalid frame size.")
    print("Fatal script error.")
    sys.exit(0)

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Compute and store cell-specific average pixel intensities
avg_px_intensities = find_avg_px_intensity(prev_gray,cells_per_row_column,px_per_cell)

# Generate the HOG feature vector and image
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(px_per_cell, px_per_cell), 
                    cells_per_block=(2, 2), visualize=True, multichannel=multi_channel, feature_vector=False)

# Acknowledge HOG generation
print("HOG generation success")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

# Display the basic (unmodified) HOG image
cv2.imshow('HOG Default', hog_image_rescaled)
cv2.waitKey(img_delay)

# ---------- UNMODIFIED BELOW THIS LINE ----------

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
    
    #PADDING WOULD GO HERE

    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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