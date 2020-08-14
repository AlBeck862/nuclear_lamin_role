import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Lucas-Kanade Optical Flow Algorithm and Displacement Vector Field Visualization
# Source: https://github.com/chuanenlin/optical-flow/blob/master/sparse-solution.py
# Adapted by Justin de Vries
# July 9, 2020
# ---------------------------------------------------------------------------------------------------------------------

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (45,45), maxLevel = 30, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("Cropped_wild_type_film_low_res.mp4")
# Variable for color to draw optical flow track - (B,G,R)
color = (0, 0, 255)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
#PADDING
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)
# Frame dimensions
width = 454
height = 540
# Creates a video file for saving output at 20 fps (can change at will)
video = cv.VideoWriter("Wild_type_result.mov",cv.VideoWriter_fourcc(*'avc1'),20,(width,height))
video.write(first_frame)

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    #PADDING
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Define coordinates where movement should be tracked - larger step size = more spaced out vectors
    step_size = 15
    coords = np.array([[x,y] for x in range(0,width,step_size) for y in range(0,height,step_size)],np.float32)
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, coords, None, **lk_params)
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
        mask = cv.arrowedLine(mask, (c,d), (a,b), color, 1, tipLength = 0.2)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame,mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    coords = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    cv.imshow("Sparse Optical Flow",output)
    # Saves output frames to video
    video.write(output)
    # Frames are read by intervals of 100 milliseconds (can change at will depending on video frame rate) 
    # The program breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()