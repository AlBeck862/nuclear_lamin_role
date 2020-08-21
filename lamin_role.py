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
from lamin_fxns import orientation_analysis,find_avg_px_intensity,pad_img,force_3d,dot_product

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

# Announce the frame currently being processed
print("***NOTICE***")
print("Now processing frame #1")

# Display the first frame
cv2.imshow('Unaltered Frame: frame 1',first_frame)
cv2.waitKey(img_delay)

# Force the frame to be RGB (three-dimensional)
first_frame,multi_channel = force_3d(first_frame)

# Display sliced image
cv2.imshow('Sliced to RGB: frame 1',first_frame)
cv2.waitKey(img_delay)

# Adapts the process to the final image size
cells_per_row_column = int(len(first_frame)/px_per_cell)

print(len(first_frame))
print(len(first_frame[0]))

# Control for invalid image sizes (padding)
if ((len(first_frame)%px_per_cell != 0) and (len(first_frame[0])%px_per_cell != 0)) or (len(first_frame) != len(first_frame[0])):
    print("Frame 1 will now be padded for compatibility purposes.")
    first_frame = pad_img(first_frame,px_per_cell)
    
    # Acknowledge padding of frame
    print("Padding success")

    # Display padded frame
    cv2.imshow('Padded: frame 1',first_frame)
    cv2.waitKey(img_delay)

else:
    print("Frame 1 is compatible with this script.")
    print("Padding is not necessary.")

# Failsafe double check
if ((len(first_frame)%px_per_cell != 0) and (len(first_frame[0])%px_per_cell != 0)) or (len(first_frame) != len(first_frame[0])):
    print("Double check: invalid frame size.")
    print("Fatal script error.")
    sys.exit(0)

print(len(first_frame))
print(len(first_frame[0]))

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Compute and store cell-specific average pixel intensities
avg_px_intensities = find_avg_px_intensity(prev_gray,cells_per_row_column,px_per_cell)

# Generate the HOG feature vector and image
fd, hog_image = hog(first_frame, orientations=9, pixels_per_cell=(px_per_cell, px_per_cell), 
                    cells_per_block=(2, 2), visualize=True, multichannel=multi_channel, feature_vector=False)

# Acknowledge HOG generation
print("HOG generation success: frame 1")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

# Display the basic (unmodified) HOG image
cv2.imshow(f'HOG Default: frame 1', hog_image_rescaled)
cv2.waitKey(img_delay)

# ---------- Image cell manipulation ----------
# Create the vector data array: one pair of coordinates for each gradient line
vectors = np.zeros((cells_per_row_column,cells_per_row_column),dtype=object)

# Segments the image into 8x8px cells which can be manipulated as units
# i controls the rows, j controls the columns
for i in range(cells_per_row_column):
    for j in range(cells_per_row_column):
        img_section = np.zeros((px_per_cell,px_per_cell))
        n = 0
        for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
            m = 0
            for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                img_section[n][m] = hog_image_rescaled[a][b]
                m += 1
            n += 1

        # Call a custom function on every mini-image, checking which orientation is dominant
        angles = (first_angle, second_angle, third_angle, fourth_angle, fifth_angle, sixth_angle, seventh_angle)
        orientation_result = orientation_analysis(px_per_cell,img_section,angles)
        selected_orientation = orientation_result[0]

        # Modify output images given orientation data
        g = 0
        for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
            h = 0
            for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                if selected_orientation[g][h] == 0:
                    hog_image_rescaled[a][b] = 0
                h += 1
            g += 1
        
        # Create vector data for the given image cell
        x_shift = px_per_cell*j
        y_shift = px_per_cell*i
        for num in range(len(angles)):
            if not (selected_orientation - angles[num]).any():
                vectors[i][j] = [[angle_coords[num][0][0]+x_shift,angle_coords[num][0][1]+y_shift],[angle_coords[num][1][0]+x_shift,angle_coords[num][1][1]+y_shift]]
        
        # No-gradient image cells are assigned zero-coordinates
        if vectors[i][j] == 0:
            vectors[i][j] = ((0,0),(0,0))

        # Safeguard against no-gradient image cells
        if vectors[i][j] == ((0,0),(0,0)):
            print("Warning: empty image cell detected.")

# Acknowledge gradient selection
print("Gradient selection success: frame 1")

# Display the grayscale selected-gradients image
cv2.imshow('Selected Gradients: frame 1',hog_image_rescaled)
cv2.waitKey(img_delay)
# ---------- Image cell manipulation ----------

# ---------- Generate and display gradient vectors ----------
# Get the largest average intensity in the original image
max_intensity = np.max(avg_px_intensities)

# Divide each element by the maximum value, normalizing all vector lengths (*3 for visualization purposes only)
avg_px_intensities_normalized = 3*(avg_px_intensities/max_intensity)

# Define the position and length of each vector (using list comprehensions)
x_positions = [vectors[i][j][0][0] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
y_positions = [vectors[i][j][0][1] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
dx_vals = [(vectors[i][j][1][0]-vectors[i][j][0][0])*avg_px_intensities_normalized[i][j] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
dy_vals = [(vectors[i][j][0][1]-vectors[i][j][1][1])*avg_px_intensities_normalized[i][j] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]

# Recombine all vector data into one array for later use and convert data to tuple
for i in range(cells_per_row_column):
    for j in range(cells_per_row_column):
        vectors[i][j][1][0] = vectors[i][j][0][0]+((vectors[i][j][1][0]-vectors[i][j][0][0])*avg_px_intensities_normalized[i][j])
        vectors[i][j][1][1] = vectors[i][j][0][1]+((vectors[i][j][1][1]-vectors[i][j][0][1])*avg_px_intensities_normalized[i][j])
        vectors[i][j][0] = tuple(vectors[i][j][0])
        vectors[i][j][1] = tuple(vectors[i][j][1])
        vectors[i][j] = tuple(vectors[i][j])

# Duplicate the HOG image to enable returning both a vector output and a clean output
img_vector = hog_image_rescaled

# Set the image on which vectors will be overlayed
if sys.argv[3] == "hog":
    display_img = img_vector
elif sys.argv[3] == "input":
    display_img = img
else:
    print("Invalid argument in third position.")
    sys.exit(0)

# Define output image parameters
screen_dpi = 227
plt.figure(figsize=(len(img_vector)/screen_dpi,len(img_vector)/screen_dpi),dpi=screen_dpi)
plt.imshow(
    display_img,            # Image name
    alpha=1.0,              # Transparency setting
    cmap="Greys_r",         # Grayscale colour map
    origin="upper",         # Image origin in the top left corner
    interpolation="none",   # Image blurring off
    resample=False,         # Image resampling off
    aspect="equal")         # Image distorting off

# Draw the vectors
plt.quiver(
    x_positions,            # X coordinate of start point
    y_positions,            # Y coordinate of start point
    dx_vals,                # X-direction movement from start point to end point
    dy_vals,                # Y-direction movement from start point to end point
    color='r',              # Vector colour (red)
    scale_units="dots",     # Vector scaling unit (pixels)
    scale=0.4,              # Vector scaling factor
    headwidth=3,            # Vector head width as a multiple of shaft width
    headlength=3,           # Vector head length as a multiple of shaft length
    headaxislength=2.5)     # Vector head length at shaft intersection

# Finalize and display the vector image
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.show()
# ---------- Generate and display gradient vectors ----------

# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

# Frame dimensions
width = len(first_frame)
height = len(first_frame[0])

# Creates a video file for saving output at 20 fps (can change at will)
video = cv2.VideoWriter("result.mov",cv2.VideoWriter_fourcc(*'avc1'),20,(width,height))
video.write(first_frame)

counter = 2
while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    
    # Break out of the loop if there are no more frames to read
    if ret == False:
        break

    # Announce the frame currently being processed
    print("***NOTICE***")
    print(f"Now processing frame #{counter}")

    # Display the frame
    cv2.imshow(f'Unaltered Frame: frame {counter}',frame)
    cv2.waitKey(img_delay)

    # Force the frame to be RGB (three-dimensional)
    frame,multi_channel = force_3d(frame)

    # Display sliced image
    cv2.imshow(f'Sliced to RGB: frame {counter}',frame)
    cv2.waitKey(img_delay)

    # Adapts the process to the final image size
    cells_per_row_column = int(len(frame)/px_per_cell)

    print(len(frame))
    print(len(frame[0]))

    # Control for invalid image sizes (padding)
    if ((len(frame)%px_per_cell != 0) and (len(frame[0])%px_per_cell != 0)) or (len(frame) != len(frame[0])):
        print(f"Frame {counter} will now be padded for compatibility purposes.")
        frame = pad_img(frame,px_per_cell)
        
        # Acknowledge padding of frame
        print(f"Padding success: frame {counter}")

        # Display padded frame
        cv2.imshow(f'Padded: frame {counter}',frame)
        cv2.waitKey(img_delay)

    else:
        print(f"Frame {counter} is compatible with this script.")
        print("Padding is not necessary.")

    # Failsafe double check
    if ((len(frame)%px_per_cell != 0) and (len(frame[0])%px_per_cell != 0)) or (len(frame) != len(frame[0])):
        print("Double check: invalid frame size.")
        print("Fatal script error.")
        sys.exit(0)

    print(len(first_frame))
    print(len(first_frame[0]))
    
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute and store cell-specific average pixel intensities
    avg_px_intensities = find_avg_px_intensity(gray,cells_per_row_column,px_per_cell)

    # Generate the HOG feature vector and image
    fd, hog_image = hog(frame, orientations=9, pixels_per_cell=(px_per_cell, px_per_cell), 
                        cells_per_block=(2, 2), visualize=True, multichannel=multi_channel, feature_vector=False)

    # Acknowledge HOG generation
    print(f"HOG generation success: frame {counter}")

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

    # Display the basic (unmodified) HOG image
    cv2.imshow(f'HOG Default: frame {counter}', hog_image_rescaled)
    cv2.waitKey(img_delay)

    # ---------- Image cell manipulation ----------
    # Create the vector data array: one pair of coordinates for each gradient line
    vectors = np.zeros((cells_per_row_column,cells_per_row_column),dtype=object)

    # Segments the image into 8x8px cells which can be manipulated as units
    # i controls the rows, j controls the columns
    for i in range(cells_per_row_column):
        for j in range(cells_per_row_column):
            img_section = np.zeros((px_per_cell,px_per_cell))
            n = 0
            for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
                m = 0
                for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                    img_section[n][m] = hog_image_rescaled[a][b]
                    m += 1
                n += 1

            # Call a custom function on every mini-image, checking which orientation is dominant
            angles = (first_angle, second_angle, third_angle, fourth_angle, fifth_angle, sixth_angle, seventh_angle)
            orientation_result = orientation_analysis(px_per_cell,img_section,angles)
            selected_orientation = orientation_result[0]

            # Modify output images given orientation data
            g = 0
            for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
                h = 0
                for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                    if selected_orientation[g][h] == 0:
                        hog_image_rescaled[a][b] = 0
                    h += 1
                g += 1
            
            # Create vector data for the given image cell
            x_shift = px_per_cell*j
            y_shift = px_per_cell*i
            for num in range(len(angles)):
                if not (selected_orientation - angles[num]).any():
                    vectors[i][j] = [[angle_coords[num][0][0]+x_shift,angle_coords[num][0][1]+y_shift],[angle_coords[num][1][0]+x_shift,angle_coords[num][1][1]+y_shift]]
            
            # No-gradient image cells are assigned zero-coordinates
            if vectors[i][j] == 0:
                vectors[i][j] = ((0,0),(0,0))

            # Safeguard against no-gradient image cells
            if vectors[i][j] == ((0,0),(0,0)):
                print("Warning: empty image cell detected.")

    # Acknowledge gradient selection
    print(f"Gradient selection success: frame {counter}")

    # Display the grayscale selected-gradients image
    cv2.imshow(f'Selected Gradients: frame {counter}',hog_image_rescaled)
    cv2.waitKey(img_delay)
    # ---------- Image cell manipulation ----------

    # ---------- Generate and display gradient vectors ----------
    # Get the largest average intensity in the original image
    max_intensity = np.max(avg_px_intensities)

    # Divide each element by the maximum value, normalizing all vector lengths (*3 for visualization purposes only)
    avg_px_intensities_normalized = 3*(avg_px_intensities/max_intensity)

    # Define the position and length of each vector (using list comprehensions)
    x_positions = [vectors[i][j][0][0] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
    y_positions = [vectors[i][j][0][1] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
    dx_vals = [(vectors[i][j][1][0]-vectors[i][j][0][0])*avg_px_intensities_normalized[i][j] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]
    dy_vals = [(vectors[i][j][0][1]-vectors[i][j][1][1])*avg_px_intensities_normalized[i][j] for i in range(cells_per_row_column) for j in range(cells_per_row_column)]

    # Recombine all vector data into one array for later use and convert data to tuple
    for i in range(cells_per_row_column):
        for j in range(cells_per_row_column):
            vectors[i][j][1][0] = vectors[i][j][0][0]+((vectors[i][j][1][0]-vectors[i][j][0][0])*avg_px_intensities_normalized[i][j])
            vectors[i][j][1][1] = vectors[i][j][0][1]+((vectors[i][j][1][1]-vectors[i][j][0][1])*avg_px_intensities_normalized[i][j])
            vectors[i][j][0] = tuple(vectors[i][j][0])
            vectors[i][j][1] = tuple(vectors[i][j][1])
            vectors[i][j] = tuple(vectors[i][j])

    # Duplicate the HOG image to enable returning both a vector output and a clean output
    img_vector = hog_image_rescaled

    # Set the image on which vectors will be overlayed
    if sys.argv[3] == "hog":
        display_img = img_vector
    elif sys.argv[3] == "input":
        display_img = img
    else:
        print("Invalid argument in third position.")
        sys.exit(0)

    # Define output image parameters
    screen_dpi = 227
    plt.figure(figsize=(len(img_vector)/screen_dpi,len(img_vector)/screen_dpi),dpi=screen_dpi)
    plt.imshow(
        display_img,            # Image name
        alpha=1.0,              # Transparency setting
        cmap="Greys_r",         # Grayscale colour map
        origin="upper",         # Image origin in the top left corner
        interpolation="none",   # Image blurring off
        resample=False,         # Image resampling off
        aspect="equal")         # Image distorting off

    # Draw the vectors
    plt.quiver(
        x_positions,            # X coordinate of start point
        y_positions,            # Y coordinate of start point
        dx_vals,                # X-direction movement from start point to end point
        dy_vals,                # Y-direction movement from start point to end point
        color='r',              # Vector colour (red)
        scale_units="dots",     # Vector scaling unit (pixels)
        scale=0.4,              # Vector scaling factor
        headwidth=3,            # Vector head width as a multiple of shaft width
        headlength=3,           # Vector head length as a multiple of shaft length
        headaxislength=2.5)     # Vector head length at shaft intersection

    # Finalize and display the vector image
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.show()
    # ---------- Generate and display gradient vectors ----------

    # Define coordinates where movement should be tracked - larger step size = more spaced out vectors
    step_size = px_per_cell
    coords = np.array([[x,y] for x in range(0,width,step_size) for y in range(0,height,step_size)],np.float32)
    
    # Calculation of sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, coords, None, **lk_params)
    
    # Selects good feature points for previous position
    coords = coords.reshape(-1,1,2)
    good_old = coords[status == 1]
    
    # Selects good feature points for next position
    next_pts = next_pts.reshape(-1,1,2)
    good_new = next_pts[status == 1]

    # Draws the optical flow tracks
    disp_vectors = np.zeros(len(coords),dtype=object)
    for i, (new, old) in enumerate(zip(next_pts, coords)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws vector between new and old position
        mask = cv2.arrowedLine(mask, (c,d), (a,b), color, 1, tipLength = 0.2)
        # Create a new array to store all displacement vectors
        disp_vectors[i] = ((c,d),(a,b))

    # Overlays the optical flow tracks on the original frame
    output = cv2.add(frame,mask)
    
    # Flatten the lamin-density array to line up the data with the displacement vector array
    vectors = vectors.flatten()

    # Compute the dot product, returned as an array of length len(vectors)
    dot_prod_result = dot_product(vectors,disp_vectors)

    # DOT PRODUCT FAILS FOR WILD TYPE: VECTORS ARE NOT THE SAME LENGTH
    # CAUSE UNKNOWN

    # Reconfigure the dot product results for accurate display
    display_dot_result = np.zeros((len(hog_image_rescaled),len(hog_image_rescaled)))
    dot_prod_result = dot_prod_result.reshape((cells_per_row_column,cells_per_row_column))

    for i in range(cells_per_row_column):
        for j in range(cells_per_row_column):
            for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
                for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                    display_dot_result[a][b] = dot_prod_result[i][j]

    # Define heatmap parameters
    screen_dpi = 227
    plt.figure(figsize=(len(img_vector)/screen_dpi,len(img_vector)/screen_dpi),dpi=screen_dpi)
    plt.imshow(display_dot_result, cmap="hot")

    # Display the heatmap according to the dot product results
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.show()

    # NEXT: ADD COLOUR BAR FOR SCALE + REFINE AND CONFIRM OUTPUT ACCURACY

    # Updates previous frame
    prev_gray = gray.copy()
    
    # Updates previous good feature points
    coords = good_new.reshape(-1, 1, 2)
    
    # Displays the frame overlaid with displacement vectors
    cv2.imshow(f"Sparse Optical Flow: frames {counter-1} to {counter}",output)
    cv2.waitKey(img_delay)

    # ---------- Overlay both vector sets ----------
    # Define output image parameters
    screen_dpi = 227
    plt.figure(figsize=(len(img_vector)/screen_dpi,len(img_vector)/screen_dpi),dpi=screen_dpi)
    plt.imshow(
        output,                 # Image name
        alpha=1.0,              # Transparency setting
        cmap="Greys_r",         # Grayscale colour map
        origin="upper",         # Image origin in the top left corner
        interpolation="none",   # Image blurring off
        resample=False,         # Image resampling off
        aspect="equal")         # Image distorting off

    # Draw the vectors
    plt.quiver(
        x_positions,            # X coordinate of start point
        y_positions,            # Y coordinate of start point
        dx_vals,                # X-direction movement from start point to end point
        dy_vals,                # Y-direction movement from start point to end point
        color='r',              # Vector colour (red)
        scale_units="dots",     # Vector scaling unit (pixels)
        scale=0.4,              # Vector scaling factor
        headwidth=3,            # Vector head width as a multiple of shaft width
        headlength=3,           # Vector head length as a multiple of shaft length
        headaxislength=2.5)     # Vector head length at shaft intersection

    # Finalize and display the vector image
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.show()
    # ---------- Overlay both vector sets ----------

    # Saves output frames to video
    video.write(output)

    # Frames are read by intervals of 100 milliseconds (can change at will depending on video frame rate) 
    # The program breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # Increase frame counter by one
    counter += 1

# The following frees up resources and closes all windows
cap.release()
cv2.destroyAllWindows()