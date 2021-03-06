"""
* Python script to determine the role of lamin within cell nuclei.
* usage: python3 lamin_role.py
*				<.mp4 file name>			# Name of video
*				<cv2 image linger in ms>	# Delay (milliseconds) between CV2 images
*				<"hog" OR "input">			# Select "hog" for vectors to be overlaid on the HOG image, or "input" for vectors to be overlayed on the original image
"""

# Written by Alexander Becker and Justin de Vries

# Importing required external libraries
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure
import sys
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# Importing required custom functions
from lamin_fxns import *
from lamin_image import LaminImage

# ---------- Output Parameters ----------
# Rotate physical gradient vectors 90 degrees?
inversion = False

# Heatmap normalization method
normalize = "no"    # Options: "no", "ratio", "max", "division", "flip_division"

# Vector scaling reference image (original input image OR HOG image)
vector_scaling = "original" # Options: "hog", "original"

# Delay (milliseconds) between CV2 images
img_delay = int(sys.argv[2])

# Select "hog" for vectors to be overlaid on the HOG image, or "input" for vectors to be overlayed on the original image 
overlay_img = sys.argv[3]
# ---------- Output Parameters ----------

# Ignore console warnings
warnings.filterwarnings("ignore")

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
lk_params = dict(winSize = (30,30), maxLevel = 30, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.03))

# The video feed is read in as a VideoCapture object
cap = cv2.VideoCapture(sys.argv[1])

# Variable for color to draw optical flow track - (B,G,R)
color = (0, 0, 255)

# Define number of pixels per HOG image cell dimension (square)
px_per_cell = 8

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()

# Create a new image object
test_frame = LaminImage(first_frame,px_per_cell,img_delay)
test_frame.show_image() #show the imported image
multi_channel = test_frame.force_3d() #convert to RGB
test_frame.show_image() #show the RGB image
test_frame.pad() #pad
test_frame.show_image() #show the padded image

# Adapts the process to the final image size
cells_per_row_column = int(len(test_frame.image)/test_frame.px_per_cell)

# Store the original (non-HOG) image's grayscale data for later use
prev_gray = cv2.cvtColor(test_frame.image, cv2.COLOR_BGR2GRAY)

first_frame = test_frame.image #temporary compatibility solution: mask requires first_frame (currently, line 294) *********************

test_frame.compute_hog()
test_frame.show_image() #show the (rescaled) HOG image

hog_image_rescaled = test_frame.image #temporary compatibility solution: return to original image format ********************

###### UNALTERED ORIGINAL SCRIPT ######

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

# Compute and store cell-specific average pixel intensities
if vector_scaling == "hog":
    avg_px_intensities = find_avg_px_intensity(hog_image_rescaled,cells_per_row_column,px_per_cell)
elif vector_scaling == "original":
    avg_px_intensities = find_avg_px_intensity(prev_gray,cells_per_row_column,px_per_cell)
else:
    print("Invalid vector scaling parameter.")
    sys.exit(0)
# ---------- Image cell manipulation ----------

# ---------- Generate and display gradient vectors ----------
# Get the largest average intensity in the original image
max_intensity = np.max(avg_px_intensities)

# Divide each element by the maximum value, normalizing all vector lengths (*3 for visualization purposes only)
avg_px_intensities_normalized = (avg_px_intensities/max_intensity)

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
if overlay_img == "hog":
    display_img = img_vector
elif overlay_img == "input":
    display_img = first_frame
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
video = cv2.VideoWriter("result.mov",cv2.VideoWriter_fourcc(*'mp4v'),1,(width,height))
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
    
    # Adapts the process to the final image size
    cells_per_row_column = int(len(frame)/px_per_cell)

    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    # Store the "previous" set of vectors for later use to generate the heatmap
    heatmap_vectors = vectors

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

    # Compute and store cell-specific average pixel intensities
    if vector_scaling == "hog":
        avg_px_intensities = find_avg_px_intensity(hog_image_rescaled,cells_per_row_column,px_per_cell)
    elif vector_scaling == "original":
        avg_px_intensities = find_avg_px_intensity(gray,cells_per_row_column,px_per_cell)
    else:
        print("Invalid vector scaling parameter.")
        sys.exit(0)
    # ---------- Image cell manipulation ----------

    # ---------- Generate and display gradient vectors ----------
    # Get the largest average intensity in the original image
    max_intensity = np.max(avg_px_intensities)

    # Divide each element by the maximum value, normalizing all vector lengths (*3 for visualization purposes only)
    avg_px_intensities_normalized = (avg_px_intensities/max_intensity)

    # Store the initial vector data for the vector overlap image later on
    old_x_positions = x_positions
    old_y_positions = y_positions
    old_dx_vals = np.array(dx_vals)
    old_dy_vals = np.array(dy_vals)

    # Define the (new) position and length of each vector (using list comprehensions)
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
    if overlay_img == "hog":
        display_img = img_vector
    elif overlay_img == "input":
        display_img = frame
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
    heatmap_vectors = heatmap_vectors.flatten()

    # Compute the dot product, returned as an array of length len(heatmap_vectors)
    # Use keyword arguments to ensure correct function usage.
    dot_prod_result = dot_product(physical=heatmap_vectors,     # Physical gradient vectors
                                temporal=disp_vectors,          # Displacement vectors
                                inverted=inversion)             # Rotate physical gradient vectors by 90 degrees?

    # Get max value in the dot product array
    dot_prod_max = max(dot_prod_result)
    
    # Normalize the output and define related parameters
    if normalize == "max":
        normalized_result = np.array([val/dot_prod_max for val in dot_prod_result])
        tick_vals = None
        extension="neither"
        colorbar_label="Dot Product Result Normalized to Max"
    elif normalize == "ratio":
        normalized_result = ratio_norm(physical=heatmap_vectors,temporal=disp_vectors,inverted=inversion)
        tick_vals = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
        extension="max"
        colorbar_label="Vector Alignment Ratio"
    elif normalize == "division":
        normalized_result = divide_magnitudes(physical=heatmap_vectors,temporal=disp_vectors)
        tick_vals = None
        extension="neither"
        colorbar_label="PLACEHOLDER"
    elif normalize == "flip_division":
        normalized_result = divide_magnitudes(physical=disp_vectors,temporal=heatmap_vectors) # The reverse of physical and temporal vectors here is intentional.
        tick_vals = None
        extension="neither"
        colorbar_label="PLACEHOLDER"
    else:
        normalized_result = dot_prod_result
        tick_vals = None
        extension="neither"
        colorbar_label="Dot Product Result"

    # Reconfigure the dot product results for accurate display
    display_result = np.zeros((len(hog_image_rescaled),len(hog_image_rescaled)))
    normalized_result = normalized_result.reshape((cells_per_row_column,cells_per_row_column))

    for i in range(cells_per_row_column):
        for j in range(cells_per_row_column):
            for a in range(px_per_cell*i,(px_per_cell*i)+px_per_cell):
                for b in range(px_per_cell*j,(px_per_cell*j)+px_per_cell):
                    display_result[a][b] = normalized_result[i][j]

    # ---------- Vector Magnitude Plot ----------
    # Compute vector magnitudes
    physical_mag,temporal_mag = compute_magnitudes(physical=heatmap_vectors,temporal=disp_vectors)

    # Initialize a new figure
    plt.figure()

    # Plot the displacement mangitudes against the intensity magnitudes
    plt.plot(physical_mag,temporal_mag,linestyle="None",marker=".",markersize=3.0)

    # Set plot properties
    plt.title("Displacement versus Intensity")
    plt.xlabel("Intensity")
    plt.ylabel("Displacement")
    # ---------- Vector Magnitude Plot ----------

    # ---------- Heatmap generation ----------
    # Initialize a new figure
    plt.figure()

    # Select image for display, as well as colormap used
    plt.imshow(display_result, cmap="hot")
    
    # Define colorbar parameters
    plt.colorbar(orientation="horizontal",      # Horizontal colorbar
                 label=colorbar_label,          # Label under colorbar
                 shrink=0.6,                    # Relative size of colorbar
                 pad=0.05,                      # Relative distance between colorbar and image
                 ticks=tick_vals,               # Define the colorbar ticks
                 extend=extension,              # Type of colorbar, if applicable
                 extendrect=True)               # Type of colorbar extension, if applicable
    
    # Disable plot ticks
    plt.tick_params(which="both",       # Modify both x- and y-axes
                    bottom=False,       # No ticks on bottom of figure
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,  # No tick labels on bottom of figure
                    labeltop=False,
                    labelleft=False,
                    labelright=False)

    # Set figure title
    plt.title("PLACEHOLDER")

    # Display the heatmap according to the dot product results
    plt.show()
    # ---------- Heatmap generation ----------
    
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
        old_x_positions,      # X coordinate of start point
        old_y_positions,      # Y coordinate of start point
        old_dx_vals,        # X-direction movement from start point to end point
        old_dy_vals,        # Y-direction movement from start point to end point
        color='r',            # Vector colour (red)
        scale_units="dots",   # Vector scaling unit (pixels)
        scale=0.4,            # Vector scaling factor
        headwidth=3,          # Vector head width as a multiple of shaft width
        headlength=3,         # Vector head length as a multiple of shaft length
        headaxislength=2.5)   # Vector head length at shaft intersection

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

# Free up resources and close all CV2 windows
cap.release()
cv2.destroyAllWindows()