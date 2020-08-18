"""
* Python script to process images using Skimage's HOG function, as well as extensive custom analysis methods.
* usage: python3 hog_processing.py
*				<filename>					# Name of image
*				<cv2 image linger in ms>	# Delay (milliseconds) between CV2 images
*				<"hog" OR "input">			# Select "hog" for vectors to be overlayed on the HOG image, or "input" for vectors to be overlayed on the original image
"""

# HOG information: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/?utm_source=blog&utm_medium=3-techniques-extract-features-from-image-data-machine-learning
# Written by Alexander Becker

# Importing required libraries
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure
import sys
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from PIL import Image
from lamin_fxns import orientation_analysis,find_avg_px_intensity,pad_img

# The terminal will not skip output lines.
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
# ---------- Possible Angles ----------

# ---------- Basic HOG image generation ----------
# Read the input image
img = imread(sys.argv[1])

# Set global image delay
img_delay = int(sys.argv[2])

cv2.imshow('Original',img)
cv2.waitKey(img_delay)

# Force the image to be RGB (three-dimensional)
try: 
	img = img[:,:,0:3]
	multi_channel = True
except:
	if img.ndim == 3:
		print("Invalid number of channels.")
		sys.exit(0)
	else:
		multi_channel = False

cv2.imshow('Sliced to RGB',img)
cv2.waitKey(img_delay)

# Define number of pixels per image cell dimension (square)
px_per_cell = 8

# Control for invalid image sizes
if ((len(img)%px_per_cell != 0) and (len(img[0])%px_per_cell != 0)) or (len(img) != len(img[0])):
	print("The image will now be padded for compatibility purposes.")
	img = pad_img(img,px_per_cell)
	
	# Acknowledge padding of image
	print("Padding success")

	cv2.imshow('Padded',img)
	cv2.waitKey(img_delay)

else:
	print("The image is compatible with this script.")
	print("Padding is not necessary.")

# Failsafe double check
if ((len(img)%px_per_cell != 0) and (len(img[0])%px_per_cell != 0)) or (len(img) != len(img[0])):
	print("Double check: invalid image size.")
	print("Fatal script error.")
	sys.exit(0)

# Adapts the process to the final image size
cells_per_row_column = int(len(img)/px_per_cell)

# Compute and store cell-specific average pixel intensities
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
avg_px_intensities = find_avg_px_intensity(gray_image,cells_per_row_column,px_per_cell)

#creating hog features
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(px_per_cell, px_per_cell), 
                    cells_per_block=(2, 2), visualize=True, multichannel=multi_channel, feature_vector=False)

# Acknowledge HOG generation
print("HOG generation success")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

# Display the basic (unmodified) HOG image
cv2.imshow('HOG Default', hog_image_rescaled)
cv2.waitKey(img_delay)
# ---------- Basic HOG image generation ----------

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
print("Gradient selection success")
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
	display_img,			# Image name
	alpha=1.0,				# Transparency setting
	cmap="Greys_r",			# Grayscale colour map
	origin="upper",			# Image origin in the top left corner
	interpolation="none",	# Image blurring off
	resample=False,			# Image resampling off
	aspect="equal")			# Image distorting off

# Draw the vectors
plt.quiver(
	x_positions,			# X coordinate of start point
	y_positions,			# Y coordinate of start point
	dx_vals,				# X-direction movement from start point to end point
	dy_vals,				# Y-direction movement from start point to end point
	color='r',				# Vector colour (red)
	scale_units="dots",		# Vector scaling unit (pixels)
	scale=0.4,				# Vector scaling factor
	headwidth=3,			# Vector head width as a multiple of shaft width
	headlength=3,			# Vector head length as a multiple of shaft length
	headaxislength=2.5)		# Vector head length at shaft intersection

# Finalize and display the vector image
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.show()
# ---------- Generate and display gradient vectors ----------

# ---------- Reverse the image padding ----------
try:
	for unpad_row in range(pad_row):
		hog_image_rescaled = hog_image_rescaled[0:-1,:]

	for unpad_col in range(pad_column):
		hog_image_rescaled = hog_image_rescaled[:,0:-1]

	# Acknowledge un-padding
	print("Padding reversal success")

# Bypass un-padding if the image was not padded in the first place
except NameError:
	pass
# ---------- Reverse the image padding ----------

# Display the grayscale selected-gradients image
cv2.imshow('Selected Gradients',hog_image_rescaled)
cv2.waitKey(img_delay)

# Close all image windows
cv2.destroyAllWindows()