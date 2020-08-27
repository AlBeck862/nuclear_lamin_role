"""
# Library of custom functions to be used for the analysis of nuclear lamin.
# usage: import lamin_fxns
# usage: from lamin_fxns import <function>
"""

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
import math

def dot_product(physical, temporal, inverted):
	"""
	# Compute the dot product between two vector sets.
	# The vectors must be stored in ((a,b),((c,d))) format, within a flat array.
	"""
	# Verify vector array format similarity
	if len(physical) != len(temporal):
		raise IndexError("Vector arrays must be the same length.")

	# Initialize an empty array of the appropriate size
	dot_result = np.zeros(len(physical))
	
	# Compute the dot product on the original pixel gradient vectors (inverted=False)
	# or those vectors rotated 90 degrees (inverted=True)
	if inverted:
		for i in range(len(physical)):
			# Get the x-direction and y-direction lengths for each vector.
			vec_a_y = abs(physical[i][1][0] - physical[i][0][0])
			vec_a_x = abs(physical[i][1][1] - physical[i][0][1])
			vec_b_x = abs(temporal[i][1][0] - temporal[i][0][0])
			vec_b_y = abs(temporal[i][1][1] - temporal[i][0][1])

			dot_result[i] = (vec_a_x * vec_b_x) + (vec_a_y * vec_b_y)
	else:
		for i in range(len(physical)):
			# Get the x-direction and y-direction lengths for each vector.
			vec_a_x = abs(physical[i][1][0] - physical[i][0][0])
			vec_a_y = abs(physical[i][1][1] - physical[i][0][1])
			vec_b_x = abs(temporal[i][1][0] - temporal[i][0][0])
			vec_b_y = abs(temporal[i][1][1] - temporal[i][0][1])

			dot_result[i] = (vec_a_x * vec_b_x) + (vec_a_y * vec_b_y)

	return dot_result

def divide_magnitudes(physical, temporal):
	"""
	# Divide the magnitudes of the vectors in each array.
	"""
	# Verify vector array format similarity
	if len(physical) != len(temporal):
		raise IndexError("Vector arrays must be the same length.")
	
	# Initialize an empty array of the appropriate size
	magnitude_quotient = np.zeros(len(physical))
	
	# Compute the magnitudes of each vector
	for i in range(len(physical)):
		phys_x = physical[i][1][0] - physical[i][0][0]
		phys_y = physical[i][1][1] - physical[i][0][1]
		mag_phys = math.sqrt((phys_x**2)+(phys_y**2))
		
		temp_x = temporal[i][1][0] - temporal[i][0][0]
		temp_y = temporal[i][1][1] - temporal[i][0][1]
		mag_temp = math.sqrt((temp_x**2)+(temp_y**2))

		# Divide the magnitudes
		try:
			magnitude_quotient[i] = mag_phys/mag_temp
		# In the case where the displacement vector magnitude is 0, the quotient is forced to 0 to avoid a divide-by-zero error
		except ZeroDivisionError:
			magnitude_quotient[i] = 0.0

	# Convert 0/0 division results (nan) to zero
	np.nan_to_num(magnitude_quotient,False)

	return magnitude_quotient


def ratio_norm(physical, temporal, inverted):
	"""
	# Relate the dot product between physical gradient and displacement vectors to the value of the dot product in the case of perfect alignment.
	# Calls the dot_product function.
	"""
	# Verify vector array format similarity
	if len(physical) != len(temporal):
		raise IndexError("Vector arrays must be the same length.")
	
	# Initialize an empty array of the appropriate size
	perfect_dot = np.zeros(len(physical))
	for i in range(len(physical)):
		phys_x = physical[i][1][0] - physical[i][0][0]
		phys_y = physical[i][1][1] - physical[i][0][1]
		mag_phys = math.sqrt((phys_x**2)+(phys_y**2))
		
		temp_x = temporal[i][1][0] - temporal[i][0][0]
		temp_y = temporal[i][1][1] - temporal[i][0][1]
		mag_temp = math.sqrt((temp_x**2)+(temp_y**2))

		# Simulate a dot product where the vectors are always perfectly aligned
		perfect_dot[i] = mag_phys * mag_temp

	# Obtain the actual dot product between vectors
	true_dot = dot_product(physical=physical,temporal=temporal,inverted=inverted)

	# Compute the ratio between the true dot product value and the perfect alignment equivalents
	norm_result = np.array([true_dot[val]/perfect_dot[val] for val in range(len(physical))])
	
	# Convert 0/0 division results (nan) to zero
	np.nan_to_num(norm_result,False)

	return norm_result

def find_avg_px_intensity(image,cells_row_column,pixels_per_cell):
	"""
	# Determine the average pixel intensity of each cell of a HOG image.
	"""
	avg_pixel_intensities = np.zeros((cells_row_column,cells_row_column))
	
	for i in range(cells_row_column):
		for j in range(cells_row_column):
			sum_pixel_intensity_cell = 0.0
			for a in range(pixels_per_cell*i,(pixels_per_cell*i)+pixels_per_cell):
				for b in range(pixels_per_cell*j,(pixels_per_cell*j)+pixels_per_cell):
					sum_pixel_intensity_cell += image[a][b]
			avg_pixel_intensity_cell = sum_pixel_intensity_cell/(pixels_per_cell**2)
			avg_pixel_intensities[i][j] = avg_pixel_intensity_cell

	return avg_pixel_intensities

def pad_img(image,pixels_per_cell):
	"""
	# Pad an image for compatibility with HOG processing.
	"""
	# If lengths are equal, pad both equally
	if len(image) == len(image[0]):
		pad_row = 0
		pad_column = 0
		while len(image)%pixels_per_cell != 0:
			image = np.pad(image,((0,1),(0,1),(0,0)))
			pad_row += 1
			pad_column += 1
	
	# If x is longer, pad x until %px_per_cell=0, then pad y until equal to x
	elif len(image) > len(image[0]):
		pad_row = 0
		while len(image)%pixels_per_cell != 0:
			image = np.pad(image,((0,1),(0,0),(0,0)))
			pad_row += 1

		pad_column = 0
		while len(image[0]) != len(image):
			image = np.pad(image,((0,0),(0,1),(0,0)))
			pad_column += 1
	
	# If y is longer, pad y until %px_per_cell=0, then pad x until equal to y
	elif len(image[0]) > len(image):
		pad_column = 0
		while len(image[0])%pixels_per_cell != 0:
			image = np.pad(image,((0,0),(0,1),(0,0)))
			pad_column += 1

		pad_row = 0
		while len(image[0]) != len(image):
			image = np.pad(image,((0,1),(0,0),(0,0)))
			pad_row += 1

	# Failsafe
	else:
		print("Image size unrecognized.")
		print("Fatal script error.")
		sys.exit(0)

	return image

def orientation_analysis(dim,img_part,angs):
	"""
	# Determine the orientation of a gradient line in a HOG image cell.
	"""
	max_intensity = 0
	major_intensities = []
	
	for ang in angs:
		result = np.multiply(img_part,ang)
		
		angle_intensity = 0
		
		for p in range(dim):
			for q in range(dim):
				angle_intensity = angle_intensity + result[p][q]
		if angle_intensity == max_intensity:
			major_intensities.append(ang)
		if angle_intensity > max_intensity:
			major_intensities = [ang]
			max_intensity = angle_intensity

	return major_intensities

def force_3d(image):
	"""
	# Ensure that the image is three-dimensional and set the HOG multichannel parameter accordingly.
	"""
	try: 
		image = image[:,:,0:3]
		hog_channel_setting = True
	except:
		if image.ndim == 3:
			print("Invalid number of channels.")
			sys.exit(0)
		else:
			hog_channel_setting = False

	return image,hog_channel_setting