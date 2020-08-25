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

def dot_product(physical, temporal, inverted):
	"""
	# Compute the dot product between two vector sets.
	# The vectors must be stored in ((a,b),((c,d))) format, within a flat array.
	"""
	if len(physical) != len(temporal):
		raise IndexError("Vector arrays must be the same length.")

	dot_result = np.zeros(len(physical))
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