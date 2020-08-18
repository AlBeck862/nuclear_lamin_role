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