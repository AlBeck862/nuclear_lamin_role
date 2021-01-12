"""
* Python script for the LaminImage class.
* importing: from lamin_image import LaminImage
* usage: call class methods to analyze images
"""

# Written by Alexander Becker

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
import itertools

# Importing required custom functions
from lamin_fxns import *

class LaminImage:
	# Track the number of times the image has been shown on the screen.
	version_number = 0

	image_numbers = itertools.count(1)

	def __init__(self, image, px_per_cell,img_delay):
		self.image = image
		self.px_per_cell = px_per_cell
		self.img_delay = img_delay
		self.hog_channel_setting = None
		
		self.image_number = next(self.image_numbers)
		print("New lamin image created. This is image #" + str(self.image_number) + ".")

	def show_image(self):
		"""Show the current image on the screen."""
		self.version_number += 1 #ensures that each time a version of a particular image is shown, it will be given a new title
		cv2.imshow('Image #' + str(self.image_number) + ", Version #" + str(self.version_number),self.image)
		cv2.waitKey(self.img_delay)

	def force_3d(self):
		"""Ensure that the image is three-dimensional and set the HOG multichannel parameter accordingly."""
		try: 
			self.image = self.image[:,:,0:3]
			self.hog_channel_setting = True
			print("Sliced to RGB successfully.")
		except:
			if self.image.ndim == 3:
				print("Invalid number of channels.")
				sys.exit(0)
			else:
				self.hog_channel_setting = False

	def pad(self):
		"""Pad an image for compatibility with HOG processing."""
		if ((len(self.image)%self.px_per_cell != 0) and (len(self.image[0])%self.px_per_cell != 0)) or (len(self.image) != len(self.image[0])):
			# If lengths are equal, pad both equally
			if len(self.image) == len(self.image[0]):
				pad_row = 0
				pad_column = 0
				while len(self.image)%self.px_per_cell != 0:
					self.image = np.pad(self.image,((0,1),(0,1),(0,0)))
					pad_row += 1
					pad_column += 1
			
			# If x is longer, pad x until %px_per_cell=0, then pad y until equal to x
			elif len(self.image) > len(self.image[0]):
				pad_row = 0
				while len(self.image)%self.px_per_cell != 0:
					self.image = np.pad(self.image,((0,1),(0,0),(0,0)))
					pad_row += 1

				pad_column = 0
				while len(self.image[0]) != len(self.image):
					self.image = np.pad(self.image,((0,0),(0,1),(0,0)))
					pad_column += 1
			
			# If y is longer, pad y until %px_per_cell=0, then pad x until equal to y
			elif len(self.image[0]) > len(self.image):
				pad_column = 0
				while len(self.image[0])%self.px_per_cell != 0:
					self.image = np.pad(self.image,((0,0),(0,1),(0,0)))
					pad_column += 1

				pad_row = 0
				while len(self.image[0]) != len(self.image):
					self.image = np.pad(self.image,((0,1),(0,0),(0,0)))
					pad_row += 1

			# Failsafe
			else:
				print("Image size unrecognized.")
				print("Fatal script error.")
				sys.exit(0)

		else:
			print("Padding is not necessary.")

	def compute_hog(self):
		"""Generate the HOG image and rescale for better viewing."""
		if self.hog_channel_setting != None:
			fd, self.image = hog(self.image, orientations=9, pixels_per_cell=(self.px_per_cell, self.px_per_cell), 
                    cells_per_block=(2, 2), visualize=True, multichannel=self.hog_channel_setting, feature_vector=False)

			self.image = exposure.rescale_intensity(self.image, in_range=(0,10))

			print("HOG generation success")
			print(self.hog_channel_setting)

		else:
			print("HOG generation error: invalid RGB argument.")
			sys.exit(0)