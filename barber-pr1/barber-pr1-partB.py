# Ryan Barber
# CISC 442
# PR1 Part B

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from random import randint
from copy import deepcopy
from helpers_partB import *


# Part B - 1
# Use built in functions to get homography matricies

image_tajmahal = cv.imread('./images/tajmahal.jpg')

# RANDOM POINTS - (from Rohit presentation)
perspective_points = np.float32([[100,250],[340,570],[480,600],[20,10]])
perspective_shift = np.float32([[125,260],[380,600],[500,620],[25,25]])
affine_points = np.float32([[100,250],[340,570],[480,600]])
affine_shift = np.float32([[110,270],[360,590],[540,610]])


# USE BUILT IN FUNCTIONS FOR HOMOGRAPHY MATRIX
perspective_hom = cv.getPerspectiveTransform(perspective_points, perspective_shift)
affine_hom = cv.getAffineTransform(affine_points, affine_shift)

print('built-in cv2.cv.getPerspectiveTransform()\n',perspective_hom)
print('Affine matrix with built-in cv2.cv.getAffineTransform()\n', affine_hom)


# GET IMAGES
perspective_image = cv.warpPerspective(image_tajmahal, perspective_hom, (image_tajmahal.shape[1], image_tajmahal.shape[0]))
affine_image = cv.warpAffine(image_tajmahal, affine_hom, (image_tajmahal.shape[1], image_tajmahal.shape[0]))


# DISPLAY ORIGINAL VS PERSPECTIVE MADE WITH BUILT IN FUNCTIONS
#fig = plt.figure(figsize=(15,15))
#plt.subplot(121), plt.imshow(image_tajmahal[:,:,::-1]), plt.title('Original')
#plt.subplot(122), plt.imshow(perspective_image[:,:,::-1]), plt.title('Perspective Transform With Built-In Functions')
#plt.show()

#fig = plt.figure(figsize=(15,15))
#plt.subplot(121), plt.imshow(image_tajmahal[:,:,::-1]), plt.title('Original Transform')
#plt.subplot(122), plt.imshow(affine_image[:,:,::-1]), plt.title('Affine Transform With Built-In Functions')
#plt.show()

#=====================================================================================================

# Part B - 2
# Compute homography matrix manually

# GET CORRESPONDING POINTS FOR PERSPECTIVE
# GET POINTS FROM ORIGINAL IMAGE & WARPED IMAGE
original_points, corresponding_points = getPoints(image_tajmahal, perspective_image, 4, 'p')


# PERSPECTIVE HOMOGRAPY CALCULATION (ph_mat)
ph_mat = perspective_homography(original_points, corresponding_points)

print('Perspective matrix with manual homography calculation\n', ph_mat)


# GET ERROR
error_ph_mat = np.sum(np.abs(np.subtract(perspective_hom,ph_mat,dtype=np.float)))

# GET IMAGE perspective
ph_image = cv.warpPerspective(image_tajmahal, ph_mat, (image_tajmahal.shape[1], image_tajmahal.shape[0]))


# GET CORRESPONDING POINTS FOR PERSPECTIVE over constrained
# GET POINTS FROM ORIGINAL IMAGE & WARPED IMAGE
original_points, corresponding_points = getPoints(image_tajmahal, perspective_image, 7, 'p')


# PERSPECTIVE HOMOGRAPHY CALCULATION over constrained
pho_mat = perspective_homography(original_points, corresponding_points)
print('Perspective matrix (over constrained) with manual homography calculation\n', pho_mat)

# GET ERROR
error_pho_mat = np.sum(np.abs(np.subtract(perspective_hom,pho_mat,dtype=np.float)))

# GET IMAGE
pho_image = cv.warpPerspective(image_tajmahal, pho_mat, (image_tajmahal.shape[1], image_tajmahal.shape[0]))

# Perspective 
# SHOW 4 IMAGES: ORIGINAL, TRANSFORMED WITH BUILT-IN, 
#                TRANSFORMED with SVD , TRANSFORMED SVD over constrained
fig = plt.figure(figsize=(15,15))
plt.subplot(221), plt.imshow(image_tajmahal[:,:,::-1]), plt.title('Original')
plt.subplot(222), plt.imshow(perspective_image[:,:,::-1]), plt.title('Perspective Transform with Built-In')
plt.subplot(223), plt.imshow(ph_image[:,:,::-1]), plt.title('Perspective Transform with 4 points')
plt.subplot(224), plt.imshow(pho_image[:,:,::-1]), plt.title('Perspective Transform with 7 points over constrained')
plt.show()

print('Perspective error of 4 Point\n', error_ph_mat)
print('Perspective error of 7 Point over constrained\n',error_pho_mat)


#==================================================================================

# GET CORRESPONDING POINTS FOR AFFINE
# GET POINTS FROM ORIGINAL IMAGE & WARPED IMAGE
original_points, corresponding_points = getPoints(image_tajmahal, affine_image, 3, 'a')

# AFFINE HOMOGRAPY CALCULATION (ah_mat)
ah_mat = affine_homography(original_points, corresponding_points)
print('Affine matrix with manual homography calculation\n', ah_mat)

# GET ERROR
error_ah_mat = np.sum(np.abs(np.subtract(affine_hom, ah_mat,dtype=np.float)))

# GET IMAGE
ah_image = cv.warpAffine(image_tajmahal, ah_mat, (image_tajmahal.shape[1], image_tajmahal.shape[0]))


# GET CORRESPONDING POINTS FOR AFFINE over constrained
# GET POINTS FROM ORIGINAL IMAGE & WARPED IMAGE
original_points, corresponding_points = getPoints(image_tajmahal, affine_image, 4, 'a')


# HOMOGRAPHY CALCULATION
aho_mat = affine_homography(original_points, corresponding_points)
print('Affine matrix (over constrained) with manual homography calculation\n', aho_mat)

# GET ERROR
error_aho_mat = np.sum(np.abs(np.subtract(affine_hom, aho_mat,dtype=np.float)))

# GET IMAGE
aho_image = cv.warpAffine(image_tajmahal, aho_mat, (image_tajmahal.shape[1], image_tajmahal.shape[0]))


fig = plt.figure(figsize=(15,15))
plt.subplot(221), plt.imshow(image_tajmahal[:,:,::-1]), plt.title('Original')
plt.subplot(222), plt.imshow(affine_image[:,:,::-1]), plt.title('Affine Transform with Built-In')
plt.subplot(223), plt.imshow(ah_image[:,:,::-1]), plt.title('Affine Transform 3 points')
plt.subplot(224), plt.imshow(aho_image[:,:,::-1]), plt.title('Affine Transform 4 points over constrained')
plt.show()

print('Affine error of 3 Point\n', error_ah_mat)
print('Affine error of 4 Point over constrained\n',error_aho_mat)

#==============================

