# Ryan Barber
# CISC 442
# PR 2

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy


# A - region based analysis

# Parameters: 
#   left_image
#   right_image
#   window_shape
#   max_disparity
#   rev -  this is used do determine if going from left image to right or right to left
#        rev = False - left to right
#        rev = True  - right to left
def matching(left_image, right_image, window_shape, max_disparity, measure, rev):

    #get window len
    # I always use odd shape
    if(window_shape[0]%2 ==0):
        windowlen = window_shape[0]//2
    else:
        windowlen = (window_shape[0]-1)//2

    #pad images top and bottom and right with 0s
    left = cv.copyMakeBorder(left_image, windowlen, windowlen, windowlen, windowlen, cv.BORDER_CONSTANT,0).astype(np.int32)
    right = cv.copyMakeBorder(right_image, windowlen, windowlen, windowlen, windowlen, cv.BORDER_CONSTANT,0).astype(np.int32)
    #instantiate final image with 0s
    final_image = np.zeros((left.shape),dtype='uint8')

    # for every row in the image not including window
    for i in range(windowlen, len(left_image)):

        #for every col in the image, staring at max_disparity+windowlen because
        # we dont include window and we need to leave room for disparity check
        for j in range(windowlen, len(left_image[0]-windowlen)):

            min_diff = -1
            disparity_value = 0
            diff = 0

            #get left patch = window around pixel at i,j
            left_patch = left[i-windowlen:i+windowlen+1, j-windowlen:j+windowlen+1]

            #start at i,j in right image, then move left by k
            #since camera has moved right, all matching features should be shifted left.
            dis_rng = max_disparity
            if(rev and len(left_image[0])-windowlen-j < max_disparity):
                dis_rng = len(left_image[0])-windowlen-j
            if(j-windowlen < max_disparity):
                dis_rng = j - windowlen
            
            for k in range(dis_rng):

                #get right patch = window around pixel at i,j shifted by k
                if(rev):
                    right_patch = right[i-windowlen:i+windowlen+1, j+k-windowlen:j+k+windowlen+1]
                else:
                    right_patch = right[i-windowlen:i+windowlen+1, j-k-windowlen:j-k+windowlen+1]
                

                #use chosen measure
                if(measure == 'SAD'):
                    diff = np.sum(np.absolute(left_patch-right_patch))
                    #print(diff)
                    #get disparity value where the diff = minimum
                    if(min_diff< 0 or diff < min_diff):
                        min_diff = diff
                        disparity_value = k
                elif(measure == 'SSD'):
                    diff = np.sum((left_patch-right_patch)**2)
                    #get disparity value where the diff = minimum
                    if(min_diff< 0 or diff < min_diff):
                        min_diff = diff
                        disparity_value = k
                elif(measure == 'NCC'):
                    a = np.sum(left_patch*right_patch)
                    b = np.sum(left_patch**2)
                    c = np.sum(right_patch**2)
                    d = b * c
                    e = math.sqrt(d)
                    diff = a/e

                    if(min_diff< 0 or diff > min_diff):
                        min_diff = diff
                        disparity_value = k

            final_image[i][j] = disparity_value


    #final_image = final_image[:,inital_j:final_j]

    return final_image
#=================================================================================

# Combine Left and Right disparity maps 
# Keep same values; 0s where the values are different
# mean blur the 0 pixels
def create_validity_image(left_image, right_image):

    #create validity image
    final_image = deepcopy(left_image)

    for i in range(len(left_image)):
        for j in range(len(left_image[0])):
            if(left_image[i][j] != right_image[i][j-(left_image[i][j])]):
                final_image[i][j] = 0
    
    valid_image = deepcopy(final_image)
    for i in range(2,len(final_image)-2):
        for j in range(2,len(final_image[0])-2):
            if(final_image[i][j] == 0):
                valid_image[i][j] = np.sum(final_image[i-2:i+3, j-2:j+3])/25
    

    return valid_image
#===========================================================================

# matching for Harris images
def matching_harris(left_image, right_image, window_shape, max_disparity, measure):
    imleft_harris = cv.cornerHarris(left_image, 2,3,0.04)
    imright_harris = cv.cornerHarris(right_image, 2,3,0.04)

    corner_thresh_l = imleft_harris.max() * 0.01
    t_l = (imleft_harris > corner_thresh_l) * 1

    corner_thresh_r = imright_harris.max() * 0.01
    t_r = (imright_harris > corner_thresh_r) *1



    #get window len
    # I always use odd shape
    if(window_shape[0]%2 ==0):
        windowlen = window_shape[0]//2
    else:
        windowlen = (window_shape[0]-1)//2

    #pad images top and bottom and right with 0s
    left = cv.copyMakeBorder(t_l, windowlen, windowlen, 0, windowlen, cv.BORDER_CONSTANT,0).astype(np.int32)
    right = cv.copyMakeBorder(t_r, windowlen, windowlen, 0, windowlen, cv.BORDER_CONSTANT,0).astype(np.int32)
    
    #instantiate final image with 0s
    final_image = np.zeros((left.shape),dtype='uint8')

    for i in range(windowlen, len(left)):
        for j in range((windowlen+max_disparity), len(left[0])-windowlen):
            if(left[i][j] != 0):
                min_diff = -1
                disparity_value = 0
                diff = 0

                #get left patch = window around pixel at i,j
                left_patch = left[i-windowlen:i+windowlen+1, j-windowlen:j+windowlen+1]
                for k in range(max_disparity):

                    right_patch = right[i-windowlen:i+windowlen+1, j-k-windowlen:j-k+windowlen+1]

                    #use chosen measure
                    if(measure == 'SAD'):
                        diff = np.sum(np.absolute(left_patch-right_patch))
                        #print(diff)
                        #get disparity value where the diff = minimum
                        if(min_diff< 0 or diff < min_diff):
                            min_diff = diff
                            disparity_value = k
                    elif(measure == 'SSD'):
                        diff = np.sum((left_patch-right_patch)**2)
                        #get disparity value where the diff = minimum
                        if(min_diff< 0 or diff < min_diff):
                            min_diff = diff
                            disparity_value = k
                    elif(measure == 'NCC'):
                        diff = (np.sum(left_patch*right_patch)) / math.sqrt(np.sum((left_patch**2))*(np.sum((right_patch**2))))

                        if(min_diff< 0 or diff > min_diff):
                            min_diff = diff
                            disparity_value = k

                final_image[i][j] = disparity_value

    return final_image


#==========================================

#creating test function - gets results for disparity, harris disparity, valididty check
def Test(left_image_path, right_image_path, window_shape, max_disparity, measure):
    #read images
    imleft = cv.cvtColor(cv.imread(left_image_path), cv.COLOR_BGR2GRAY)
    imright = cv.cvtColor(cv.imread(right_image_path), cv.COLOR_BGR2GRAY)

    # Get depth map
    print('Creating disparity maps...')
    depth_map_left2right = matching(imleft,imright, (window_shape,window_shape), max_disparity, measure, False)
    depth_map_right2left = matching(imright,imleft, (window_shape,window_shape), max_disparity, measure, True)

    #show disparity images
    #fig = plt.figure(figsize=(10,10))
    #plt.subplot(121),plt.imshow(depth_map_left2right, cmap='gray'),plt.title('left')
    #plt.subplot(122),plt.imshow(depth_map_right2left, cmap='gray'),plt.title('right')
    #plt.show()

    #write out disparity images
    plt.imsave('disparity_left2right.png',depth_map_left2right, cmap='gray')
    plt.imsave('disparity_right2left.png',depth_map_right2left, cmap='gray')

    # Get disparity image of Harris corner image
    print('Creating disparity map of harris corner image...')
    disp_harris = matching_harris(imleft, imright, (window_shape,window_shape), max_disparity, measure)

    #show harris disparity image
    #fig = plt.figure(figsize=(10,10))
    #plt.imshow(disp_harris, cmap='gray'),plt.title('disp_harris')
    #plt.show()

    #write out harris disparity image
    plt.imsave('harris_disparity.png',disp_harris,cmap='gray')

    # Validity check image
    print('Creating validity check image...')
    validity = create_validity_image(depth_map_left2right, depth_map_right2left)

    #show validity image
    #fig = plt.figure(figsize=(10,10))
    #plt.imshow(validity, cmap='gray'),plt.title('left')
    #plt.show()

    #write out validity image
    plt.imsave('validity_check.png',validity,cmap='gray')
#==============================================================================================


#================================================================

# GET OUTPUT FOR SUBMISSION

# First image pair (bowling pins)
# params: 
#   window size = 9 x 9
#   max disparity = 20
#   measure = SSD
#Test('./images/image1_left.png', './images/image1_right.png', 9, 20, 'SSD')

# Second image pair (tsokuba)
# params: 
#   window size = 15 x 15
#   max disparity = 30
#   measure = SAD
#Test('./images/image2_left.png', './images/image2_right.png', 15, 30, 'SAD')

# Third image pair (art) - Takes the longest to run (5+ min)
# params: 
#   window size = 9 x 9
#   max disparity = 20
#   measure = NCC
#Test('./images/image3_left.png', './images/image3_right.png', 9, 20, 'NCC')

# Fourth image pair (Moebius)
# params: 
#   window size = 11 x 11
#   max disparity = 25
#   measure = SSD
#Test('./images/image4_left.png', './images/image4_right.png', 11, 25, 'SSD')

# Fifth image pair (Dwarves)
# params: 
#   window size = 11 x 11
#   max disparity = 25
#   measure = SAD
#Test('./images/image5_left.png', './images/image5_right.png', 11, 25, 'SAD')
#=============================================================

# Run

#read images - using tsukuba
imleft = cv.cvtColor(cv.imread('./images/image2_left.png'), cv.COLOR_BGR2GRAY)
imright = cv.cvtColor(cv.imread('./images/image2_right.png'), cv.COLOR_BGR2GRAY)


# SET PARAMETERS

# Shape of window - recommended 7 to 25
window = 15

# Max disparity range - recommended 10 to 30
max_disp_range = 20

# Measure for matching 'SAD' 'SSD' 'NCC'
measure = 'SAD'


# Get depth map
print('Creating disparity maps...')
depth_map_left2right = matching(imleft,imright, (window,window), max_disp_range, measure, False)
depth_map_right2left = matching(imright,imleft, (window,window), max_disp_range, measure, True)

fig = plt.figure(figsize=(10,10))
plt.subplot(121),plt.imshow(depth_map_left2right, cmap='gray'),plt.title('left')
plt.subplot(122),plt.imshow(depth_map_right2left, cmap='gray'),plt.title('right')
plt.show()

# Validity check image
print('Creating validity check image...')
validity = create_validity_image(depth_map_left2right, depth_map_right2left)
fig = plt.figure(figsize=(10,10))
plt.imshow(validity, cmap='gray'),plt.title('left')
plt.show()


# Harris corner detector
print('Creating harris disparity image...')
disp_harris = matching_harris(imleft, imright, (window,window), max_disp_range, measure)
plt.imshow(disp_harris, cmap='gray'),plt.title('disp_harris')
plt.show()

