# Ryan Barber
# CISC 442
# PR1 Part A

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from helpers_partB import *

# Part A - 1
# Write a function to convolve an input image with a kernel to produce a convolved image.

#return float64
def convolve(input_image, kernel_array):
    #get len of kernel array - always N x N where N is odd
    kernel_len = len(kernel_array)

    #pad image with built in opencv function copyMakeBorder()
    pad_size = int((len(kernel_array)-1) / 2)
    padded_image = cv.copyMakeBorder(input_image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_REPLICATE)
    
    #determine how many channels there are
    channels = len(input_image[0][0])

    #create ouput image
    output_image = np.zeros(input_image.shape)

    len_x = len(padded_image[0])

    #loop through the padded image from the first pixel that's not padding
    for i in range(pad_size, (len(padded_image)-pad_size)):
        for j in range(pad_size, (len_x-pad_size)):
            for channel in range(3):
                #output is sum of piece wise product of kernel and relative image cells
                section = kernel_array * padded_image[i-pad_size:i-pad_size+kernel_len, j-pad_size:j-pad_size+kernel_len, channel]
                
                output_image[i-pad_size][j-pad_size][channel] = abs(np.sum(section))

    return output_image
#=============================================================================


# Part A - 2
# Write a function to reduce an image by half its width and height

#return whats given
def reduce(input_image):

    len_in_x = len(input_image[0][0])

    output_image = np.zeros( (len(input_image)//2 , len(input_image[0])//2 , len_in_x) )
    #print(output_image.shape)

    #range for odd num
    if(len(input_image)%2 != 0):
        rangeI = len(input_image)-1
    else:
        rangeI = len(input_image)

    if(len(input_image[0])%2 != 0 ):
        rangeJ = len(input_image[0])-1
    else:
        rangeJ = len(input_image[0])

    #row col for output
    row = 0
    for i in range(rangeI):
        col = 0
        if(i%2 != 0):
            continue

        for j in range(rangeJ):
            if(j % 2 ==0):
                for p in range(len_in_x):
                    output_image[row][col][p] = input_image[i][j][p]
                col = col + 1
                
        row = row + 1
    return output_image

#=====================================================================================

# Part A - 3
# Write a function to expand an input image by twice its width and height

def expand(input_image):

    len_in_x = len(input_image[0][0])

    output_image = np.zeros( (int(len(input_image)*2) , int(len(input_image[0])*2) , len_in_x) )
    
    #loop through all pixels in larger image
    #if even row num and even col num, use value from small image
    #if not even, use top left pixel relative:
    #   1   0    all 0 pixels will be value of the 1 pixel
    #   0   0 

    for i in range(len(output_image)):
        for j in range(len(output_image[0])):
            if(i%2==0 and j%2==0):
                for k in range(len_in_x):
                    output_image[i][j][k] = input_image[(i//2)][(j//2)][k]
            elif(i%2==0 and j%2!=0):
                for k in range(len_in_x):
                    output_image[i][j][k] = input_image[(i//2)][((j-1)//2)][k]
            elif(i%2!=0 and j%2==0):
                for k in range(len_in_x):
                    output_image[i][j][k] = input_image[((i-1)//2)][(j//2)][k]
            else:
                for k in range(len_in_x):
                    output_image[i][j][k] = input_image[((i-1)//2)][((j-1)//2)][k]

    gaussian3 = np.array([[1,2,1,],[2,4,2,],[1,2,1,]])*(1/16)
    meanfilt = np.array([[1,1,1],[1,1,1],[1,1,1]])*(1/9)
    output_image = convolve(output_image, meanfilt)
    return output_image


#=====================================================================================

# Part A - 4
# Use the "Reduce" function to write the GaussianPyramid(i,n) function where n is the no. of levels

#returns float64
def GaussianPyramid(input_image, n):
    #create list to store images in the pyramid
    image_list = [None]*n
    
    #bottom of pyramid is original
    image_list[0] = input_image

    #using Gaussian kernel from slides
    gaussian3 = np.array([[1,2,1,],[2,4,2,],[1,2,1,]])*(1/16)
    meanfilt = np.array([[1,1,1],[1,1,1],[1,1,1]])*(1/9)

    #loop n times bluring and reducing, storing image in list
    for i in range(1,n):
        next_image = convolve(image_list[i-1], meanfilt)
        image_list[i] = reduce(next_image)


    return image_list
#=============================================================================================

# Part A - 5
# Use the above functions to write LaplacianPyramids(i,n) that producesn level 
# Laplacian pyramid of i. 

def LaplacianPyramid(input_image, n):
    #create list to store images in the pyramid
    image_list = [None]*n

    #create gaussian pyramid
    gaussian_pyr = GaussianPyramid(input_image, n) # come out as float64
    for i in range(len(gaussian_pyr)):
        gaussian_pyr[i] = gaussian_pyr[i]

    #top of pyramid is last element of gaussian
    image_list[-1] = gaussian_pyr[-1]

    #loop, expand, subtract gaussian[i] - expanded = laplacian[i]
    for i in range(n-1):
        expanded = expand(gaussian_pyr[n-1-i]) # uint8

        #if expanded is != shape of gaussian (becuase of rounding down)
        #   copy last row or col
        while(gaussian_pyr[n-2-i].shape != expanded.shape):
            if(gaussian_pyr[n-2-i].shape[0] != expanded.shape[0]):
                expanded = np.vstack((expanded,np.array([expanded[-1,:]])))
            if(gaussian_pyr[n-2-i].shape[1] != expanded.shape[1]):
                l = len(expanded)
                expanded = np.hstack((expanded, np.array([expanded[:,-1]]).reshape((l,1,3))))

        diff = gaussian_pyr[n-2-i] - expanded
        image_list[n-2-i] = diff

    return image_list
#===========================================================================

# Part A - 6
# Write the Reconstruct(Li, n) function which collapses the laplacian pyramid Li of n
# levels to generate the original image. Report error using image difference

def Reconstruct(pyramid, n):
    #Expand element then add next element
    final_image = pyramid[-1]
    for i in range(n-1):
        final_image = expand(final_image)
        while(pyramid[n-2-i].shape != final_image.shape):
            if(pyramid[n-2-i].shape[0] != final_image.shape[0]):
            # rows
                final_image = np.vstack((final_image,np.array([final_image[-1]])))
            if(pyramid[n-2-i].shape[1] != final_image.shape[1]):
            #cols
                b = np.array(final_image[:,-1]).reshape((len(final_image),1,3))
                final_image = np.hstack((final_image, b))

        #fig = plt.figure(figsize=(15,15))
        #plt.subplot(121), plt.imshow(pyramid[n-2-i].astype(np.int64)[:,:,::-1]), plt.title('pyr')
        #plt.subplot(122), plt.imshow(final_image.astype(np.int64)[:,:,::-1]), plt.title('final')
        #plt.show()
        final_image = pyramid[n-2-i] + final_image

    return final_image
#==============================================================================

# Test Blocks for Part 1 - 6

#input image
image_earth = cv.imread('./images/earth.jpg')
image_lena = cv.imread('./images/lena.png')


#gaussian bluring
gaussian3 = np.array([[1,2,1,],[2,4,2,],[1,2,1,]])*(1/16)
meanfilt = np.array([[1,1,1],[1,1,1],[1,1,1]])*(1/9)



# TESTS

# Part A - 1
#convolved_image = convolve(image_lena, gaussian3).astype(np.uint8)
#fig = plt.figure(figsize=(15,15))
#plt.subplot(121), plt.imshow(image_lena[:,:,::-1]), plt.title('Original')
#plt.subplot(122), plt.imshow(convolved_image[:,:,::-1]), plt.title('Convoloved')
#plt.show()

# Part A - 2
#small = reduce(image_lena).astype(np.uint8)
#fig = plt.figure(figsize=(15,15))
#plt.subplot(121), plt.imshow(image_lena[:,:,::-1]), plt.title('Original')
#plt.subplot(122), plt.imshow(small[:,:,::-1]), plt.title('Reduced')
#plt.show()

# Part A - 3
#large = expand(image_earth).astype(np.uint8)
#fig = plt.figure(figsize=(15,15))
#plt.subplot(121), plt.imshow(image_earth[:,:,::-1]), plt.title('Original')
#plt.subplot(122), plt.imshow(large[:,:,::-1]), plt.title('Expanded')
#plt.show()

# Part A - 4
#first image is the original
#pyramid = GaussianPyramid(image_earth, 5)
#o = 1
#fig = plt.figure(figsize=(15,15))
#for i in pyramid:
#    i = i.astype(np.uint8)
#    plt.subplot(1, len(pyramid), o), plt.imshow(i[:,:,::-1]), plt.title('Image' + str(o))
#    o += 1
#plt.show()

# Part A - 5
#laplacian_pyr = LaplacianPyramid(image_earth, 5)
#o = 1
#fig = plt.figure(figsize=(15,15))
#for i in laplacian_pyr:
#    plt.subplot(1, len(laplacian_pyr), o), plt.imshow(i.astype(np.uint8)[:,:,::-1]), plt.title('Image' + str(o))
#    o += 1
#plt.show()

# Part A - 6 - reconstruction (uncomment for submit)
print('Doing Reconstruct() test\n')
laplacian_pyr = LaplacianPyramid(image_earth, 5)
recon = Reconstruct(laplacian_pyr, 5).astype(np.uint8)
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(image_earth[:,:,::-1]), plt.title('Original')
plt.subplot(122), plt.imshow(recon[:,:,::-1]), plt.title('Reconstructed')
plt.show()

#=======================================================
#=======================================================
#=======================================================

# Blending Images (part 7)

# BLENDING FUNCTION
def blend(image1, image2, midpoint):
    #num of levels
    n=5
    
    #give only part image2 we want (from click to the right)
    image2 = image2[:,midpoint:]

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    #fig = plt.figure(figsize=(15,15))
    #plt.imshow(image1.astype(np.int64)[:,:,::-1])
    #plt.show()
    #fig = plt.figure(figsize=(15,15))
    #plt.imshow(image2.astype(np.int64)[:,:,::-1])
    #plt.show()

    # Show Not Blended (just put together)
    #im3 = np.hstack((image1,image2))
    #fig = plt.figure(figsize=(15,15))
    #plt.imshow(im3.astype(np.int64)[:,:,::-1])
    #plt.show()

    #create laplacian pyramid of image1
    im1_lap_pyr = LaplacianPyramid(image1,n)

    # Display laplacian pyramid image 1
    #o = 1
    #fig = plt.figure(figsize=(15,15))
    #for i in im1_lap_pyr:
    #    plt.subplot(1, len(im1_lap_pyr), o), plt.imshow(i.astype(np.uint8)[:,:,::-1]), plt.title('Image' + str(o))
    #    o += 1
    #plt.show()

    print('image 1 pyramid made')

    #create laplacian pyramid of image2
    im2_lap_pyr = LaplacianPyramid(image2,n)

    # Display laplacian pyramid image 2
    #o = 1
    #fig = plt.figure(figsize=(15,15))
    #for i in im2_lap_pyr:
    #    plt.subplot(1, len(im2_lap_pyr), o), plt.imshow(i.astype(np.uint8)[:,:,::-1]), plt.title('Image' + str(o))
    #    o += 1
    #plt.show()

    print('image 2 pyramid made')


    #create output pyramid
    lout = [None] * n
    for i in range(n):
        # I could not figure out how to use explicit bitmask with 1 and 0 so
        # I just put the parts of the laplacian pyramids together manually
        # using numpy.hstack(image1, image2)

        #combine laplacian pyramids of each
        bit = np.zeros((len(im1_lap_pyr[n-1-i]), len(im1_lap_pyr[n-1-i][0])+len(im2_lap_pyr[n-1-i][0]), 3))
        bit = np.hstack((im1_lap_pyr[n-1-i], im2_lap_pyr[n-1-i]))

        lout[n-1-i] = bit
    
    return lout
#==================================================================================



# Affine Unwarping + blend
# Uncomment for submission
# SLOW, FEW MINUTES

im1 = cv.imread('./images/image1_1.JPG')
im1 = im1[:, :-1] # -1 becuase there is a white border on the edge
im2 = cv.imread('./images/image1_2.JPG')

#affine homography 
n=3
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Pick 3 Points')
plt.subplot(122), plt.imshow(im2[:,:,::-1]), plt.title('Pick 3 Corresponding Points (in same order)')
tup = plt.ginput(2*n,timeout=0,show_clicks=True) #unlimited time to pick
plt.close()

original_points = np.zeros((n,2),dtype="float32")
corresponding_points = np.zeros((n,2),dtype='float32')
for i in range(len(tup)):
    if(i < n):
        for j in range(len(original_points[i])):
            original_points[i][j] = round(tup[i][j])
    else:
        for j in range(len(corresponding_points[i-n])):
            corresponding_points[i-n][j] = round(tup[i][j])


affine_mat = affine_homography(original_points, corresponding_points)
affine_mat = np.vstack((affine_mat, np.array([0,0,1])))
print('affine matrix\n', affine_mat)
affine_mat = np.linalg.inv(affine_mat)
print('inverse affine matrix\n', affine_mat)

#copied from Rohit presentation
result = cv.warpAffine(im2, affine_mat[:2,:], (im1.shape[1] + im2.shape[1], im1.shape[0]))
#result[:, :im2.shape[1]] = im1


fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Left')
plt.subplot(122), plt.imshow(result[:,:,::-1]), plt.title('Pick point that corresponds to the right edge of the left image')
tup = plt.ginput(1,timeout=0,show_clicks=True)
plt.close()

# tup give as [ x , y ]

midpoint = int(round(tup[0][0]))

print('Blending Images...')
output_pyr = blend(im1, result, midpoint)

out_img = Reconstruct(output_pyr, len(output_pyr)).astype(np.int64)

fig = plt.figure(figsize=(15,15))
plt.imshow(out_img[:,:,::-1]), plt.title('Final')
plt.show()

#write out image
#cv.imwrite('./outputimages/image1_12.JPG',out_img)


#===================================================================

"""

# Perspective unwarping and Blend
# SLOW, FEW MINUTES

im1 = cv.imread('./images/image2_1.jpg')
im2 = cv.imread('./images/image2_2.jpg')

#perspective homography using overconstrained 7 points
n=7
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Pick 7 Points')
plt.subplot(122), plt.imshow(im2[:,:,::-1]), plt.title('Pick 7 Corresponding Points (in same order)')
tup = plt.ginput(2*n,timeout=0,show_clicks=True) #unlimited time to pick
plt.close()

original_points = np.zeros((n,2),dtype="float32")
corresponding_points = np.zeros((n,2),dtype='float32')
for i in range(len(tup)):
    if(i < n):
        for j in range(len(original_points[i])):
            original_points[i][j] = round(tup[i][j])
    else:
        for j in range(len(corresponding_points[i-n])):
            corresponding_points[i-n][j] = round(tup[i][j])


perspective_mat = perspective_homography(original_points, corresponding_points)
print('perspective matrix\n', perspective_mat)
perspective_mat = np.linalg.inv(perspective_mat)
print('inverse perspective matrix\n', perspective_mat)

#copied from Rohit presentation
result = cv.warpPerspective(im2, perspective_mat, (im1.shape[1] + im2.shape[1], im1.shape[0]))
#result[:, :im2.shape[1]] = im1


fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Left')
plt.subplot(122), plt.imshow(result[:,:,::-1]), plt.title('Pick point that corresponds to the right edge of the left image')
tup = plt.ginput(1,timeout=0,show_clicks=True)
plt.close()

# tup give as [ x , y ]

midpoint = int(round(tup[0][0]))

print('Blending Images...')
output_pyr = blend(im1, result, midpoint)

out_img = Reconstruct(output_pyr, len(output_pyr)).astype(np.int64)

fig = plt.figure(figsize=(15,15))
plt.imshow(out_img[:,:,::-1]), plt.title('Final')
plt.show()

#write out image
#cv.imwrite('./outputimages/image2_12.jpg',out_img)

"""

#============================================================================

"""

# No unwarping Blending 1
# VERY SLOW SEVERAL MINUTES

im1 = cv.imread('./images/image3_1.png')
im2 = cv.imread('./images/image3_2.png')


# Choose midpoint 
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Left')
plt.subplot(122), plt.imshow(im2[:,:,::-1]), plt.title('Pick point that corresponds to the right edge of the left image')
tup = plt.ginput(1,timeout=0,show_clicks=True)
plt.close()

# tup give as [ x , y ]

midpoint = int(round(tup[0][0]))

print('Blending Images...')
output_pyr = blend(im1, im2, midpoint)

out_img = Reconstruct(output_pyr, len(output_pyr)).astype(np.int64)

fig = plt.figure(figsize=(15,15))
plt.imshow(out_img[:,:,::-1]), plt.title('Final')
plt.show()

#write out image
cv.imwrite('./outputimages/image3_12.png',out_img)

"""

#=============================================================

"""

# No unwarping Blending 2 
# SLOWEST, MANY MINUTES ~5

im1 = cv.imread('./images/image4_1.png')
im2 = cv.imread('./images/image4_2.png')


# Choose midpoint 
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(im1[:,:,::-1]), plt.title('Left')
plt.subplot(122), plt.imshow(im2[:,:,::-1]), plt.title('Pick point that corresponds to the right edge of the left image')
tup = plt.ginput(1,timeout=0,show_clicks=True)
plt.close()

# tup give as [ x , y ]

midpoint = int(round(tup[0][0]))

print('Blending Images...')
output_pyr = blend(im1, im2, midpoint)

out_img = Reconstruct(output_pyr, len(output_pyr)).astype(np.int64)


fig = plt.figure(figsize=(15,15))
plt.imshow(out_img[:,:,::-1]), plt.title('Final')
plt.show()

#write out image
#cv.imwrite('./outputimages/image4_12.png', out_img)

"""

#===============================================