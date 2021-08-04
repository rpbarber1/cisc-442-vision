
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from random import randint
from copy import deepcopy

# point selection function
# display image, get user clicks, return points array(s)
def getPoints(original, warped, n, type):
    if(type == 'p'):
        fig = plt.figure(figsize=(20,20))
        plt.subplot(121), plt.imshow(original[:,:,::-1]), plt.title('Select %i Points for Correspondence - PERSPECTIVE' %n)
        plt.subplot(122), plt.imshow(warped[:,:,::-1]), plt.title('Select the %i Corresponding Points - PERSPECTIVE' %n)
        tup = plt.ginput(2*n,timeout=0,show_clicks=True)
        plt.close()
    elif(type == 'a'):
        fig = plt.figure(figsize=(15,15))
        plt.subplot(121), plt.imshow(original[:,:,::-1]), plt.title('Select %i Points for Correspondence - AFFINE' %n)
        plt.subplot(122), plt.imshow(warped[:,:,::-1]), plt.title('Select the %i Corresponding Points - AFFINE' %n)
        tup = plt.ginput(2*n,timeout=0,show_clicks=True)
        plt.close()

    # points given [ x,y ]

    # GET RANDOM SHIFTED POINTS
    original_points = np.zeros((n,2),dtype="float32")
    warped_points = np.zeros((n,2),dtype='float32')
    for i in range(len(tup)):
        if(i < n):
            for j in range(len(original_points[i])):
                original_points[i][j] = round(tup[i][j])
        else:
            for j in range(len(warped_points[i-n])):
                warped_points[i-n][j] = round(tup[i][j])

    return original_points, warped_points
#============================================================================



#functions to find homography matrix

# Perspective homography Using SVD - use on both correct number of points and over constrained
def perspective_homography(original_points, shifted_points):
    # Using notation A*h = b

    # set up matrix A
    A = np.zeros((2*len(original_points),9), dtype="float32")
    for i in range(len(original_points)):
        A[i*2] = np.array([original_points[i][0],original_points[i][1],1,0,0,0,(-original_points[i][0]*shifted_points[i][0]),
                            (-original_points[i][1]*shifted_points[i][0]), (-shifted_points[i][0])])
        A[(i*2)+1] = np.array([0,0,0, original_points[i][0], original_points[i][1],1, (-original_points[i][0]*shifted_points[i][1]),
                            (-original_points[i][1]*shifted_points[i][1]), (-shifted_points[i][1])])

    #transpose A
    #At = np.transpose(A)

    # product for svd = At * A 
    #product = np.matmul(At,A)

    # SVD function  svd(At * A)
    U, S, V = np.linalg.svd(A)

    #get homography matrix; last row of V
    hom_mat = V[-1]
    hom_mat /= hom_mat[-1]
    hom_mat = hom_mat.reshape((3,3))


    return hom_mat
#================================================================================


# Affine homography Using Least Squares - for both correct number of points and over constrained
def affine_homography(original_points, shifted_points):
    # Using notation A*h = b

    # set up matrix A
    A = np.zeros((2*len(original_points),6), dtype="float32")
    for i in range(len(original_points)):
        A[i*2] = np.array([original_points[i][0],original_points[i][1],1,0,0,0])
        A[(i*2)+1] = np.array([0,0,0, original_points[i][0], original_points[i][1],1])

    #set up matrix b
    b = np.zeros((2*len(shifted_points),1), dtype="float32")
    for i in range(len(shifted_points)):
        b[i*2] = shifted_points[i][0]
        b[(i*2)+1] = shifted_points[i][1]

    hom_mat = np.linalg.lstsq(A,b, rcond=None)[0]

    hom_mat = hom_mat.reshape(2,3)

    return hom_mat
#=================================================================================