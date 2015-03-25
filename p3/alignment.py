import math
import cv2
import numpy as np
import random

eTranslate = 0
eHomography = 1

def computeHomography(f1, f2, matches):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature 
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9 
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)
    
    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        #Fill in first row and repeat
        A[2*i,0] = a_x
        A[2*i,1] = a_y
        A[2*i,2] = 1
        A[2*i,3] = 0
        A[2*i,4] = 0 
        A[2*i,5] = 0 
        A[2*i,6] = -b_x*a_x 
        A[2*i,7] = -b_x*a_y 
        A[2*i,8] = -b_x 

        #Fill in second row
        A[2*i+1,0] = 0
        A[2*i+1,1] = 0 
        A[2*i+1,2] = 0 
        A[2*i+1,3] = a_x 
        A[2*i+1,4] = a_y 
        A[2*i+1,5] = 1 
        A[2*i+1,6] = -b_y*a_x 
        A[2*i+1,7] = -b_y*a_y 
        A[2*i+1,8] = -b_y 

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)
    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    #Fill with eigenvector corresponding to lowest eigenvalue (last row in Vt)
    H=Vt[-1].reshape(3,3)

    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold
       
    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
 	    Choose a minimal set of feature matches.
 	    Estimate the transformation implied by these matches
 	    count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''    

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of 
    #motion models, pure translations (m == eTranslation) and 
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as 
    #the use of RANSAC is almost identical for both cases.
    
    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN

    #Store the indices of the highest scoring homography and the homography itself
    inlier_indices_best = []
    H_best = np.eye(3)
    
    for i in range(nRANSAC):

        #Randomly select the matches and compute homography
        #Translation and rotation requires 2 random matches and Homographies require 4
        if m == eTranslate: 
            sample_size=2
            if len(matches)<sample_size: raise RuntimeError("Not enough matches for translation")
            sample_matches=random.sample(matches,sample_size)
            H_next=leastSquaresFit(f1,f2,sample_matches,m,[0,1])
        else: 
            sample_size=4
            if len(matches)<sample_size: raise RuntimeError("Not enough matches for Homography")
            sample_matches=random.sample(matches,sample_size)         
            H_next=computeHomography(f1,f2,sample_matches)

        #Count the number of inliers for found homography (matches for which
        #delta<=RANSACthresh) and find H with greatest number of inliers.
        inlier_indices_next = getInliers(f1, f2, matches, H_next, RANSACthresh)
        if len(inlier_indices_next)>len(inlier_indices_best):
            inlier_indices_best = inlier_indices_next
            H_best = H_next

    #Compute least-squares motion-estimate using the inliers of the 
    #maximum score homography
    M = leastSquaresFit(f1, f2, matches, m, inlier_indices_best)

    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed 
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt
        p = np.array([[a_x], [a_y], [1]])
        Mp = np.dot(M, p)
        Mp = np.array([Mp[0]/Mp[2], Mp[1]/Mp[2]])
        q = np.array([[b_x], [b_y]])
        distance = np.linalg.norm(Mp-q)
        if distance < RANSACthresh:
            inlier_indices = inlier_indices + [i]
        
        #TODO-BLOCK-END
        #END TODO
        
    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate) 
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a 
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN

            inlier = matches[inlier_indices[i]]
            u = u + f2[inlier.trainIdx].pt[0] - f1[inlier.queryIdx].pt[0]
            v = v + f2[inlier.trainIdx].pt[1] - f1[inlier.queryIdx].pt[1]

            #TODO-BLOCK-END
            #END TODO
    
        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN

        inlier_matches = []
        for i in range(len(inlier_indices)):
            inlier_matches = inlier_matches + [matches[inlier_indices[i]]]

        M = computeHomography(f1 , f2, inlier_matches)

        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

