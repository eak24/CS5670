import os
import cv2
import numpy as np

def warpLocal(src, uv):
    '''
    Input:
        src --    source image in a numpy array with values in [0, 255]. 
                  The dimensions are (rows, cols, color bands BGR).
        uv --     warped image in terms of addresses of each pixel in the source 
                  image in a numpy array.
                  The dimensions are (rows, cols, addresses of pixels [:,:,0] 
                  are x (i.e., cols) and [:,,:,1] are y (i.e., rows)).
    Output:
        warped -- resampled image from the source image according to provided 
                  addresses in a numpy array with values in [0, 255]. The 
                  dimensions are (rows, cols, color bands BGR).
    '''
    warped = cv2.remap(src, uv[:, :, 0].astype(np.float32),\
             uv[:, :, 1].astype(np.float32), cv2.INTER_CUBIC)
    return warped


def computeSphericalWarpMappings(dstShape, f, k1, k2, R):
    '''
    Compute the spherical warp. Compute the addresses of each pixel of the 
    output image in the source image.

    Input:
        dstShape -- shape of input / output image in a numpy array. 
                    [number or rows, number of cols, number of bands]
        f --        focal length in pixel as int
                    See assignment description on how to find the focal length
        k1 --       horizontal distortion as a float
        k2 --       vertical distortion as a float
        R --        rotational transformation matrix as a 3x3 numpy array
    Output:
        uvImg --    warped image in terms of addresses of each pixel in the 
                    source image in a numpy array.
                    The dimensions are (rows, cols, addresses of pixels 
                    [:,:,0] are x (i.e., cols) and [:,,:,1] are y (i.e., rows)).
    '''

    # calculate minimum y value
    vec = np.zeros(3)
    vec[0] = np.sin(0.0) * np.cos(0.0)
    vec[1] = np.sin(0.0)
    vec[2] = np.cos(0.0) * np.cos(0.0)
    vec = R.dot(vec)
    min_y = vec[1]

    # calculate spherical coordinates
    # (x,y) is the spherical image coordinates.
    # (xf,yf) is the spherical coordinates, e.g., xf is the angle theta
    # and yf is the angle phi
    one = np.ones((dstShape[0],dstShape[1]))
    xf = one * np.arange(dstShape[1])
    yf = one.T * np.arange(dstShape[0])
    yf = yf.T

    xf = ((xf - 0.5 * dstShape[1]) / f)
    yf = ((yf - 0.5 * dstShape[0]) / f - min_y)
    # BEGIN TODO 1
    # add code to apply the spherical correction, i.e.,
    # compute the Euclidean coordinates,
    # and rotate according to r,
    # and project the point to the z=1 plane at (xt/zt,yt/zt,1),
    # then distort with radial distortion coefficients k1 and k2
    # Use xf, yf as input for your code block and compute xt, yt
    # as output for your code. They should all have the shape 
    # (img_height, img_width)
    # TODO-BLOCK-BEGIN

    #>>>COMPUTE THE EUCLIDEAN COORDINATES<<<<
    points = np.ones((3,dstShape[0]*dstShape[1]))
    raise "TSears error: Should xhat be sin(theta)cos(phi) or sin(phi)??"
    points[0] = (np.sin(xf)*np.cos(yf)).flatten() #xhat
    points[1] = np.sin(yf).flatten() #yhat
    points[2] = (np.cos(xf)* np.cos(yf)).flatten() #zhat
    ##
    
    #>>>ROTATE ACCORDING TO R<<
    points = np.dot(R,points)
    #

    #>>>PROJECT to the Z=1 plane#
    points /= points[2]

    xf = points[0].reshape(xf.shape)
    yf = points[1].reshape(yf.shape)

    gamma_2 = xf*xf + yf*yf
    gamma_4 = gamma_2 *gamma_2
    xt = xf*(one + k1*gamma_2 +k2*gamma_4)
    yt = yf*(one + k1*gamma_2 +k2*gamma_4)

    #import inspect
    #frameinfo = inspect.getframeinfo(inspect.currentframe())
    #print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
    # TODO-BLOCK-END
    # END TODO
    # Convert back to regular pixel coordinates
    xn = 0.5 * dstShape[1] + xt * f
    yn = 0.5 * dstShape[0] + yt * f
    uvImg = np.dstack((xn,yn))
    return uvImg


def getRotationMatrix(theta):
    '''
    Input:
        theta -- rotational angle
    Output:
        R -- 3x3 transformation matrix in a numpy array
    '''
    R = np.zeros((3, 3))
    R[0,0] = np.cos(theta)
    R[0,1] = np.sin(theta)
    R[0,2] = 0.0
    R[1,0] = -np.sin(theta)
    R[1,1] = np.cos(theta)
    R[1,2] = 0.0
    R[2,0] = 0.0
    R[2,1] = 0.0
    R[2,2] = 1.0
    return R


def warpSpherical(image, focalLength, k1=-0.15, k2=0.001, theta=0):
    '''
    Input:
        image --       filename of input image as string
        focalLength -- focal length in pixel as int
                       see assignment description on how to find the focal 
                       length
        k1, k2 --      Radial distortion parameters
        theta --       Rotation angle in radians
    Output:
        dstImage --    output image in a numpy array with
                       values in [0, 255]. The dimensions are (rows, cols, 
                       color bands BGR).
    '''
    # set camera parameters
    R = getRotationMatrix(theta)

    # compute spherical warp
    # compute the addresses of each pixel of the output image in the 
    # source image
    uv = computeSphericalWarpMappings(np.array(image.shape), focalLength, k1, \
        k2, R)

    # warp image based on backwards coordinates
    return warpLocal(image, uv)


