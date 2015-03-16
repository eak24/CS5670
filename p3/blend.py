import sys
import math

import numpy as np
import cv2

class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position

def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    import inspect
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)

def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors 
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    import inspect
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
    #TODO-BLOCK-END
    # END TODO

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains 
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    import inspect
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
    #TODO-BLOCK-END
    # END TODO
    return img

def blendImages(ipv, blendWidth, is360=False):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1 # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        import inspect
        frameinfo = inspect.getframeinfo(inspect.currentframe())
        print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    translation = np.array([[1, 0, -minX],[0, 1, -minY], [0, 0, 1]])
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

        # First image
        if count == 0:
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_final, y_final = p[:2] / p[2]
     
    compImage = normalizeBlend(acc)
    
    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 11
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    #TODO-BLOCK-BEGIN
    import inspect
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
    #TODO-BLOCK-END
    # END TODO

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(compImage, A, (outputWidth, accHeight),
        flags=cv2.INTER_LINEAR)

    return croppedImage

