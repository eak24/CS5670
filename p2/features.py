import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                            each pixel
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris score for 'srcImage' at each pixel
        # and store in 'harrisImage'.  See the project page for
        # pointers on how to do this. You should also store an
        # orientation for each pixel in 'orientationImage.'
        # You may need to compute a few filtered images to start with
        #TODO-BLOCK-BEGIN
        import inspect
        frameinfo = inspect.getframeinfo(inspect.currentframe())
        print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
        #TODO-BLOCK-END

        return harrisImage, orientationImage

    def computeLocalMaxima(self, srcImage):
        '''
        Input:
            srcImage -- numpy array containing the Harris score at
                            each pixel.
        Output:
            destImage -- numpy array containing True/False at
                            each pixel, depending on whether
                            the pixel is a local maxima in 7x7 around it
        '''
        destImage = np.zeros_like(srcImage, np.bool)

        # TODO 2: Compute the local maxima image
        #TODO-BLOCK-BEGIN
        import inspect
        frameinfo = inspect.getframeinfo(inspect.currentframe())
        print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
        #TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You'll need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You'll need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                #TODO 3: Fill in feature with location and orientation data here
                #TODO-BLOCK-BEGIN
                import inspect
                frameinfo = inspect.getframeinfo(inspect.currentframe())
                print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
                #TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        image = image.astype(np.float32)
        image /= 255.
        height,width = image.scale
        image = np.vstack([image[1].reshape(1,width) ,
            image[0].reshape(1,width),
            image,
            image[-1].reshape(1,width),
            image[-2].reshape(1,width)]
            )
        image = np.hstack([image[:,1].reshape(height+4,1) ,
            image[:,0].reshape(height+4,1) ,
            image,
            image[:,-1].reshape(height+4,1) ,
            image[:,-2].reshape(height+4,1)])

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x+=2
            y+=2
            desc[i]=image[y-2:y+2,x-2:x+2].flatten()
            # TODO 4:
            # The descriptor is a 5x5 window of intensities sampled centered on
            # the feature point.
            #TODO-BLOCK-BEGIN
            import inspect
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
            #TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You'll need to compute the transform from
            # each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels
            # in the 8x8 feature descriptor image
            transMx = np.zeros((2, 3))

            #TODO-BLOCK-BEGIN
            import inspect
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
            #TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: fill in the feature descriptor desc for a MOPS descriptor
            #TODO-BLOCK-BEGIN
            import inspect
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
            #TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]


        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        switched = False
        if desc1.shape[0]>desc2.shape[0]:
            switched=True
            x=desc1.copy()
            desc1=desc2.copy()
            desc2 = x
            del x


        # TODO 7: Perform simple feature matching.  This just uses the SSD
        # distance between two feature vectors, and matches a feature in the
        # first image with the closest feature in the second image.  It can
        # match multiple features in the first image to the same feature in
        # the second image.
        #TODO-BLOCK-BEGIN
        #if desc1.shape[0]>desc2.shape[0]: #I want desc1 to always have the least elements
        #    return self.matchFeatures(desc2,desc1)
        d_matrix = spatial.distance.cdist(desc1,desc2)
        print d_matrix # d_matrix[i][j]=distnace between f1 and f

        matches=[]
        for hubby in xrange(desc1.shape[0]):
            best = int(d_matrix[hubby].argmin())
            #np.save('d_matrix',d_matrix)
            #print hubby, best, type(hubby),type(best)
            if not switched:
                matches.append(cv2.DMatch(
                _queryIdx= best,
                _trainIdx = hubby,
                _distance = d_matrix[hubby,best] 
                ))
            else:
                matches.append(cv2.DMatch(
                _trainIdx= hubby,
                _queryIdx = best,
                _distance = d_matrix[hubby,best]
                ))

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        switched = False
        if desc1.shape[0]>desc2.shape[0]:
            switched=True
            x=desc1.copy()
            desc1=desc2.copy()
            desc2 = x
            del x


        # TODO 7: Perform simple feature matching.  This just uses the SSD
        # distance between two feature vectors, and matches a feature in the
        # first image with the closest feature in the second image.  It can
        # match multiple features in the first image to the same feature in
        # the second image.
        #TODO-BLOCK-BEGIN
        #if desc1.shape[0]>desc2.shape[0]: #I want desc1 to always have the least elements
        #    return self.matchFeatures(desc2,desc1)
        d_matrix = spatial.distance.cdist(desc1,desc2)
        print d_matrix # d_matrix[i][j]=distnace between f1 and f

        matches=[]
        for hubby in xrange(desc1.shape[0]):
            best,secondbest = d_matrix[hubby].argpartition(2)[:2] #find the top two 
            #np.save('d_matrix',d_matrix)
            #print hubby, best, type(hubby),type(best)
            if not switched:
                matches.append(cv2.DMatch(
                _queryIdx= best,
                _trainIdx = hubby,
                _distance = d_matrix[hubby,best] /d_matrix[hubby,secondbest] 
                ))
            else:
                matches.append(cv2.DMatch(
                _trainIdx= hubby,
                _queryIdx = best,
                _distance = d_matrix[hubby,best] /d_matrix[hubby,secondbest] 
                ))

        return matches

        # TODO 8: Perform ratio feature matching.
        # This just uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # It can match multiple features in the first image to the same feature in
        # the second image.  (See class notes for more information)
        # You don't need to threshold matches in this function
        #TODO-BLOCK-BEGIN
        import inspect
        frameinfo = inspect.getframeinfo(inspect.currentframe())
        print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
        #TODO-BLOCK-END

        return matches

