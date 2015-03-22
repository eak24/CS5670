import Tkinter as tk
import tkFileDialog
import tkMessageBox
import ttk
import os
import math
import json
import argparse
import numpy as np
import numpy.linalg
import cv2
from PIL import Image, ImageTk, ImageDraw

import uiutils
import warp
import alignment
import blend

DEFAULT_FOCAL_LENGTH = 678
DEFAULT_K1 = -0.21
DEFAULT_K2 = 0.26

def parse_args():
  parser = argparse.ArgumentParser(description="Panorama Maker")
  parser.add_argument("--extra-credit", dest="ec", action='store_true',
      help="Flag to toggle extra credit features")
  return parser.parse_args()


class AutostitchUIFrame(tk.Frame):
    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.notebook.add(HomographyVisualizationFrame(self.notebook, root),
            text='Homography')
        self.notebook.add(SphericalWarpFrame(self.notebook, root),
            text='Spherical Warp')
        self.notebook.add(AlignmentFrame(self.notebook, root),
            text='Alignment')
        self.notebook.add(PanoramaFrame(self.notebook, root),
            text='Panorama')
        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
   
    def updateUI(self):
        self.parent.update()
   
class BaseFrame(tk.Frame):
    '''The base frame shared by all the tabs in the UI.'''

    def __init__(self, parent, root, nrows, ncolumns):
        assert nrows >= 2 and ncolumns >= 1

        tk.Frame.__init__(self, parent)
        self.root = root
        self.grid(row=0, sticky=tk.N + tk.S + tk.E + tk.W)

        self.status = tk.Label(self, text='Welcome to Autostitch UI')
        self.status.grid(row=nrows - 1, columnspan=ncolumns, sticky=tk.S)

        self.imageCanvas = uiutils.ImageWidget(self)
        self.imageCanvas.grid(row=nrows - 2, columnspan=ncolumns,
            sticky=tk.N + tk.S + tk.E + tk.W)

        self.grid_rowconfigure(nrows - 2, weight=1)

        for i in range(ncolumns):
            self.grid_columnconfigure(i, weight=1)

    def setStatus(self, text):
        self.status.configure(text=text)
        self.root.update()

    def setImage(self, cvImage):
        if cvImage is not None:
            self.imageCanvas.drawCVImage(cvImage)

    def askForImage(self):
        filename = tkFileDialog.askopenfilename(parent=self,
            filetypes=uiutils.supportedFiletypes)
        if filename and os.path.isfile(filename):
            image = cv2.imread(filename)
            self.setStatus('Loaded ' + filename)
            return image
        return None

    def saveScreenshot(self):
        if self.imageCanvas.hasImage():
            filename = tkFileDialog.asksaveasfilename(parent=self,
                filetypes=uiutils.supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved screenshot to ' + filename)
        else:
            uiutils.error('Load image before taking a screenshot!')


class HomographyVisualizationFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self,root, parent, 3, 3)

        tk.Button(self, text='Load Image', command=self.loadImage).grid(
            row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=1, sticky=tk.W + tk.E)

        tk.Button(self, text='Apply Homography', command=self.applyHomography) \
            .grid(row=0, column=2, sticky=tk.W + tk.E)

        self.image = None

    def loadImage(self):
        image = self.askForImage()
        if image is not None:
            self.image = image
            self.setImage(image)

    def applyHomography(self):
        if self.image is not None:
            homography = uiutils.showMatrixDialog(self, text='Apply', rows=3,
                columns=3)
            if homography is not None:
                height, width, _ = self.image.shape
                self.setImage(cv2.warpPerspective(self.image, homography,
                    (width, height)))
                self.setStatus('Applied the homography.')
        else:
            uiutils.error('Select an image before applying a homography!')


class SphericalWarpFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root, 7, 6)

        tk.Button(self, text='Load Image', command=self.loadImage).grid(
            row=0, column=0, columnspan=2, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=2, columnspan=2, sticky=tk.W + tk.E)

        tk.Button(self, text='Warp Image', command=self.warpImage) \
            .grid(row=0, column=4, columnspan=2, sticky=tk.W + tk.E)

        # TODO: specify units and correct ranges
        tk.Label(self, text='Focal Length').grid(row=1, column=0, \
                columnspan=2, sticky=tk.W)
        self.focalLengthSlider = tk.Scale(self, from_=100, to=1000,
            resolution=0.1, orient=tk.HORIZONTAL)
        self.focalLengthSlider.grid(row=1, column=2, \
                columnspan=2, sticky=tk.W+tk.E)
        self.focalLengthSlider.set(600)
        self.focalLengthSlider.bind("<ButtonRelease-1>", self.warpImage)

        tk.Label(self, text='k1:').grid(row=2, column=4,
            sticky=tk.W)
        self.k1Entry = tk.Entry(self)
        self.k1Entry.insert(0, str(DEFAULT_K1)) 
        self.k1Entry.grid(row=2, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k2:').grid(row=3, column=4,
            sticky=tk.W)
        self.k2Entry = tk.Entry(self)
        self.k2Entry.insert(0, str(DEFAULT_K2)) 
        self.k2Entry.grid(row=3, column=5, sticky=tk.W + tk.E)

        self.image = None

    def loadImage(self):
        image = self.askForImage()
        if image is not None:
            self.image = image
            self.setImage(image)

    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2

    def warpImage(self, *args):
        if self.image is not None:
            outImage = np.copy(self.image)
            focalLength = float(self.focalLengthSlider.get())
            k1 = self.getK1()
            k2 = self.getK2()
            warpedImage = warp.warpSpherical(self.image, focalLength, \
                    k1, k2)
            self.setImage(warpedImage)
            self.setStatus('Warped image with focal length ' + str(focalLength))
        elif len(args) == 0: # i.e., click on the button
            uiutils.error('Select an image before warping!')


class StitchingBaseFrame(BaseFrame):
    def __init__(self, parent, root, nrows, ncolumns):
        BaseFrame.__init__(self, parent, root, nrows, ncolumns)

        self.motionModelVar = tk.IntVar()

        tk.Label(self, text='Motion Model:').grid(row=0, column=2, sticky=tk.W)

        tk.Radiobutton(self, text='Translation', variable=self.motionModelVar,
            value=alignment.eTranslate).grid(row=0, column=3, sticky=tk.W)

        tk.Radiobutton(self, text='Homography', variable=self.motionModelVar,
            value=alignment.eHomography).grid(row=0, column=3, sticky=tk.E)

        self.motionModelVar.set(alignment.eHomography)
        
        tk.Label(self, text='Percent Top Matches for Alignment:').grid(row=1, 
            column=0, sticky=tk.W)

        self.matchPercentSlider = tk.Scale(self, from_=0.0, to=100.0, 
            resolution=1, orient=tk.HORIZONTAL)
        self.matchPercentSlider.set(5.0)
        self.matchPercentSlider.grid(row=1, column=1, sticky=tk.W + tk.E)
        self.matchPercentSlider.bind("<ButtonRelease-1>", self.compute)

        tk.Label(self, text='Number of RANSAC Rounds:').grid(row=1,
            column=2, sticky=tk.W)
        
        # TODO: determine sane values for this
        self.nRANSACSlider = tk.Scale(self, from_=1, to=10000, resolution=10,
            orient=tk.HORIZONTAL)
        self.nRANSACSlider.set(100)
        self.nRANSACSlider.grid(row=1, column=3, sticky=tk.W + tk.E)
        self.nRANSACSlider.bind("<ButtonRelease-1>", self.compute)

        tk.Label(self, text='RANSAC Threshold:').grid(row=1, column=4,
            sticky=tk.W)
        
        # TODO: determine sane values for this
        self.RANSACThresholdSlider = tk.Scale(self, from_=0.1, to=100,
            resolution=0.1, orient=tk.HORIZONTAL)
        self.RANSACThresholdSlider.set(5)
        self.RANSACThresholdSlider.grid(row=1, column=5, sticky=tk.W + tk.E)
        self.RANSACThresholdSlider.bind("<ButtonRelease-1>", self.compute)
        
        tk.Label(self, text='Focal Length (pixels):').grid(row=2, column=4,
            sticky=tk.W)
        self.focalLengthEntry = tk.Entry(self)
        self.focalLengthEntry.insert(0, str(DEFAULT_FOCAL_LENGTH)) 
        self.focalLengthEntry.grid(row=2, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k1:').grid(row=3, column=4,
            sticky=tk.W)
        self.k1Entry = tk.Entry(self)
        self.k1Entry.insert(0, str(DEFAULT_K1)) 
        self.k1Entry.grid(row=3, column=5, sticky=tk.W + tk.E)

        tk.Label(self, text='k2:').grid(row=4, column=4,
            sticky=tk.W)
        self.k2Entry = tk.Entry(self)
        self.k2Entry.insert(0, str(DEFAULT_K2)) 
        self.k2Entry.grid(row=4, column=5, sticky=tk.W + tk.E)

    def computeMapping(self, leftImage, rightImage):
        leftGrey = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
        rightGrey = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB()
        leftKeypoints, leftDescriptors = orb.detectAndCompute(leftGrey, None)
        rightKeypoints, rightDescriptors = orb.detectAndCompute(rightGrey, 
            None)
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(leftDescriptors, rightDescriptors)
        matches = sorted(matches, key = lambda x:x.distance)
        nMatches = int(float(self.matchPercentSlider.get()) * 
            len(matches) / 100)
      
        if nMatches < 4:
            return None

        matches = matches[:nMatches]
        motionModel = self.motionModelVar.get()
        nRANSAC = int(self.nRANSACSlider.get())
        RANSACThreshold = float(self.RANSACThresholdSlider.get())
      
        return alignment.alignPair(leftKeypoints, rightKeypoints, matches,
            motionModel, nRANSAC, RANSACThreshold)

    def compute(self, *args):
        raise NotImplementedError('Implement the computation')

    def getFocalLength(self):
        focalLength = 0
        try:
            focalLength = float(self.focalLengthEntry.get())
            if focalLength > 0:
                return focalLength
        except:
            pass
        uiutils.error('You entered an invalid focal length! Please try again.')
        return 0

    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2

class AlignmentFrame(StitchingBaseFrame):
    def __init__(self, parent, root):
        StitchingBaseFrame.__init__(self, parent, root, 9, 6)

        tk.Button(self, text='Load Left Image', command=self.loadLeftImage)\
            .grid(row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Load Right Image', command=self.loadRightImage)\
            .grid(row=0, column=1, sticky=tk.W + tk.E)
        
        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=4, sticky=tk.W + tk.E)

        tk.Button(self, text='Align Images', command=self.alignImagesClick) \
            .grid(row=0, column=5, sticky=tk.W + tk.E)

        self.leftImage = None
        self.rightImage = None

    def loadLeftImage(self):
        image = self.askForImage()
        if image is not None:
            self.leftImage = image
            self.applyVisualization()

    def loadRightImage(self):
        image = self.askForImage()
        if image is not None:
            self.rightImage = image
            self.applyVisualization()

    def applyVisualization(self):
        self.setImage(uiutils.concatImages([self.leftImage, self.rightImage]))

    def alignImagesClick(self):
        if self.leftImage is None or self.rightImage is None:
            uiutils.error('Both the images must be selected for alignment to ' +
                'be possible!')
        else:
            self.compute()

    def compute(self, *args):
        if self.leftImage is not None and self.rightImage is not None:
            focalLength = self.getFocalLength()
            k1 = self.getK1()
            k2 = self.getK2()
            if focalLength <= 0:
                return
            if self.motionModelVar.get() == alignment.eTranslate:
                left = warp.warpSpherical(self.leftImage, focalLength, \
                        k1, k2)
                right = warp.warpSpherical(self.rightImage, focalLength, \
                        k1, k2)
            else:
                left = self.leftImage
                right = self.rightImage
            mapping = self.computeMapping(left, right)
            height, width, _ = right.shape
            
            # TODO what if the mapping is singular?
            mapping = np.linalg.inv(mapping)

            topRight = np.array([width, height, 1])
            tranTopRight = np.dot(mapping, topRight)
            tranTopRight = np.divide(tranTopRight, tranTopRight[2])

            bottomRight = np.array([width, 0, 1])
            tranBottomRight = np.dot(mapping, bottomRight)
            tranBottomRight = np.divide(tranBottomRight, tranBottomRight[2])

            newHeight = int(abs(tranTopRight[1] - tranBottomRight[1]))
            newWidth = int(max(tranTopRight[0], tranBottomRight[0]))

            warpedRightImage = cv2.warpPerspective(right, mapping, 
                (newWidth, newHeight))
            warpedLeftImage = cv2.warpPerspective(left, np.eye(3, 3), 
                (newWidth, newHeight))

            alpha = 0.5
            beta = 1.0 - alpha
            gamma = 0.0
            dst = cv2.addWeighted(warpedLeftImage, alpha, warpedRightImage, \
                beta, gamma)
            self.setImage(dst)
            

class PanoramaFrame(StitchingBaseFrame):
    def __init__(self, parent, root):
        StitchingBaseFrame.__init__(self, parent, root, 9, 6)
        
        tk.Button(self, text='Load Directory', command=self.loadImages) \
            .grid(row=0, column=0, sticky=tk.W + tk.E)

        tk.Button(self, text='Screenshot', command=self.saveScreenshot).grid(
            row=0, column=4, sticky=tk.W + tk.E)

        tk.Button(self, text='Stitch', command=self.compute) \
            .grid(row=0, column=5, sticky=tk.W + tk.E)
        
        tk.Label(self, text='Blend Width (pixels):').grid(row=2, column=0,
            sticky=tk.W)
        self.blendWidthSlider = tk.Scale(self, from_=0, to=200,
            resolution=1, orient=tk.HORIZONTAL)
        self.blendWidthSlider.grid(row=2, column=1, sticky=tk.W + tk.E)
        self.blendWidthSlider.set(50)
        
        self.is360Var = tk.IntVar()
        tk.Checkbutton(self, text='360 degree Panorama?', 
            variable=self.is360Var, offvalue=0, onvalue=1).grid(row=2, column=3, 
            sticky=tk.W)
        self.is360Var.set(0)
               
        self.images = None
 
    def loadImages(self):
        dirpath = tkFileDialog.askdirectory(parent=self)
        if not dirpath:
            return
        files = sorted(os.listdir(dirpath))
        files = [f for f in files if f.endswith('.jpg') or f.endswith('.png') \
            or f.endswith('.ppm')]
        self.images = [cv2.imread(os.path.join(dirpath, i)) for i in files]
        self.setImage(uiutils.concatImages(self.images))
        self.setStatus('Loaded {0} images from {1}'.format(len(self.images),
          dirpath))
        
    def getK1(self):
        k1 = DEFAULT_K1
        try:
            k1 = float(self.k1Entry.get())
        except:
            uiutils.error('You entered an invalid k1! Please try again.')
        return k1

    def getK2(self):
        k2 = DEFAULT_K2
        try:
            k2 = float(self.k2Entry.get())
        except:
            uiutils.error('You entered an invalid k2! Please try again.')
        return k2

    def compute(self, *args):
        if self.images is not None and len(self.images) > 0:
            f = self.getFocalLength()
            if f <= 0:
              return
            k1 = self.getK1()
            k2 = self.getK2()

            processedImages = None

            if self.motionModelVar.get() == alignment.eTranslate:
                processedImages = [warp.warpSpherical(i, f, k1, k2) \
                    for i in self.images]
            else:
                processedImages = self.images

            t = np.eye(3)
            ipv = []
            for i in range(0, len(processedImages) - 1):
                self.setStatus('Computing mapping from {0} to {1}'.format(i,
                  i+1))
                ipv.append(blend.ImageInfo('', processedImages[i], t))
                t = self.computeMapping(processedImages[i+1], processedImages[i])\
                    .dot(t) 
            ipv.append(blend.ImageInfo('', processedImages[len(processedImages)-1],t))
            self.setStatus('Blending Images')
            self.setImage(blend.blendImages(ipv, 
                int(self.blendWidthSlider.get()), self.is360Var.get() == 1))
            self.setStatus('Panorama generated')
        else:
          uiutils.error('Select a folder with images before creating the ' + 
              'panorama!');


if __name__ == '__main__':
    args = parse_args()
    root = tk.Tk()
    app = AutostitchUIFrame(root, root)
    root.title('Cornell CS 4670 - Autostitch Project')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry("%dx%d+0+0" % (w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.mainloop()

