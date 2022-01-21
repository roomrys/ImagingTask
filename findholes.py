"""Locates centers of holes in 'square.jpg' and 'subsquare.jpg' images.
Produces new versions of images with the center of each hole labeled.
Find the region on 'square.jpg' that corresponds to 'substrate.jpg'
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Image:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    def __init__(self, filePathAndName):
        self.img = cv2.imread(filePathAndName)
        self.grayScaleImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.findFeatures()
        self.findContours()

    def findFeatures(self):
        self.keyPoints, self.descriptors = self.sift.detectAndCompute(self.img, None)
    
    def findContours(self):
        _, self.binaryImg = cv2.threshold(self.grayScaleImg, 150, 255, cv2.THRESH_BINARY)
        self.contours= cv2.findContours(self.binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
    def showContours(self):
        self.contourImg = self.img.copy()
        cv2.drawContours(self.contourImg, self.contours, -1, (0, 255, 0), 3)
        self.showImage(self.contourImg)

    def showImage(self, img=None):
        img = self.img if img is None else img
        plt.imshow(img, cmap='gray')
        plt.show()

    def showFeatures(self):
        img = cv2.drawKeypoints(self.img, self.keyPoints, self.img)
        self.showImage(img)

    def matchFeatures(self, comparisonImage):
        matches = self.bf.knnMatch(self.descriptors, comparisonImage.descriptors, k=2)
        self.matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                self.matches.append(m)

    def showMatches(self, comparisonImage):
        img = cv2.drawMatches(self.img, self.keyPoints, comparisonImage.img, comparisonImage.keyPoints, self.matches, self.img)
        self.showImage(img)

if __name__=='__main__':
    square = Image('square.jpg')
    # square.showFeatures()

    subsquare = Image('subsquare.jpg')
    # subsquare.showFeatures()
    subsquare.showContours()

    print("NUmber of contours found = " + str(len(subsquare.contours)))
    # subsquare.matchFeatures(square)
    # subsquare.showMatches(square)



""" FEATURE DETECTION/MATCHING
1. find features in image using SIFT algorithm
2. match features using brute force algorithm
3. use homography to highlight magnified region >> need at least 4 points
"""

"""LOCATE/LABEL HOLES
1. convert image to grayscale then to binary based on threshold
2. find contours in image using opencv.findContours
3. find centers of contours using opencv.moments
"""

"""POSSIBLE IMROVEMENTS
A. BINARY MASK THEN FEATURE DETECTION
    1. binary dilation on each circle
    2. create mask of holes only
    3. use mask to filter out any non-holes
    4. perform feature extraction only on filtered image
    r. (no features unneccessary features detected)
B. IMAGE AS CLASS WITH METHODS FOR IMAGE PROCESSING
    1. Methods
        * SIFT
        * BF Matcher
    2. Attributes
        * (keypoints, descriptors)
C. HOLES AS CLASSES WITH ATTRIBUTES
    * HOLES extend IMAGE class?
D. USE TEMPLATE MATCHING INSTEAD OF CONTOURS
"""