"""
findholes.py

Locates centers of holes in 'square.jpg' and 'subsquare.jpg' images.
Produces new versions of images with the center of each hole labeled.
Find the region on 'square.jpg' that corresponds to 'subsquare.jpg'
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# tools for feature detection
sift = cv2.SIFT_create()  # Scale Invariant Feature Transform detector
bf = cv2.BFMatcher()  # Brute Force Matcher


class Image(object):
    """
    A class used to represent an image.
    """
    # tunable variables for contour detection
    threshold_area_slider = 1 / 5  # (0, 1]: lower number includes more contours of smaller area
    """threshold_area_slider : slider used to filter contours based on area"""
    threshold_gs_slider = 1 / 2  # [0, 1]: lower number means higher contrast needed to pass threshold
    """threshold_gs_slider : slider used to determine grayscale threshold when converting image to binary image"""

    def __init__(self, fp):
        """
        :param fp: str
            file path (including name and file type) to image
        """
        self.key_points = None
        """key_points : list of keypoints describing features found though SIFT"""
        self.descriptors = None
        """descriptors : numpy array of descriptors describing features found through SIFT"""
        self.binary_img = None
        """binary_img : binary version of original image"""
        self.contours = None
        """contours : list of arrays containing contours found in binary image"""
        self.contour_img = None
        """contour_img : original image with contours drawn on top"""
        self.matches = None
        """matches : list of matching features across images"""
        self.path = fp
        """fp : file path (including name and extension) to image"""
        self.img = cv2.imread(fp)
        """img : image loaded from fp"""
        self.gray_scale_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        """gray_scale_img : gray scale version of img"""

        # find features and contours in image
        self.find_contours()
        self.find_features()

    def find_features(self):
        """
        A method that uses sift to detect key_points and descriptors in image.
        :return:
        """
        if self.contours is None:
            self.find_contours()
        # mask = np.dstack([cv2.bitwise_not(self.binary_img)]*3)
        self.key_points, self.descriptors = sift.detectAndCompute(self.img, None)
        self.partition_features_amongst_contours()

    def partition_features_amongst_contours(self):
        """A method to partition features amongst contours."""
        feature_index = -1
        for kp, des in zip(self.key_points, self.descriptors):
            feature_index += 1
            for contour_obj in self.contours:
                if cv2.pointPolygonTest(contour_obj.contour, kp.pt, True):
                    contour_obj.add_feature(kp, des, feature_index)
                    break

    def find_contours(self):
        """
        A method to find contours in image.
        :return:
        """
        _, self.binary_img = cv2.threshold(self.gray_scale_img, int(self.threshold_gs_slider * 255), 255, cv2.THRESH_BINARY)
        contour_list = cv2.findContours(self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        self.create_contour_obj(self.filter_contours(contour_list))

    def create_contour_obj(self, contours_list):
        """A method to create a list of Contour objects."""
        self.contours = [Contour(contour, label, self) for label, contour in enumerate(contours_list)]

    def filter_contours(self, contour_list):
        """
        A method to filter the contours in image based on a threshold area.
        :param contour_list: list of all contours in image
        :return: list of contours in image with acceptable area
        """
        contour_areas = [cv2.contourArea(contour) for contour in contour_list]
        threshold_area = self.threshold_area_slider * (min(contour_areas) + max(contour_areas))
        return [contour for contour in contour_list if cv2.contourArea(contour) > threshold_area]

    def find_moments_of_contours(self, font_size=0.5, dot_size=2):
        """
        A method to find and label the center of contours in binary_img.
        :param font_size: font size of contour label
        :param dot_size: size of dot which marks the center of the contour
        :return:
        """
        if self.contours is None:
            self.find_contours()
        if self.contour_img is None:
            self.draw_contours()
        for contour_obj in self.contours:
            moments = cv2.moments(contour_obj.contour)
            cX = int(moments["m10"]/moments["m00"])
            cY = int(moments["m01"]/moments["m00"])
            contour_obj.set_center((cX, cY))
            cv2.circle(self.contour_img, (cX, cY), dot_size, (0, 0, 255), -1)
            cv2.putText(self.contour_img, str(contour_obj.label), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
        cv2.imwrite(self.path.split('.')[0] + '_labeled.jpg', self.contour_img)

    def draw_contours(self, contours_to_draw=None):
        """
        A method to (create and) draw contours on contour_img.
        :return:
        """
        if self.contours is None:
            self.find_contours()
        self.contour_img = self.img.copy()
        if contours_to_draw is None:
            contours_to_draw = [contour_obj.contour for contour_obj in self.contours]
        cv2.drawContours(self.contour_img, contours_to_draw, -1, (0, 255, 0), 2)

    def show_contours(self):
        """
        A method to display contour_img.
        :return:
        """
        if self.contours is None:
            self.find_contours()
        if self.contour_img is None:
            self.draw_contours()
        self.show_image(self.contour_img)

    def show_image(self, img=None):
        """
        A method to display img.
        :param img: image to display
        :return:
        """
        img = self.img if img is None else img
        plt.imshow(img, cmap='gray')
        plt.show()

    def show_features(self):
        """
        A method to display the key_points of img.
        :return:
        """
        img = cv2.drawkey_points(self.img, self.key_points, self.img)
        self.show_image(img)

    def match_features(self, comparison_image):
        """
        A method to find matching features between self and comparison_image (Image object).
        :param comparison_image: Image object to find matching features in.
        :return:
        """
        # use Brute Force algorithm with k=2 Nearest Neighbors to find matching features
        matches = bf.knnMatch(self.descriptors, comparison_image.descriptors, k=2)
        self.matches = []
        # Lowe's ratio test to check that distances between two nearest neighbors are different enough
        for neighbor1, neighbor2 in matches:
            if neighbor1.distance < 0.75*neighbor2.distance:
                self.matches.append(neighbor1)
        # self.highlight_matched_contours(comparison_image)

    def highlight_matched_contours(self, comparison_image):
        """
        A method to highlight the contours that have been matched across images.
        :param comparison_image: Image object to find matching features in.
        :return:
        """
        if self.matches is None:
            self.match_features(comparison_image)
        highlighted_contours = []
        for match in self.matches:
            for contour_obj in self.contours:
                feature_idx_list = [ft[2] for ft in contour_obj.features]
                if match.trainIdx in feature_idx_list:
                    highlighted_contours.append(contour_obj.contour)
        self.draw_contours(highlighted_contours)

    def show_matches(self, comparison_image):
        """
        A method to display matching features between img and comparison_image.
        :param comparison_image: Image object to find matching features in.
        :return:
        """
        if self.matches is None:
            self.match_features(comparison_image)
        img = cv2.drawMatches(self.img, self.key_points, comparison_image.img, comparison_image.key_points, self.matches, self.img)
        cv2.imwrite(self.path.split('.')[0] + '_matches.jpg', img)
        self.show_image(img)


class Contour(object):
    """
    A class used to represent a contour.
    """
    def __init__(self, contour, label, parent):
        self.features = []
        """features : tuple of keypoints, descriptors, and index describing features found though SIFT"""
        self.center = None
        """center : center of contour"""
        self.parent = parent
        """parent : parent image of contour"""
        self.contour = contour
        """contour : numpy array describing contour"""
        self.label = label
        """label : label of contour"""

    def set_center(self, center):
        """
        A method to set the center of the contour as (cX, cY).
        """
        self.center = center

    def add_feature(self, key_points, descriptors, idx):
        """
        A method to set the key_points, descriptors, and features index of the contour.
        """
        self.features.append((key_points, descriptors, idx))


if __name__=='__main__':
    # create instance of image for square.jpg
    square = Image('square.jpg')
    square.find_moments_of_contours()  # find center and label contours
    square.show_contours()  # show labeled contours

    # create instance of image for subsquare.jpg
    subsquare = Image('subsquare.jpg')
    subsquare.find_moments_of_contours(dot_size=5, font_size=2)  # find center and label contours
    subsquare.show_contours()  # show labeled contours

    # match features between square and subsquare
    subsquare.show_matches(square)

""" FEATURE DETECTION/MATCHING
1. find features in image using SIFT algorithm
2. match features using brute force algorithm
3. use homography to highlight magnified region >> need at least 4 points
OR
1. scalable template matching
"""

"""LOCATE/LABEL HOLES
1. convert image to grayscale then to binary based on threshold
2. find contours in image using opencv.find_contours
3. find centers of contours using opencv.moments
OR
1. scalable template matching
"""

"""POSSIBLE IMPROVEMENTS
A. BINARY MASK THEN FEATURE DETECTION
    1. binary dilation on each circle
    2. create mask of holes only
    3. use mask to filter out any non-holes
    4. perform feature extraction only on filtered image
    r. (no features unnecessary features detected)
B. IMAGE AS CLASS WITH METHODS FOR IMAGE PROCESSING
    1. Methods
        * SIFT
        * BF Matcher
    2. Attributes
        * (key_points, descriptors)
C. HOLES AS CLASSES WITH ATTRIBUTES
    * HOLES extend IMAGE class?
D. USE TEMPLATE MATCHING INSTEAD OF CONTOURS
E. USE DISTRIBUTION OF CONTOUR AREA FOR FILTERING CONTOURS
F. HIGHLIGHT MATCHING CONTOURS IN EACH IMAGE
"""