#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import math
import time
import classes as cl
import tf
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

class Motion:
    def __init__(self):
        rospy.init_node("motion_detector_node")
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('camera/visible/image', Image, queue_size=2)
        # self.pub2 = rospy.Publisher()
        rospy.Subscriber("usb_cam/image_raw", Image, self.imageCallback)

        self.br = tf.TransformBroadcaster()

        self.motion_detector0 = cl.ColorFilter()

        # pts_src = np.array([[0, 0], [290, 0], [290, 290], [0, 290]])  #size of paper in real world, must be int && mm
        pts_src = np.array([[145, -145], [145,145], [-145, 145], [-145, -145]])  #size of paper in real world, must be int && mm
        A = np.matrix([[741.2212917530331, 0, 311.8358797867751],
                       [0, 741.2317153584389, 240.6847621777156], [0.0, 0.0, 1.0]])  #intrinsic parameters of camera
        self.motion_detector1 = cl.GetTrans(pts_src,A)
        self.motion_detector2 = cl.GetCreases()
        rospy.spin()

    def imageCallback(self, image):

        # if self.motion_detector2:
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        clean_image = cv_image.copy()
        image0 = self.motion_detector0.detect(cv_image)
        (R,T), result_img1, img_perspCorr = self.motion_detector1.detect(image0, clean_image)
        # (R,T), result_img1= self.motion_detector1.detect(image0, clean_image)

        result_img2 = None
        if img_perspCorr is not None:
            result_img2 = self.motion_detector2.detect(img_perspCorr)

        if R is not None:
            if T[2] <= 1000:
                quaternion = tf.transformations.quaternion_from_euler(R[0], R[1], R[2], axes='sxyz')
                T = (T[0]/1000,T[1]/1000,T[2]/1000)
                self.br.sendTransform(T,quaternion,rospy.Time.now(),"paper","usb_cam1")

                # self.br.sendTransform(T,quaternion,rospy.Time.now(),"usb_cam1","paper")
                # if result_img2 is not None:
                #     image = self.bridge.cv2_to_imgmsg(result_img2)
                #     self.pub.publish(image)
                image = self.bridge.cv2_to_imgmsg(result_img1)
                self.pub.publish(image)
        # self.pub.publish(image)


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections

if __name__ == '__main__':
    # detector = Motion()
    # Read in the image.
    img_src = cv2.imread("corner_match_night/left0199.jpg")
    # img_src = cv2.imread("corner_match_night/left0121.jpg")
    # cv2.imshow('src',img_src)
    # motion_detector = cl.HarrisCorner()
    motion_detector = cl.CornerMatch()
    result_img = motion_detector.detect(img_src)

    # Show output
    # cv2.imshow("Original_image", img_src)
    # cv2.imshow("Image1", img1)
    # cv2.imshow("Image2", img2)
    cv2.imshow("Image", result_img)
    result_img2 = motion_detector.ROI_mask(result_img)
    # motion_detector2 = cl.HarrisCorner()
    # result_img2 = motion_detector2.detect(img1)
    cv2.imshow('image2',result_img2)
    # rho, theta, thresh = 5, 8*np.pi/180, 105
    # lines = cv2.HoughLines(result_img2, rho, theta, thresh)
    # print 'lines',lines
    # for i in range(len(lines)):
    #     for rho,theta in lines[i]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))
    #         cv2.line(img_src,(x1,y1),(x2,y2),(0,255,0),8)


    #Hough transform to detect lanes from the detected edges
    lines = cv2.HoughLinesP(result_img2,
                            rho=2,              #Distance resolution in pixels
                            theta=5*np.pi / 180,  #Angle resolution in radians
                            threshold=100,      #Min. number of intersecting points to detect a line
                            lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
                            minLineLength=10,   #Line segments shorter than this are rejected
                            maxLineGap=25       #Max gap allowed between points on the same line
                            )
    print 'lines',lines

    # img_src_1 = copy.deepcopy(img_src)
    # img_src_1 = motion_detector.draw_lines(img_src_1,np.array([lines]),5)
    # cv2.imshow("Image3", img_src_1)

    # Visualisations
    averaged_lines = motion_detector.avg_lines(img_src, lines)              #Average the Hough lines as left or right lanes

    combined_image = motion_detector.draw_lines(img_src, averaged_lines, 5)

    cv2.imshow('img3',combined_image)
    # segmented = segment_by_angle_kmeans(lines)
    # intersections = segmented_intersections(segmented)
    cv2.waitKey(0)
