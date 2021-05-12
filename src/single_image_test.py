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


if __name__ == '__main__':
    # detector = Motion()
    # Read in the image.

    for i in range(100):
        # print 'i',i
        img_src = cv2.imread('corner_match_night/left'+"{0:0>4}".format(i)+'.jpg')
        print 'str','corner_match_night/left'+"{0:0>4}".format(i)+'.jpg'
        motion_detector = cl.CornerMatch()

        color = 'green'
        # #step1: color filter and canny
        img_src1 = motion_detector.filter(img_src,color)
        cv2.imshow('color filter',img_src1)
        result_img = motion_detector.detect(img_src1)
        cv2.imshow("canny detection", result_img)

        #step2: roi mask
        result_img2 = motion_detector.ROI_mask(result_img,color)
        cv2.imshow('roi region',result_img2)

        #step3: houghline transform
        lines = cv2.HoughLinesP(result_img2,
                                rho=2,              #Distance resolution in pixels
                                theta=np.pi / 180,  #Angle resolution in radians
                                threshold=60,      #Min. number of intersecting points to detect a line
                                lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
                                minLineLength=2,   #Line segments shorter than this are rejected
                                maxLineGap=25       #Max gap allowed between points on the same line
                                )
        # print 'lines',lines
        if lines is None:
            continue
        averaged_lines = motion_detector.avg_lines(img_src, lines)              #Average the Hough lines as left or right lanes
        combined_image = motion_detector.draw_lines(img_src, averaged_lines, 5)
        cv2.imshow('houghline transform',combined_image)

        cv2.waitKey(0)


    # img_src = cv2.imread("corner_match_night/left0199.jpg")
    # img_src = cv2.imread("corner_match_night/left0257.jpg")
    # img_src = cv2.imread("corner_match_night/left0088.jpg")
    # motion_detector = cl.CornerMatch()
    # cv2.imshow('src image',img_src)
    # color = 'green'
    # # #step1: color filter and canny
    # img_src1 = motion_detector.filter(img_src,color)
    # cv2.imshow('color filter',img_src1)
    # result_img = motion_detector.detect(img_src1)
    # cv2.imshow("canny detection", result_img)
    #
    # #step2: roi mask
    # result_img2 = motion_detector.ROI_mask(result_img,color)
    # cv2.imshow('roi region',result_img2)
    #
    # #step3: houghline transform
    # lines = cv2.HoughLinesP(result_img2,
    #                         rho=2,              #Distance resolution in pixels
    #                         theta=np.pi / 180,  #Angle resolution in radians
    #                         threshold=60,      #Min. number of intersecting points to detect a line
    #                         lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
    #                         minLineLength=2,   #Line segments shorter than this are rejected
    #                         maxLineGap=25       #Max gap allowed between points on the same line
    #                         )
    # # print 'lines',lines
    #
    # averaged_lines = motion_detector.avg_lines(img_src, lines)              #Average the Hough lines as left or right lanes
    # combined_image = motion_detector.draw_lines(img_src, averaged_lines, 5)
    # cv2.imshow('houghline transform',combined_image)
    #
    # cv2.waitKey(0)
