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

    # to get the lower and upper bound of one color
    # img_src = cv2.imread('corner_match_night/left0176.jpg')
    # motion_detector = cl.CornerMatch()
    # motion_detector.hsv_calc(img_src)


    for i in range(176,178):
        # print 'i',i
        img_src = cv2.imread('corner_match_night/left'+"{0:0>4}".format(i)+'.jpg')
        print 'str','corner_match_night/left'+"{0:0>4}".format(i)+'.jpg'
        motion_detector = cl.CornerMatch()

        #step1: color filter
        img_src1 = motion_detector.filter(img_src,'white')
        cv2.imshow('color filter1',img_src1)
        img_src2 = motion_detector.filter(img_src,'green')
        cv2.imshow('color filter2',img_src2)

        #step2: canny detection
        result_img_white = motion_detector.detect(img_src1)
        # cv2.imshow("canny detection", result_img)
        result_img_green = motion_detector.detect(img_src2)
        # cv2.imshow('canny detection green',result_img_green)
        result_img_white = motion_detector.get_white_line(result_img_white,img_src2) #get white line
        cv2.imshow('canny detection white line',result_img_white)

        #step3: roi mask
        result_img2_white = motion_detector.ROI_mask(result_img_white,'white')
        cv2.imshow('roi region white',result_img2_white)
        result_img2_green = motion_detector.ROI_mask(result_img_green,'green')
        cv2.imshow('roi region green',result_img2_green)

        #step4: houghline transform and get intersection point
        #a vertex is the intersection of two lines, return none if only one line
        lines_white = cv2.HoughLinesP(result_img2_white,
                                      rho=2,              #Distance resolution in pixels
                                      theta=np.pi / 180,  #Angle resolution in radians
                                      threshold=60,      #Min. number of intersecting points to detect a line
                                      lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
                                      minLineLength=2,   #Line segments shorter than this are rejected
                                      maxLineGap=25       #Max gap allowed between points on the same line
                                      )
        lines_green = cv2.HoughLinesP(result_img2_green,
                                      rho=2,              #Distance resolution in pixels
                                      theta=np.pi / 180,  #Angle resolution in radians
                                      threshold=60,      #Min. number of intersecting points to detect a line
                                      lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
                                      minLineLength=2,   #Line segments shorter than this are rejected
                                      maxLineGap=25       #Max gap allowed between points on the same line
                                      )
        # print 'lines',lines
        if lines_white is None or lines_green is None:
            continue
        averaged_lines_white = motion_detector.avg_lines(img_src, lines_white)              #Average the Hough lines as left or right lanes
        averaged_lines_green = motion_detector.avg_lines(img_src, lines_green)              #Average the Hough lines as left or right lanes

        combined_image = motion_detector.draw_lines(img_src, averaged_lines_white,
                                                    averaged_lines_green,5,
                                                    color1=[0, 0, 255],color2=[0,255,255]) #draw line for white zone and green zone
        cv2.imshow('houghline transform',combined_image)

        white_vertex = motion_detector.get_intersection_point(averaged_lines_white)
        print 'white vertex',white_vertex
        green_vertex = motion_detector.get_intersection_point(averaged_lines_green)
        print 'green vertex',green_vertex

        cv2.waitKey(0)
