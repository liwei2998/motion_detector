#!/usr/bin/env python

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import math
import time
import classes as cl
import tf

from skimage import img_as_ubyte,img_as_float,exposure



if __name__ == '__main__':


##test 1: GetCreeses
 # # For one image
 #    # Read the image.
 #    img_src = cv2.imread("cropped_sample/left0000.jpg")
 #
 #    motion_detector1 = cl.GetCreases()
 #    result_img1, creases = motion_detector1.detect(img_src)
 #    # Show output
 #    # cv2.imshow("Original_image", img_src)
 #    cv2.imshow("Image", result_img1)
 #    cv2.waitKey(0)

 # For all images
    # for i in range(1):
    #     print 'i',i
    #     img_src = cv2.imread('cropped_sample/left'+"{0:0>4}".format(i)+'.jpg')
    #     print 'cropped_sample/left'+"{0:0>4}".format(i)+'.jpg'
    #
    #     motion_detector1 = cl.GetCreases()
    #     result_img1, creases = motion_detector1.detect(img_src)
    #     # Show output
    #     # cv2.imshow("Original_image", img_src)
    #     print "crease info", creases
    #     cv2.imshow("Image", result_img1)
    #     cv2.waitKey(0)
#
# ##test 2: CornerMatch
#  #For one image:
#
#     img_src = cv2.imread('corner_match_sample/left0023.jpg')
#     motion_detector = cl.CornerMatch()
#     #step1: image process and canny
#     result_img = motion_detector.detect(img_src)
#     # cv2.imshow("result_img", result_img)
#
#     #step2: roi mask
#     result_img2 = motion_detector.ROI_mask(result_img)
#     # cv2.imshow('result_img2',result_img2)
#
#     #step3: houghline transform
#     lines = cv2.HoughLinesP(result_img2,
#                             rho=2,              #Distance resolution in pixels
#                             theta=np.pi / 180,  #Angle resolution in radians
#                             threshold=60,      #Min. number of intersecting points to detect a line
#                             lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2]
#                             minLineLength=2,   #Line segments shorter than this are rejected
#                             maxLineGap=25       #Max gap allowed between points on the same line
#                             )
#     # print 'lines',lines
#
#     averaged_lines = motion_detector.avg_lines(img_src, lines)              #Average the Hough lines as left or right lanes
#     combined_image = motion_detector.draw_lines(img_src, averaged_lines, 5)
#
#     cv2.imshow('image',combined_image)
#     cv2.waitKey(0)
#
#  #For all images:
#
#     for i in range(0,277):
#         # print 'i',i
#         img_src = cv2.imread('corner_match_sample/left'+"{0:0>4}".format(i)+'.jpg')
#         print 'str','corner_match_sample/left'+"{0:0>4}".format(i)+'.jpg'
#         motion_detector = cl.CornerMatch()
#
#         image,white_vertex,green_vertex = motion_detector.mainFuc(img_src)
#         cv2.imshow("corner match test",image)
#         print 'white vertex',white_vertex
#         print 'green vertex',green_vertex
#
#         cv2.waitKey(0)
#
# ##test 3: CornerMatch_new:
#  #For 1 image:
#
#         i=1
#
#
#
#         img_src = cv2.imread('corner_match_sample3/left'+"{0:0>4}".format(i)+'.jpg')
#         print 'str','corner_detect_img/left'+"{0:0>4}".format(i)+'.jpg'
#         motion_detector = cl.CornerMatch_new()
#
#       #   skimage_src = img_as_float(img_src)
#       #   skimage = exposure.equalize_adapthist(skimage_src, clip_limit=0.01)
#       #   img_src = img_as_ubyte(skimage)
#
#         image,white_vertex,green_vertex = motion_detector.hsv_calc(img_src)
#         cv2.imshow("corner match",image)
#         print 'white vertex',white_vertex
#         print 'green vertex',green_vertex
#
#         cv2.waitKey(0)


 ##For all images:

    # for i in range(0,1):
    #     # print 'i',i
    #     img_src = cv2.imread('corner_match_sample6/left'+"{0:0>4}".format(i)+'.jpg')
    #     print 'str','imgs_temp/left'+"{0:0>4}".format(i)+'.jpg'
    #     # motion_detector = cl.CornerMatch_new()
    #     # motion_detector.hsv_calc(img_src)
    #     # motion_detector.mainFuc(img_src)
    #     mtx = np.matrix([[741.2212917530331, 0, 311.8358797867751],
    #                     [0, 741.2317153584389, 240.6847621777156],
    #                     [0.0, 0.0, 1.0]])
    #
    #     dist = np.array([[0.04535771070349276, 0.1260442705706077, 0.0008201940323893598, -0.004140650347414909, 0]])
    #
    #     motion_detector = cl.CurveLine(mtx,dist)
    #     motion_detector.mainFuc(img_src)
    #
    #
    #
    #     # image,white_vertex,green_vertex = motion_detector.mainFuc(img_src)
    #     # cv2.imshow("corner match",image)
    #     # print 'white vertex',white_vertex
    #     # print 'green vertex',green_vertex
    #
    #     cv2.waitKey(0)

    # pts_src = np.array([[-145, -145], [145, -145], [145, 145], [-145, 145]])
    # creases = [[[-145,-145],[145,145]]]
    # predict = cl.Predictor(pts_src,creases)
    # countour = predict.get_new_pts_contour()

##test 5: Predictor
 # For one image

    # #step1: get segmented image (segmented by creases).
    # pts_src0 = np.array([[0, 0], [290, 0], [290, 290], [0, 290]])
    # img_src = cv2.imread("cropped_sample/left0000.jpg")
    # motion_detector1 = cl.GetCreases()
    # result_img1, creases = motion_detector1.detect(img_src,pts_src0)
    # print 'creases',creases
    # # cv2.imshow("Original_image", img_src)
    # # cv2.imshow("Image", result_img1)
    #
    # #step2: detect polygons
    # pts_src = np.array([[-145, -145], [145, -145], [145, 145], [-145, 145]])
    # motion_detector2 = cl.Predictor(pts_src,creases,result_img1)
    #
    # cv2.waitKey(0)

 # For all images
    # for i in range(1):
    #     print 'i',i
    #     img_src = cv2.imread('cropped_sample/left'+"{0:0>4}".format(i)+'.jpg')
    #     print 'cropped_sample/left'+"{0:0>4}".format(i)+'.jpg'
    #
    #     motion_detector1 = cl.GetCreases()
    #     result_img1, creases = motion_detector1.detect(img_src)
    #     # Show output
    #     # cv2.imshow("Original_image", img_src)
    #     print "crease info", creases
    #     cv2.imshow("Image", result_img1)
    #     cv2.waitKey(0)

##test 6: get trans

    # cap = cv2.VideoCapture(1)
    # # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    # # cap = cv2.VideoCapture('output.avi')
    # pts_src =[[0, 0], [145, 0],[0, 145]]
    # # pts_src = np.array([[0, 0], [145, 0],[0, 145]])
    # # pts_src = np.array([[0, 0], [205, 0], [102.5, 145]])
    # A = np.matrix([[741.2212917530331, 0, 311.8358797867751],
    #                [0, 741.2317153584389, 240.6847621777156], [0.0, 0.0, 1.0]])  #intrinsic parameters of camera
    #
    #
    # # pts_src = pts_src / 166.5  # convert pixels to meters, can be changed for different sized "H"
    # pts_src = pts_src[::-1]  # reverse the order of the array
    #
    # rotations = []  # create a structure to store information for matlab
    # trans = []
    #
    # while(True):
    #     _, frame = cap.read()
    #
    #     # motion_detector = cl.CornerMatch_new()
    #     # motion_detector.hsv_calc(frame)
    #     # print 'size',frame.shape
    #     motion_detector0 = cl.ColorFilter()
    #     frame=motion_detector0.green_filter(frame)
    #     motion_detector = cl.GetTrans_new(pts_src,A)
    #     motion_detector1 = cl.ColorFilter()
    #     # image0 = motion_detector1.detect(frame)
    #     R_mat, (R,T), result_img1, img2= motion_detector.detect(frame, frame)
    #     cv2.imshow('image',result_img1)
    #
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # #out.release()
    # cv2.destroyAllWindows()
    pts_src = [[0, 0], [145, 0], [0, 145]]

    A = np.matrix([[741.2212917530331, 0, 311.8358797867751],
                   [0, 741.2317153584389, 240.6847621777156], [0.0, 0.0, 1.0]])  #intrinsic parameters of camera

    motion_detector0 = cl.GetTrans_new(pts_src,A)
    motion_detector0.mainFuc()

    print 'R',motion_detector0.R
    print 'T',motion_detector0.T
