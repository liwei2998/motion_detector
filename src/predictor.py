#!/usr/bin/env python

import message_filters
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import math
import time
import classes_liweii as cl
import tf
import copy
from std_msgs.msg import UInt16

class Motion_predictor:
    def __init__(self):
        #step1: initialize node, publishers, subscribers
        rospy.init_node("motion_detector_predictor_node")
        self.bridge = CvBridge()
        # self.pub1 = rospy.Publisher('camera/visible/image1', Image, queue_size=2)
        self.pub1 = rospy.Publisher('camera/matching_points_image', Image, queue_size=2)
        self.pub3 = rospy.Publisher('cornerMatch/vertexG',Point, queue_size=2)
        self.pub4 = rospy.Publisher('cornerMatch/vertexW',Point, queue_size=2)
        self.pub5 = rospy.Publisher('cornerMatch/MatchPoints',Point, queue_size=2)
        self.sub1 = rospy.Subscriber("usb_cam_k/image_raw", Image, self.imageCallback1)
        self.br = tf.TransformBroadcaster()
        self.MidPoint = None

        #step2: get the paper's position (main class: GetTrans_new)
        #input result img!!!!
        img_src = cv2.imread("cropped_sample/left0030.jpg")
        motion_detector1 = cl.GetCreases()
        result_img, creases = motion_detector1.detect(img_src,np.array([[0, 0], [290, 0], [290, 290], [0, 290]]))
        creases = [[[0,0],[290,290]],[[0,290],[290,0]]]
        pts_src = np.array([[-145, -145], [145, -145], [145, 145], [-145, 145]])
        self.motion_detector0 = cl.Predictor(pts_src,creases,creases[0],result_img)
        self.motion_detector0.get_facets_info(result_img,0)
        self.grasp_point = copy.deepcopy(self.motion_detector0.state['state1']['match_info']['grasp_pts_src'])
        self.target_point = copy.deepcopy(self.motion_detector0.state['state1']['match_info']['target_pts_src'])

    def imageCallback1(self,image):
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        clean_image = cv_image.copy()
        is_blue=self.motion_detector1.detect_blue(clean_image) #if there is blue color, return 1
        # print 'if blue',is_blue
        # self.MidPoint = None
        # print 'pts src',self.motion_detector1.pts_src
        pt1 = copy.deepcopy(self.motion_detector1.grasp_point)
        pt2 = copy.deepcopy(self.motion_detector1.target_point)
        mid_point,result_img, image2 = self.motion_detector1.detect_mid_point(clean_image,pt1,pt2)

        if mid_point is not None and is_blue==0:
            MidPoint = Point()
            MidPoint.x = mid_point[0]
            MidPoint.y = mid_point[1]
            MidPoint.z = mid_point[2]
            print 'mid point',mid_point
            self.MidPoint = MidPoint
            self.pub5.publish(MidPoint)

            image = self.bridge.cv2_to_imgmsg(result_img)
            image3 = self.bridge.cv2_to_imgmsg(image2)
            self.pub1.publish(image)
            self.pub2.publish(image3)
        else:
            if self.MidPoint is not None:
                self.pub5.publish(self.MidPoint)

if __name__ == '__main__':
    detector = Motion_predictor()
