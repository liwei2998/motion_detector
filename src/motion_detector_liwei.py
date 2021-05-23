#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
import math
import time
import classes as cl
import tf
import copy

class Motion:
    def __init__(self):
        rospy.init_node("motion_detector_node")
        self.bridge = CvBridge()
        self.pub1 = rospy.Publisher('camera/visible/image1', Image, queue_size=2)
        self.pub2 = rospy.Publisher('camera/visible/image2', Image, queue_size=2)
        # self.pub3 = rospy.Publisher('cornerMatch/vertexG',Point, queue_size=2)
        # self.pub4 = rospy.Publisher('cornerMatch/vertexW',Point, queue_size=2)

        # self.sub2 = rospy.Subscriber("usb_cam_k/image_raw", Image, self.imageCallback2)
        self.br = tf.TransformBroadcaster()

        self.motion_detector0 = cl.ColorFilter()
        pts_src = [[0, 0], [145, 0], [0, 145]]
        # pts_src = pts_src[::-1]
        A = np.matrix([[741.2212917530331, 0, 311.8358797867751],
                       [0, 741.2317153584389, 240.6847621777156], [0.0, 0.0, 1.0]])  #intrinsic parameters of camera
        self.motion_detector1 = cl.GetTrans_new(pts_src,A)
        self.motion_detector1.mainFunc()
        self.sub1 = rospy.Subscriber("usb_cam_k/image_raw", Image, self.imageCallback1)

        rospy.spin()

    def imageCallback1(self,image):

        T = copy.deepcopy(self.motion_detector1.T)
        R = copy.deepcopy(self.motion_detector1.R)
        print 'T',T

        if T[2] > 0:
            error = T[2] - 0.645
            # print 'error',error
            T[0] = -T[0]-error
            T[1] = -T[1]-error
            # print 't0',T[0]
            # print 't1',T[1]
            trans_T = [T[1],-T[0],T[2]]
        else:
            error = -0.645-T[2]
            # T[0] = T[0]*(1-error)
            # T[1] = T[1]*(1-error)
            T[0]=T[0]-error
            T[1]=T[1]-error
            # print 'error',error
            # print 't0',T[0]
            # print 't1',T[1]            
            trans_T = [T[1],-T[0],-T[2]]
  
        if abs(T[2]) <= 1:
            quaternion = tf.transformations.quaternion_from_euler(R[0], R[1], R[2], axes='sxyz')
            self.br.sendTransform(trans_T,quaternion,rospy.Time.now(),"paper","usb_cam1")


    def imageCallback2(self, image):

        # if self.motion_detector2:
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        clean_image = cv_image.copy()

        result_img3,vertex_w,vertex_g = self.motion_detector3.mainFuc(clean_image)

        image3 = self.bridge.cv2_to_imgmsg(result_img3)
        self.pub2.publish(image3)

        if vertex_g is not None:
            vertexG = Point()
            vertexG.x = vertex_g[0]
            vertexG.y = vertex_g[1]
            vertexG.z = -1
            self.pub3.publish(vertexG)
        else:
            vertexG = Point()
            vertexG.x = vertexG.y = vertexG.z =-1
            self.pub3.publish(vertexG)
        if vertex_w is not None:
            vertexW = Point()
            vertexW.x = vertex_w[0]
            vertexW.y = vertex_w[1]
            vertexW.z = -1
            self.pub4.publish(vertexW)
        else:
            vertexW = Point()
            vertexW.x = vertexW.y = vertexW.z = -1
            self.pub4.publish(vertexW)

if __name__ == '__main__':
    detector = Motion()