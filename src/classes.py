import rospy
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import shapeUtil as su
import time
from numpy.linalg import inv
import geopandas as gpd
from shapely.geometry import Polygon
from skimage import img_as_ubyte,img_as_float,exposure
from skimage.morphology import closing, square
from skimage.measure import label
from skimage.filters import threshold_otsu
from shapely.ops import cascaded_union
from numpy import asarray
import glob
import copy
import Util as ut
import homo_test as ht
kernel_elliptic_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_elliptic_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
area_threshold = 2000


class ColorFilter:
  def __init__(self):
    self.size=[1280,720]

  def detect(self,image):

    lower_black = np.array([0,0,0])  #-- Lower range --
    upper_black = np.array([70,70,70])  #-- Upper range --

    # red color boundaries [B, G, R]; lower = [1, 0, 20]; upper = [60, 40, 200]
    lower_red = np.array([1,0,20])  #-- Lower range --
    upper_red = np.array([40,40,255])  #-- Upper range --

    # lower_white = np.array([150,150,150])  #-- Lower range --
    # upper_white = np.array([255,255,255])  #-- Upper range --

    black_mask1 = cv2.inRange(image, lower_black, upper_black)
    kernel = np.ones((5,5),np.uint8)
    black_mask2 = cv2.dilate(black_mask1,kernel,iterations = 1)
    image[np.where(black_mask2 == [255])] = [160]

    # red_mask = cv2.inRange(image,lower_red,upper_red)
    # ret, thresh = cv2.threshold(red_mask, 50, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)

    # white_mask1 = cv2.inRange(image, lower_white, upper_white)
    # image[np.where(white_mask1 == [0])] = [0]

    return image

class GetTrans:
    def __init__(self,pts_src,A):

        self.A = A
        self.red_point = (0, 0)
        # self.red_lower = [115, 100, 100]
        # self.red_upper = [125, 255, 255]
        #pts_src = pts_src / 1.05  # convert pixels to meters, can be changed for different sized "H"

        self.pts_src = pts_src[::-1]  # reverse the order of the array


    def detect(self, frame, ori_img):

        #global out
        A = self.A
        pts_src = self.pts_src
        R, T = None, None
        im_perspCorr = None # black_image (300,300,3)   np.zeros((300,300,3), np.uint8)
        blurr = cv2.GaussianBlur(frame, (5, 5), 0)
        imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
        imgC = cv2.Canny(imgG, 50, 60)
        imgC = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
        # imgC = cv2.dilate(imgC, (3, 3), iterations=2)
        # (_,cont, _) = cv2.findContours(imgC.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        (_,cont, _)=cv2.findContours(imgC.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        best_approx = None
        lowest_error = float("inf")

        #contour selection
        for c in cont:
            pts_dst = []
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .01 * perim, True)
            area = cv2.contourArea(c)

            if len(approx) == len(self.pts_src):
                right, error = su.rightA(approx, 80) #change the thresh if not look vertically
                # print(right)
                if error < lowest_error and right:
                    lowest_error = error
                    best_approx = approx

        # red_point, _ = su.detectColor(blurr, red_lower, red_upper)
        # if red_point is not None:
           # cv2.circle(frame, (red_point[0] + frame.shape[0] / 2, red_point[1] + frame.shape[1] / 2), 5, (0, 0, 255), 2)

        if best_approx is not None:
            # print 'best approx',best_approx
            # print 'red point',self.red_point
            cv2.drawContours(frame, [best_approx], 0, (255, 0, 0), 3)

            for i in range(0, len(best_approx)):
                pts_dst.append((best_approx[i][0][0], best_approx[i][0][1]))
                # cv2.circle(frame, pts_dst[-1], 3, (i*30, 0, 255-i*20), 3)

            # Correction method for contour points.  Need to make sure the points are mapped correctly
            pts_dst = su.sortContour(
                np.array((self.red_point[0] + pts_dst[0][0], self.red_point[1] + pts_dst[0][1])), pts_dst)

            cv2.circle(frame, pts_dst[0], 7, (0, 255, 0), 4)
            #
            # center = su.line_intersect(pts_dst[0][0],pts_dst[0][1],pts_dst[2][0],pts_dst[2][1],
            #                            pts_dst[1][0],pts_dst[1][1],pts_dst[3][0],pts_dst[3][1])
            # cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), 2)

            for i in range(0, len(best_approx)):
                cv2.circle(frame, pts_dst[i], 3, (i * 30, 0, 255 - i * 20), 3)

            h, status = cv2.findHomography(np.array(pts_src).astype(float), np.array(pts_dst).astype(float))
            center = np.dot(h,(0,0,1))
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), 2)
            # print 'status',status

            (R, T) = su.decHomography(A, h)
            Rot = su.decRotation(R)

            zR = np.matrix([[math.cos(Rot[2]), -math.sin(Rot[2])], [math.sin(Rot[2]), math.cos(Rot[2])]])
            cv2.putText(imgC, 'rX: {:0.2f} rY: {:0.2f} rZ: {:0.2f}'.format(Rot[0] * 180 / np.pi, Rot[1] * 180 / np.pi, Rot[2] * 180 / np.pi), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(imgC, 'tX: {:0.2f} tY: {:0.2f} tZ: {:0.2f}'.format(T[0, 0], T[0, 1], T[0, 2]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            pDot = np.dot((-200, -200), zR)
            # pDot = np.dot((-148, -148),zR)
            self.red_point = (int(pDot[0, 0]), int(pDot[0, 1]))

            # cv2.circle(frame, (int(pDot[0, 0]) + pts_dst[0][0], int(pDot[0, 1]) + pts_dst[0][1]), 5, (0, 0, 255), 2)

            # # get perspective corrected paper
            pts1 = pts_dst
            half_len = int(abs(pts_src[0][0]))
            pts2 = pts_src + np.ones((4,2),dtype=int)*half_len
            M = cv2.getPerspectiveTransform(np.float32(pts1),np.float32(pts2))
            img_size = (half_len*2, half_len*2)
            im_perspCorr = cv2.warpPerspective(ori_img,M,img_size)

        merged_img = np.concatenate((frame, cv2.cvtColor(imgC, cv2.COLOR_BAYER_GB2BGR)), axis=1)
        # merged_img = im_perspCorr

        if R is not None:
            # print 'R',R
            # print 'T',T
            Rotation = Rot
            Translation = (T[0, 0], T[0, 1], T[0, 2])
            return R,(Rotation, Translation), merged_img, im_perspCorr
            # return (Rotation, Translation), merged_img
        else:
            return None, (None, None), merged_img, None
            # return (None, None), merged_img

class GetTrans_new:
    def __init__(self,pts_src,A):

        self.A = A
        self.red_point = (0, 0)
        # self.red_lower = [115, 100, 100]
        # self.red_upper = [125, 255, 255]
        #pts_src = pts_src / 1.05  # convert pixels to meters, can be changed for different sized "H"
        self.pts_src = pts_src[::-1]  # reverse the order of the array


    def detect(self, frame, ori_img):

        #global out
        A = self.A
        pts_src = self.pts_src
        R, T = None, None
        im_perspCorr = None # black_image (300,300,3)   np.zeros((300,300,3), np.uint8)
        blurr = cv2.GaussianBlur(frame, (5, 5), 0)
        imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
        imgC = cv2.Canny(imgG, 50, 60)
        imgC = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
        # imgC = cv2.dilate(imgC, (3, 3), iterations=2)
        # (_,cont, _)=cv2.findContours(imgC.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_,cont, _) = cv2.findContours(imgC.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # print 'contour num',len(cont)
        image = cv2.drawContours(frame,cont,0,(0,0,255),3)
        # cv2.imshow('image',image)
        best_approx = None
        lowest_error = float("inf")

        #contour selection
        for c in cont:
            pts_dst = []
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .01 * perim, True)
            area = cv2.contourArea(c)
            # print 'approx',approx
            # print 'type approx',type(approx)

            if len(approx) == len(self.pts_src):
                right, error, new_approx = su.rightA(approx, 5) #80#change the thresh if not look vertically
                # print(right)
                new_approx = np.array(new_approx)
                # print 'new approx',new_approx
                if error < lowest_error and right:
                    lowest_error = error
                    best_approx = new_approx

        # red_point, _ = su.detectColor(blurr, red_lower, red_upper)
        # if red_point is not None:
           # cv2.circle(frame, (red_point[0] + frame.shape[0] / 2, red_point[1] + frame.shape[1] / 2), 5, (0, 0, 255), 2)

        if best_approx is not None:
            # print 'best approx',best_approx
            # print 'red point',self.red_point
            cv2.drawContours(frame, [best_approx], 0, (255, 0, 0), 3)

            for i in range(0, len(best_approx)):
                pts_dst.append((best_approx[i][0][0], best_approx[i][0][1]))
                # cv2.circle(frame, pts_dst[-1], 3, (i*30, 0, 255-i*20), 3)
            print 'pts  dst',pts_dst
            mid_point = (int((pts_dst[1][0]+pts_dst[2][0])/2),int((pts_dst[1][1]+pts_dst[2][1])/2))
            pts_dst.append(pts_dst[2])
            pts_dst[2] = mid_point
            print 'pts  dst2',pts_dst
            pts_src1 = np.array([[0, 0], [145, 0],[72.5,72.5], [0, 145]])

            # Correction method for contour points.  Need to make sure the points are mapped correctly
            # pts_dst = su.sortContour(
            #     np.array((self.red_point[0] + pts_dst[0][0], self.red_point[1] + pts_dst[0][1])), pts_dst)

            # cv2.circle(frame, pts_dst[0], 7, (0, 255, 0), 4)

            # center = su.line_intersect(pts_dst[0][0],pts_dst[0][1],pts_dst[2][0],pts_dst[2][1],
            #                            pts_dst[1][0],pts_dst[1][1],pts_dst[3][0],pts_dst[3][1])
            # cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), 2)

            # for i in range(0, len(best_approx)):
            #     cv2.circle(frame, pts_dst[i], 3, (i * 30, 0, 255 - i * 20), 3)
            # print 'best approx',best_approx
            h1, status = cv2.findHomography(np.array(pts_src1).astype(float), np.array(pts_dst).astype(float),cv2.RANSAC,5.0)
            h, status = cv2.findHomography(np.array(pts_src1).astype(float), np.array(pts_dst).astype(float))
            print 'h',h
            print 'h1',h1
            center1 = np.dot(h,(0,0,1))
            print 'center1',center1
            cv2.circle(frame, (int(center1[0]), int(center1[1])), 10, (0, 0, 255), 2)
            center2 = np.dot(h,(145,0,1))
            print 'center2',center2
            cv2.circle(frame, (int(center2[0]), int(center2[1])), 10, (0, 255, 0), 2)
            center3 = np.dot(h,(0,145,1))
            print 'center3',center3
            cv2.circle(frame, (int(center3[0]), int(center3[1])), 10, (0, 255, 255), 2)
            # print 'center',center1
            # print 'center1',center1
            print 'status',status

            (R, T) = su.decHomography(A, h)
            ########liwei: change the decompose homography method and do one more transformation (from pixel frame to camera frame)
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(h, A)
            '''
            num possible solutions will be returned.
            Rs contains a list of the rotation matrix.
            Ts contains a list of the translation vector.
            Ns contains a list of the normal vector of the plane.
            '''
            # print 'num',num
            # print 'Ts',Ts
            u0 = A[0,2]
            v0 = A[1,2]
            f = A[0,0]
            Translation = [Ts[0][2]/f*(Ts[0][0]),Ts[0][2]/f*(Ts[0][1]),Ts[0][2]]
            # print 'R',R
            # print 'RS',Rs
            # print 'tranlation3',Translation
            Rot = su.decRotation(np.matrix(Rs[3]))
            ########liwei: change the decompose homography method and do one more transformation (from pixel frame to camera frame)

            zR = np.matrix([[math.cos(Rot[2]), -math.sin(Rot[2])], [math.sin(Rot[2]), math.cos(Rot[2])]])
            cv2.putText(imgC, 'rX: {:0.2f} rY: {:0.2f} rZ: {:0.2f}'.format(Rot[0] * 180 / np.pi, Rot[1] * 180 / np.pi, Rot[2] * 180 / np.pi), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(imgC, 'tX: {:0.2f} tY: {:0.2f} tZ: {:0.2f}'.format(Translation[0][0], Translation[1][0], Translation[2][0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            # pDot = np.dot((-200, -200), zR)
            # pDot = np.dot((-148, -148),zR)
            # self.red_point = (int(pDot[0, 0]), int(pDot[0, 1]))

            # cv2.circle(frame, (int(pDot[0, 0]) + pts_dst[0][0], int(pDot[0, 1]) + pts_dst[0][1]), 5, (0, 0, 255), 2)

            # # get perspective corrected paper
            # pts1 = pts_dst
            # half_len = int(abs(pts_src[0][0]))
            # pts2 = pts_src + np.ones((4,2),dtype=int)*half_len
            # M = cv2.getPerspectiveTransform(np.float32(pts1),np.float32(pts2))
            # img_size = (half_len*2, half_len*2)
            # im_perspCorr = cv2.warpPerspective(ori_img,M,img_size)

        merged_img = np.concatenate((frame, cv2.cvtColor(imgC, cv2.COLOR_BAYER_GB2BGR)), axis=1)
        # merged_img = im_perspCorr

        if R is not None:
            # print 'R',R
            # print 'T',T
            Rotation = Rot
            Translation = (T[0, 0], T[0, 1], T[0, 2])

            # return R,(Rotation, Translation), merged_img, im_perspCorr
            return R, (Rotation, Translation), merged_img, image
        else:
            # return None, (None, None), merged_img, None
            return None,(None, None), merged_img, image

class GetCreases:
  def __init__(self):
    self.size=[1280,720]

  def detect(self,image,pts_src):

    lower_black = np.array([0,0,0])  #-- Lower range --
    upper_black = np.array([70,70,70])  #-- Upper range --

    black_mask1 = cv2.inRange(image, lower_black, upper_black)
    kernel = np.ones((5,5),np.uint8)

    black_mask3 = cv2.dilate(black_mask1,kernel,iterations = 1)
    # black_mask3= cv2.GaussianBlur(black_mask2,(5,5),0)

    # imgC = cv2.Canny(black_mask2, 50, 60)
    # black_mask2 = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
    # dim =  (600,600)
    # black_mask3 = cv2.resize(black_mask2, dim, interpolation = cv2.INTER_AREA)

    minLineLength = 40
    maxLineGap = 40 #250
    lines = cv2.HoughLinesP(black_mask3,1,np.pi/60,100,minLineLength,maxLineGap)
    edges_img = cv2.cvtColor(black_mask3,cv2.COLOR_GRAY2RGB)

    # if lines is not None:
    #   for line in lines:
    #     for x1,y1,x2,y2 in line:
    #       cv2.line(edges,(x1,y1),(x2,y2),(0,255,0),1)

    ##get post process result, and merge similar lines
    merged_lines = None
    # print 'ori lines',lines
    if lines is not None:
      pp = HoughBundler()
      pp_result = pp.process_lines(lines, black_mask3)
      merged_lines = np.array(pp_result)

    if merged_lines is not None:
      for line in merged_lines:
        point1 = (line[0][0],line[0][1])
        point2 = (line[1][0],line[1][1])

        cv2.line(edges_img,point1,point2,(0,255,0),3)

    for i in range(len(pts_src)):

        pt0 = (pts_src[i][0],pts_src[i][1])
        if i != len(pts_src)-1:
            pt1 = (pts_src[i+1][0],pts_src[i+1][1])
        else:
            pt1 = (pts_src[0][0],pts_src[0][1])
        cv2.line(edges_img,pt0,pt1,(0,255,0),3)



    # print lines
    # print "merged_result", merged_lines

    return edges_img, merged_lines

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2((line[0] - line[2]), (line[1] - line[3]))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    # print "###########distance", self.get_distance(line_old, line_new)

                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # print "#######angle", abs(orientation_new - orientation_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def merge_lines_pipeline_2(self, lines,min_distance_to_merge = 10,min_angle_to_merge = 10):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with; original paras: 30, 30
        # min_distance_to_merge = 10
        # min_angle_to_merge = 10
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing groups, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        # if 45 < orientation < 135:
        if 84 < orientation < 96:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img, min_distance_to_merge = 10, min_angle_to_merge = 10):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:

            orientation = self.get_orientation(line_i)
            # if vertical
            # if 45 < orientation < 135:
            if 84 < orientation < 96:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
          # print "line_x", lines_x
          # print "line_y", lines_y
          if len(i) > 0:
            groups = self.merge_lines_pipeline_2(i, min_distance_to_merge, min_angle_to_merge)
            merged_lines = []

            for group in groups:
              merged_lines.append(self.merge_lines_segments1(group))

            merged_lines_all.extend(merged_lines)

        return merged_lines_all

    def distance_to_line(self, point, line):
      """Get distance between point and line
      https://stackoverflow.com/questions/40970478/python-3-5-2-distance-from-a-point-to-a-line
      """
      px, py = point
      x1, y1, x2, y2 = line
      x_diff = x2 - x1
      y_diff = y2 - y1
      num = abs(y_diff * px - x_diff * py + x2 * y1 - y2 * x1)
      den = math.sqrt(y_diff**2 + x_diff**2)
      return num / den

    def get_distance(self, a_line, b_line):
      """Get all possible distances between each dot of two lines and second line
      return the shortest
      """
      dist1 = self.distance_to_line(a_line[:2], b_line)
      dist2 = self.distance_to_line(a_line[2:], b_line)
      dist3 = self.distance_to_line(b_line[:2], a_line)
      dist4 = self.distance_to_line(b_line[2:], a_line)
      # print "#asjdfja",dist1, dist2, dist3, dist4
      return min(dist1, dist2, dist3, dist4)

class CornerMatch:

    def __init__(self):
        self.size=[1280,720]

    def mainFuc(self, image):

        img_src = image

        #step1: color filter
        img_src1 = self.filter(img_src,'white')
        # cv2.imshow('color filter1',img_src1)

        img_src2 = self.filter(img_src,'green')

        # cv2.imshow('color filter2',img_src2)

        #step2: canny detection
        result_img_white = self.detect(img_src1)
        # cv2.imshow("canny detection", result_img)
        result_img_green = self.detect(img_src2)
        # cv2.imshow('canny detection green',result_img_green)
        result_img_white = self.get_white_line(result_img_white,img_src2) #get white line
        # cv2.imshow('canny detection white line',result_img_white)

        #step3: roi mask
        result_img2_white = self.ROI_mask(result_img_white,'white')
        # cv2.imshow('roi region white',result_img2_white)
        result_img2_green = self.ROI_mask(result_img_green,'green')
        # cv2.imshow('roi region green',result_img2_green)

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
        # if (lines_white is None) or (lines_green is None):
        #     continue

        if lines_white is None:
            averaged_lines_white = None
        else:
            averaged_lines_white = self.avg_lines(img_src, lines_white)              #Average the Hough lines as left or right lanes

        if lines_green is None:
            averaged_lines_green = None
        else:
            averaged_lines_green = self.avg_lines(img_src, lines_green)              #Average the Hough lines as left or right lanes

        combined_image = self.draw_lines(img_src, averaged_lines_white,
                                                    averaged_lines_green,5,
                                                    color1=[0, 0, 255],color2=[0,255,255]) #draw line for white zone and green zone
        # cv2.imshow('houghline transform',combined_image)

        white_vertex = self.get_intersection_point(averaged_lines_white)
        # print 'white vertex',white_vertex
        green_vertex = self.get_intersection_point(averaged_lines_green)
        # print 'green vertex',green_vertex
        return combined_image,white_vertex,green_vertex

    def detect(self,frame):
        blurr = cv2.GaussianBlur(frame, (5, 5), 0)
        imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
        imgC = cv2.morphologyEx(imgG, cv2.MORPH_CLOSE, (11, 11))
        imgC = cv2.Canny(imgC, 50, 60)
        imgC = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
        return imgC

    def ROI_mask(self,image,color):
        #add mask for roi
        height = image.shape[0]
        width = image.shape[1]

        #roi varies according to the detected colors
        # A rectangular polygon to segment the lane area and discarded other irrelevant parts in the image
        # Defined by three (x, y) coordinates
        if color == 'white':
            polygons = np.array([[(round(width)/2, round(height/8)), (round(width/2), round(height/2)), (round(width*3/4),round(height/2)), (round(width*3/4), round(height/8))]],dtype=np.int32)

            mask = np.zeros_like(image)
            cv2.fillPoly(mask, polygons, 255)  ## 255 is the mask color

            # Bitwise AND between canny image and mask image
            masked_image = cv2.bitwise_and(image, mask)

            return masked_image

        elif color == 'green':
            polygons = np.array([[(round(width)/2, round(height/8)), (round(width/2), round(height)), (round(width),round(height)), (round(width), round(height/8))]],dtype=np.int32)

            mask = np.zeros_like(image)
            cv2.fillPoly(mask, polygons, 255)  ## 255 is the mask color

            # Bitwise AND between canny image and mask image
            masked_image = cv2.bitwise_and(image, mask)

            return masked_image

        else:
            return image

    def get_coordinates(self,image, params):

        slope, intercept = params
        y1 = image.shape[0]
        y2 = int(y1 * (3/5)) # Setting y2 at 3/5th from y1
        x1 = int((y1 - intercept) / slope) # Deriving from y = mx + c
        x2 = int((y2 - intercept) / slope)

        if abs(slope) < 0.001:
            y1 = int(intercept)
            y2 = int(intercept)
            x1 = image.shape[1]
            x2 = int(x1 * (3/5))

        return np.array([x1, y1, x2, y2])

    # Returns averaged lines on left and right sides of the image
    def avg_lines(self,image, lines):

        left = []
        right = []

        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)

                print 'line', x1,y1,x2,y2
                # Fit polynomial, find intercept and slope
                params = np.polyfit((x1, x2), (y1, y2), deg = 1)
                slope = params[0]
                y_intercept = params[1]

                # print 'slope',slope
                # print 'y_intercept',y_intercept

                if slope < 0:
                    left.append((slope, y_intercept)) #Negative slope = left lane
                else:
                    right.append((slope, y_intercept)) #Positive slope = right lane

        # Avg over all values for a single slope and y-intercept value for each line

        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)

        # print 'lines',lines
        # print 'left',left_avg
        # print 'right',right_avg

        if len(left)==0 and len(right)==0:
            return np.array([])
        elif len(left)==0 and len(right)>0:
            right_line = self.get_coordinates(image, right_avg)
            return np.array([right_line])
        elif len(left)>0 and len(right)==0:
            left_line = self.get_coordinates(image, left_avg)
            return np.array([left_line])
        else:
            # Find x1, y1, x2, y2 coordinates for left & right lines
            left_line = self.get_coordinates(image, left_avg)
            right_line = self.get_coordinates(image, right_avg)
            return np.array([left_line, right_line])

    # Draws lines of given thickness over an image
    def draw_lines(self,image, lines1, lines2,thickness, color1, color2):

        # print(lines)
        line_image = np.zeros_like(image)
        # color=[0, 0, 255]


        if lines1 is not None:
            # print 'line',lines
            for x1, y1, x2, y2 in lines1:
                cv2.line(line_image, (x1, y1), (x2, y2), color1, thickness)
        if lines2 is not None:
            # print 'line',lines
            for x1, y1, x2, y2 in lines2:
                cv2.line(line_image, (x1, y1), (x2, y2), color2, thickness)

        # Merge the image with drawn lines onto the original.
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

        return combined_image

    def get_intersection_point(self,lines):
        #get two lines intersection point
        if lines is not None:
            if len(lines) == 2:
                intersection = su.line_intersect(lines[0][0],lines[0][1],
                                             lines[0][2],lines[0][3],
                                             lines[1][0],lines[1][1],
                                             lines[1][2],lines[1][3])
                return intersection
            else:
                return None
        else:
            return None

    def hsv_calc(self,frame):

        def nothing(x):
            pass

        cv2.namedWindow("Trackbars",)
        cv2.createTrackbar("lh","Trackbars",0,179,nothing)
        cv2.createTrackbar("ls","Trackbars",0,255,nothing)
        cv2.createTrackbar("lv","Trackbars",0,255,nothing)
        cv2.createTrackbar("uh","Trackbars",179,179,nothing)
        cv2.createTrackbar("us","Trackbars",255,255,nothing)
        cv2.createTrackbar("uv","Trackbars",255,255,nothing)
        while True:
            #frame = cv2.imread('candy.jpg')
            height, width = frame.shape[:2]
            #frame = cv2.resize(frame,(width/5, height/5), interpolation = cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("lh","Trackbars")
            ls = cv2.getTrackbarPos("ls","Trackbars")
            lv = cv2.getTrackbarPos("lv","Trackbars")
            uh = cv2.getTrackbarPos("uh","Trackbars")
            us = cv2.getTrackbarPos("us","Trackbars")
            uv = cv2.getTrackbarPos("uv","Trackbars")

            l_blue = np.array([lh,ls,lv])
            u_blue = np.array([uh,us,uv])
            mask = cv2.inRange(hsv, l_blue, u_blue)
            result = cv2.bitwise_or(frame,frame,mask=mask)

            cv2.imshow("result",result)
            cv2.imshow("mask",mask)
            key = cv2.waitKey(1)
            #press esc to exit
            if key == 27:
                break
        cv2.destroyAllWindows()

    def filter(self,image,color):
        blurr = cv2.GaussianBlur(image, (7, 7), 0)
        blurr_hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)

        #hsv color
        if color == 'green':
            # lowerG = (19,48,0)
            lowerG = (33,51,0)
            upperG = (84,255,255)

            # lowerB = (30,119,0)
            # upperB = (158,255,255)

            # maskB = cv2.inRange(blurr_hsv, lowerB, upperB)
            # maskB = cv2.GaussianBlur(maskB, (7, 7), 0)
            # maskB = cv2.morphologyEx(maskB, cv2.MORPH_CLOSE, np.ones((9, 9)))
            # maskB = cv2.morphologyEx(maskB, cv2.MORPH_OPEN, np.ones((9, 9)))

            maskG = cv2.inRange(blurr_hsv, lowerG, upperG)
            maskG = cv2.GaussianBlur(maskG, (5, 5), 0)
            maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, np.ones((7 ,7)))
            # maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, np.ones((7 ,7)))
            maskG1 = maskG.copy()

            h, w = image.shape[:2]
            mask_flood = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(maskG, mask_flood, (0,0), 255)
            maskG = cv2.bitwise_not(maskG)
            maskG2 = cv2.bitwise_or(maskG,maskG1)

            result = cv2.cvtColor(maskG2,cv2.COLOR_GRAY2BGR)
            # cv2.imshow("blue_mask",maskG2)

            # result = cv2.bitwise_or(image,image,mask=maskG2)
            # result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        elif color == 'white':
            lower = (46,0,106)
            upper = (102,255,255)
            mask = cv2.inRange(blurr_hsv, lower, upper)
            result = cv2.bitwise_or(image,image,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        elif color == 'red':
            lower = (0,0,0)
            upper = (16,251,180)
            mask = cv2.inRange(blurr_hsv, lower, upper)
            result = cv2.bitwise_or(image,image,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        elif color == 'blue':
            lower = (30,119,0)
            upper = (158,255,255)
            mask = cv2.inRange(blurr_hsv, lower, upper)
            result = cv2.bitwise_or(image,image,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        return result

    def get_white_line(self,img1,img2):

        ''' The function aims to clear edges between white and green
        '''
        kernel = np.ones((1,1),np.uint8)
        # erosion1 = cv2.erode(img,kernel,iterations = 1)
        img_dilate1 = cv2.dilate(img1,kernel,iterations = 1)

        #green mask
        lower = (10,10,10)
        upper = (255,255,255)
        mask1 = cv2.inRange(img2, lower, upper)
        kernel = np.ones((9,9),np.uint8)
        mask2 = cv2.dilate(mask1,kernel,iterations = 5)
        # cv2.imshow('mask',mask2)

        result_img1 = cv2.bitwise_and(img_dilate1, mask2)
        # cv2.imshow('and',result_img1)

        result_img2 = cv2.bitwise_xor(img_dilate1,result_img1)
        # cv2.imshow('or',result_img2)
        return result_img2

class CornerMatch_new:
    def __init__(self):
        self.size=[1280,720]

    def mainFuc(self, image):

        # skimage_src = img_as_float(image)
        # skimage = exposure.equalize_adapthist(skimage_src, clip_limit=0.01)
        # img_src = img_as_ubyte(skimage)

        img_src = image

        #step1: color filter
        img_white,mask_white = self.filter(img_src,'white')
        cv2.imshow('color white',img_white)
        img_green,mask_green = self.filter(img_src,'green')
        cv2.imshow('color green',img_green)
        # img_red,mask_red = self.filter(img_src,'red')
        # cv2.imshow('color red',img_red)
        img_blue,mask_blue = self.filter(img_src,'blue')
        _, mask_paper = self.filter(img_src, 'paper')
        cv2.imshow('paper mask',mask_paper)

        #step2: canny detection
        canny_img_green = self.detect(mask_green)
        canny_img_white = self.detect(mask_white)
        result_img_green = self.get_edge_lines(canny_img_green, mask_blue, mask_blue, 'top') # get close green line

        result_img_white = self.get_edge_lines(canny_img_white, mask_green, mask_blue, 'bottom') #get white line
        cv2.imshow('canny detection green',result_img_green)
        cv2.imshow('canny detection white line',result_img_white)


        #step3: roi mask
        result_img2_white = self.ROI_mask(result_img_white,mask_paper)
        cv2.imshow('roi region white',result_img2_white)
        result_img2_green = self.ROI_mask(result_img_green,mask_paper)
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
        # if (lines_white is None) or (lines_green is None):
        #     continue

        if lines_green is None:
            averaged_lines_green = None
        else:
            # averaged_lines_green = self.avg_lines(img_src, lines_green)              #Average the Hough lines as left or right lanes
            a = HoughBundler()
            print "green lines", lines_green
            # previous: 140,50
            merged_lines_green = a.process_lines(lines_green, result_img2_green, min_distance_to_merge =120, min_angle_to_merge = 55)
            out = np.empty(shape=[0, 4])
            for line in merged_lines_green:
                out = np.append(out,[[line[0][0], line[0][1], line[1][0], line[1][1]]],axis=0)
            averaged_lines_green = out.astype(int)
            # print "line green",averaged_lines_green

        if lines_white is None:
            averaged_lines_white = None
        else:
            averaged_lines_white = self.avg_lines(img_src, lines_white)              #Average the Hough lines as left or right lanes
            # print "line white",averaged_lines_white
        combined_image = self.draw_lines(img_src, averaged_lines_white,
                                        averaged_lines_green,5,
                                        color1=[0, 0, 255],color2=[0,255,255]) #draw line for white zone and green zone
        cv2.imshow('houghline transform',combined_image)

        white_vertex = self.get_intersection_point(averaged_lines_white)
        # print 'white vertex',white_vertex
        green_vertex = self.get_intersection_point(averaged_lines_green)
        # print 'green vertex',green_vertex
        return combined_image,white_vertex,green_vertex

    def detect(self,frame):
        imgG = cv2.GaussianBlur(frame, (5, 5), 0)
        # imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
        imgC = cv2.morphologyEx(imgG, cv2.MORPH_CLOSE, (11, 11))
        imgC = cv2.Canny(imgC, 50, 60)
        imgC = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (3, 3))
        return imgC

    def ROI_mask(self,image,backgound_mask):
        #add mask for roi
        height = image.shape[0]
        width = image.shape[1]

        #roi varies according to the detected colors
        mask = cv2.GaussianBlur(backgound_mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (9, 9))

        # Bitwise AND between canny image and mask image
        masked_image = cv2.bitwise_and(image, mask)


        return masked_image

    def get_coordinates(self,image, params):

        slope, intercept = params
        y1 = image.shape[0]
        y2 = int(y1 * (3/5)) # Setting y2 at 3/5th from y1
        x1 = int((y1 - intercept) / slope) # Deriving from y = mx + c
        x2 = int((y2 - intercept) / slope)

        if abs(slope) < 0.001:
            y1 = int(intercept)
            y2 = int(intercept)
            x1 = image.shape[1]
            x2 = int(x1 * (3/5))

        return np.array([x1, y1, x2, y2])

    # Returns averaged lines on left and right sides of the image
    def avg_lines(self,image, lines):

        left = []
        right = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            # Fit polynomial, find intercept and slope
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            y_intercept = params[1]

            # print 'slope',slope
            # print 'y_intercept',y_intercept

            if slope < 0:
                left.append((slope, y_intercept)) #Negative slope = left lane
            else:
                right.append((slope, y_intercept)) #Positive slope = right lane

        # Avg over all values for a single slope and y-intercept value for each line

        left_avg = np.average(left, axis = 0)
        right_avg = np.average(right, axis = 0)

        # print 'lines',lines
        # print 'left',left_avg
        # print 'right',right_avg

        if len(left)==0 and len(right)==0:
            return np.array([])
        elif len(left)==0 and len(right)>0:
            right_line = self.get_coordinates(image, right_avg)
            return np.array([right_line])
        elif len(left)>0 and len(right)==0:
            left_line = self.get_coordinates(image, left_avg)
            return np.array([left_line])
        else:
            # Find x1, y1, x2, y2 coordinates for left & right lines
            left_line = self.get_coordinates(image, left_avg)
            right_line = self.get_coordinates(image, right_avg)
            return np.array([left_line, right_line])

    # Draws lines of given thickness over an image
    def draw_lines(self,image, lines1, lines2,thickness, color1, color2):

        # print(lines)
        line_image = np.zeros_like(image)
        # color=[0, 0, 255]


        if lines1 is not None:
            # print 'line',lines
            for x1, y1, x2, y2 in lines1:
                cv2.line(line_image, (x1, y1), (x2, y2), color1, thickness)
        if lines2 is not None:
            # print 'line',lines
            for x1, y1, x2, y2 in lines2:
                cv2.line(line_image, (x1, y1), (x2, y2), color2, thickness)

        # Merge the image with drawn lines onto the original.
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

        return combined_image

    def get_intersection_point(self,lines):
        #get two lines intersection point
        if lines is not None:
            if len(lines) == 2:
                intersection = su.line_intersect(lines[0][0],lines[0][1],
                                                 lines[0][2],lines[0][3],
                                                 lines[1][0],lines[1][1],
                                                 lines[1][2],lines[1][3])
                return intersection
            else:
                return None
        else:
            return None

    def hsv_calc(self,frame):

        def nothing(x):
            pass

        cv2.namedWindow("Trackbars",)
        cv2.createTrackbar("lh","Trackbars",0,179,nothing)
        cv2.createTrackbar("ls","Trackbars",0,255,nothing)
        cv2.createTrackbar("lv","Trackbars",0,255,nothing)
        cv2.createTrackbar("uh","Trackbars",179,179,nothing)
        cv2.createTrackbar("us","Trackbars",255,255,nothing)
        cv2.createTrackbar("uv","Trackbars",255,255,nothing)
        while True:
            #frame = cv2.imread('candy.jpg')
            height, width = frame.shape[:2]
            #frame = cv2.resize(frame,(width/5, height/5), interpolation = cv2.INTER_CUBIC)
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("lh","Trackbars")
            ls = cv2.getTrackbarPos("ls","Trackbars")
            lv = cv2.getTrackbarPos("lv","Trackbars")
            uh = cv2.getTrackbarPos("uh","Trackbars")
            us = cv2.getTrackbarPos("us","Trackbars")
            uv = cv2.getTrackbarPos("uv","Trackbars")

            l_blue = np.array([lh,ls,lv])
            u_blue = np.array([uh,us,uv])
            mask = cv2.inRange(hsv, l_blue, u_blue)
            result = cv2.bitwise_or(frame,frame,mask=mask)

            cv2.imshow("result",result)
            cv2.imshow("mask",mask)
            key = cv2.waitKey(1)
            #press esc to exit
            if key == 27:
                break
        cv2.destroyAllWindows()

    def filter(self,image,color):
        blurr = cv2.GaussianBlur(image, (7, 7), 0)
        blurr_hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)
        #hsv color
        if color == 'green':
            # lowerG = (19,48,0)
            lowerG = (24,0,39)
            upperG = (96,255,255)
            maskG = cv2.inRange(blurr_hsv, lowerG, upperG)

            maskG = cv2.GaussianBlur(maskG, (5, 5), 0)
            mask = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, np.ones((19 ,19)))
            result = cv2.bitwise_and(blurr_hsv,blurr_hsv,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        elif color == 'white':

            _,maskGW = self.filter(image, 'paper')
            lowerG = (24,0,39)
            upperG = (96,255,255)
            maskG = cv2.inRange(blurr_hsv, lowerG, upperG)
            maskNoG = cv2.bitwise_not(maskG)
            mask = cv2.bitwise_and(maskGW,maskNoG)

            # cv2.imshow('maskGW', maskGW)
            # cv2.imshow('maskW', mask)

            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9 ,9)))
            result = cv2.bitwise_or(blurr_hsv,blurr_hsv,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        elif color == 'red':
            lowerR = (0,119,0)
            upperR = (60,255,255)
            mask = cv2.inRange(blurr_hsv, lowerR, upperR)
            result = cv2.bitwise_or(blurr_hsv,blurr_hsv,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        elif color == 'blue':
            lowerB = (19,81,0)
            upperB = (168,255,255)

            kernel = np.ones((5,3),np.uint8)
            mask = cv2.inRange(blurr_hsv, lowerB, upperB)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7 ,7)))
            mask = cv2.dilate(mask,kernel,iterations = 3)

            result = cv2.bitwise_or(blurr_hsv,blurr_hsv,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        elif color == 'paper':

            # lowerRB = (0, 85, 0)
            # upperRB = (179,255,255)

            lowerRB = (0, 69, 0)
            upperRB = (179,255,255)


            maskRB = cv2.inRange(blurr_hsv, lowerRB, upperRB)
            mask = cv2.bitwise_not(maskRB)

            lcc = self.largestConnectComponent(mask)
            lcc = np.asarray(lcc, dtype="uint8")

            mask = lcc
            result = cv2.bitwise_or(blurr_hsv,blurr_hsv,mask=mask)
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

        return result, mask

    def largestConnectComponent(self,bw_image):
        '''
        compute largest Connect component of an labeled image
        Parameters:
        ---
        bw_image:
            grey image in cv format
        Example:
        ---
            >>> lcc = largestConnectComponent(bw_img)
        '''
        bw_img = img_as_float(bw_image)
        thresh = threshold_otsu(bw_img)
        binary = bw_image > thresh

        labeled_img, num = label(binary, neighbors=4, background=0, return_num=True)
        # plt.figure(), plt.imshow(labeled_img, 'gray')
        max_label = 0
        max_num = 0
        for i in range(1, num): # Start from 1 here to prevent the background from being set to the largest connected domain
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
        cv_image = img_as_ubyte(lcc)
        return cv_image

    def get_edge_lines(self,canny_img1,mask_img1,mask_img2, layer):

        ''' The function aims to clear edges between white and green
        ----- input paras:
        canny_img1: the canny image of target layer
        mask_img1: the binary mask of the neighbour
        mask_img2: the binary mask of the finger
        layer: either 'top' or 'bottom', top means include the close region to mask, bottom as exclude
        '''

        if layer == 'bottom':
            kernel1 = np.ones((13,13),np.uint8)
            mask1 = cv2.dilate(mask_img1,kernel1,iterations = 6)
            result_img1 = cv2.bitwise_and(canny_img1, mask1)
            result_img1 = cv2.bitwise_xor(canny_img1,result_img1)

            kernel1 = np.ones((25,25),np.uint8)
            mask2 = cv2.dilate(mask_img2,kernel1,iterations = 11)
            result_img = cv2.bitwise_and(result_img1, mask2)

            # cv2.imshow("result img1", result_img1)
            # cv2.imshow("overall", result_img)
            # cv2.imshow('mask2', mask2)

        elif layer == 'top':
            kernel = np.ones((25,25),np.uint8)
            mask1 = cv2.dilate(mask_img2,kernel,iterations = 5)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((11 ,11)))
            mask2 = cv2.dilate(mask_img2,np.ones((5,5),np.uint8),iterations = 4)

            result_img1 = cv2.bitwise_and(canny_img1, mask1)
            result_img2 = cv2.bitwise_and(result_img1, mask2)
            result_img = cv2.bitwise_xor(result_img1,result_img2)

            result_img =cv2.dilate(result_img,np.ones((2,2),np.uint8),iterations=4)
            result_img = cv2.morphologyEx(result_img, cv2.MORPH_OPEN, np.ones((2,2)))

            # for i in range(5):
            #     _, contours, _ = cv2.findContours(result_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #     if contours is not None:
            #         hull = cv2.convexHull(contours[0])
            #         cv2.drawContours(result_img, [hull], 0, 255, 1)

        return result_img

class Predictor:
    # predict the next state
    # 1) next pts_src (used in class GetTrans)
    # 2) next top color (used for class CornerMatch)
    def __init__(self,pts_src,crease,original_image,step=0):

        self.pts_src = pts_src
        # the creases are stored by folding sequence, folded creases are deleted. e.g creases[0] is the first-folded crease
        # crease direction is given by opencv
        self.crease = crease
        self.state = {}

        #get all facet information
        contour_image = self.get_facets_info(original_image,step)
        # cv2.imshow('contour image',contour_image)

    def crease_update(self,new_crease):
        # update crease info
        self.crease = new_crease

    def get_facets_info(self,image,step):
        # get all facets in the image, implemented by opencv polygon detection

        if step == 0:
            # get all facets information at the first step
            facet_pts = {}
            facet_colors = {}

            #step1: get facets contours (findContours)
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image,contours,-1,(0,0,255),3)
            print 'contour nums',len(contours)
            # print 'contour', contours[1]
            # print 'hierarchy',hierarchy

            #step2: get and store points of each contour
            for i in range(1,len(contours)):
                cnt_temp = contours[i].reshape(len(contours[i]),2)
                facet_pts.setdefault(str(i),cnt_temp)
                facet_colors.setdefault(str(i),0)
            state1 = {'facet_pts':copy.deepcopy(facet_pts),'facet_colors':copy.deepcopy(facet_colors)}
            self.state.setdefault('state1',state1)
            # print 'state1',self.state
            return image

        else:
            #update the folded facets information

            #step1: find facets on the left side of the current state
            crease = copy.deepcopy(self.crease)
            state = 'state'+str(step)
            facet_pts = copy.deepcopy(self.state[state]['facet_pts'])
            left_facets,_ = ut.get_side_facets(crease,facet_pts)

            #step2: reverse all points on the left facets, update color information
            for facet in left_facets:
                pts = copy.deepcopy(self.state[state]['facet_pts'][facet])
                reversed_pts = []
                for pt in pts:
                    reversed_pt = ut.reversePoint(crease,pt)
                    reversed_pts.append(reversed_pt)
                self.state[state]['facet_pts'][facet] = reversed_pts

                colors =copy.deepcopy(self.state[state]['facet_colors'][facet])
                self.state[state]['facet_colors'][facet] = colors+1

            #step3: new contour image
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image,contours,-1,(0,0,255),3)
            return image

    def get_new_contour(self):
        # get the new paper contour after folding
        crease = copy.deepcopy(self.crease)
        pts = copy.deepcopy(self.pts_src)
        a,b,c = ut.lineToFunction(crease)

        #step1: get pts at the left of the crease
        left_pts = []
        right_pts = []
        for pt in pts:
            product = a*pt[0]+b*pt[1]+c
            if product<0:
                left_pts.append(pt)
            elif product>0:
                right_pts.append(pt)
            elif product==0:
                left_pts.append(pt)
                right_pts.append(pt)

        # print 'left_pts',left_pts
        # print 'right pts',right_pts

        #step2: reverse the left pts
        reversed_left_pts = []
        for pt in left_pts:
            reversed_pt = ut.reversePoint(crease,pt)
            reversed_left_pts.append(reversed_pt)
        # print 'reversed left pts',reversed_left_pts

        #step3: compare the two pts set and determine the new contour
        poly1 = Polygon(right_pts)
        poly2 = Polygon(reversed_left_pts)
        polygon = [poly1,poly2]
        contour = cascaded_union(polygon)
        # print 'countour',contour
        # boundary = gpd.GeoSeries(cascaded_union(polygon))
        # boundary.plot(color = 'red')
        # plt.show()
        new_pts_src = np.array(contour.exterior.coords)
        new_pts_src = np.array(list(set([tuple(t) for t in new_pts_src])))
        # print 'new pts src',new_pts_src

        return new_pts_src

    def get_colors(self,step,reflect):
        #get colors for corner matching, return the upper color and lower color

        #step1: find facet on the right side of the crease (fixed facets)
        crease = copy.deepcopy(self.crease)
        state = 'state'+str(step+1)
        facet_pts = copy.deepcopy(self.state[state]['facet_pts'])
        _,right_facets = ut.get_side_facets(crease,facet_pts)

        #step2: get the color of lower paper (fixed paper)
        #assume there is only one color
        right_facet_colors = copy.deepcopy(self.state[state]['facet_colors'])
        color_max = 0
        for facet in right_facet_colors.keys():
            color = right_facet_colors[facet]
            if color>=color_max:
                color_max=color
        if color_max % 2 == 0:
            lower_color = 'white'
        else:
            lower_color = 'green'

        #step3: get the color of upper paper, this needs to be further modified
        if reflect == 0:
            upper_color = 'green'
        if reflect == 1:
            upper_color = 'white'

        return lower_color,upper_color
