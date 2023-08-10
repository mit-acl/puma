#!/usr/bin/env python
# subscribe to T265's raw images and distort it and run it on FastSAM, and get the mean of the blobs
# Author: Kota Kondo
# Atribution: Jouko Kinnari (https://gitlab.com/mit-acl/clipper/uncertain-localization/depth-estimation-experiments/)

import os
import argparse
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import rospy
import rospkg
import numpy as np
from utils import compute_blob_mean_and_covariance, compute_3d_position_of_each_centroid
import motlee_msgs.msg as motlee_msgs
from termcolor import colored
import message_filters
import skimage
from FastSAM.fastsam import *


class FastSAM_ROS:

    def __init__(self):

        # get ROS params
        self.is_sim = rospy.get_param('~is_sim', True)

        # set up sim/hw-specific params
        if self.is_sim:
            self.camera_name_topic = "camera/rgb"
            self.camera = "sim_camera"
            self.world_name_topic = "goal"
        else:
            self.camera_name_topic = "t265/fisheye1"
            self.camera = rospy.get_param('~camera', "t265_fisheye1")
            self.world_name_topic = "world"

        # get undistortion params
        self.get_undistortion_params()
        
        # set up FastSAM
        rospack = rospkg.RosPack()
        path_to_fastsam = rospack.get_path('primer') + '/scripts/models/FastSAM-x.pt'
        print("path_to_fastsam: ", path_to_fastsam)
        self.fastSamModel = FastSAM(path_to_fastsam)
        
        # define FastSAM params
        self.conf = 0.5
        self.iou = 0.9
        self.DEVICE = 'cuda'
        self.MAX_BLOB_SIZE = 4000.0
        self.MIN_BLOB_SIZE = 0.0

        # define mapper covariances
        # note: covariance[0] and covariance[4] = 0.01 from Kota's experiments (see the Kota Update google slide Aug 2023)
        # note: covariance[8] = 0.0 because of 2D assumption
        self.mapper_covariance = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])

        # set up ROS communications
        subs = []
        subs.append(message_filters.Subscriber(rospy.get_namespace()+'/'+self.world_name_topic, PoseStamped, queue_size=100))
        subs.append(message_filters.Subscriber(f'{self.camera_name_topic}/image_raw', Image, queue_size=1)) # we only need the most recent image
        self.ats = message_filters.ApproximateTimeSynchronizer(subs, queue_size=100, slop=.05)
        self.ats.registerCallback(self.fastsam_cb)
        self.pub_objarray = rospy.Publisher('detections', motlee_msgs.ObjArray, queue_size=1)

    # get undistortion params
    def get_undistortion_params(self):
        # get camera matrix and distortion coefficients 
        # for realsense
        print(colored("getting undistortion params", 'yellow'))
        msg = rospy.wait_for_message(f"{self.camera_name_topic}/camera_info", CameraInfo, timeout=None)
        print(colored("got undistortion params", 'yellow'))
        self.K = np.array(msg.K).reshape(3,3)
        self.D = np.array(msg.D)
        self.R = np.array(msg.R).reshape(3,3)
        self.P = np.array(msg.P).reshape(3,4)



    # function for image processing
    def fastsam_cb(self, pose_msg, img):

        # convert img to cv2_img
        bridge = CvBridge()
        # note: FastSAM expects RGB images (https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/Inference.py)
        # reference for encoding: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython 
        cv_img = bridge.imgmsg_to_cv2(img, desired_encoding="rgb8")

        # undistort cv2_img
        # undistorted_img = self.undistort_image(cv_img)
        undistorted_img = cv_img

        ### debug
        # show undistorted_img
        print("show undistorted_img")
        cv2.imshow("undistorted_img", undistorted_img)
        print("after imshow")
        cv2.waitKey(1)
        #
        # get undistorted_img from file
        # image_bgr = cv2.imread("/media/kota/T7/frame/pngs-csvs/test4-partial/pngs/undistorted_images/t265_fisheye1/frame000500_undistorted.png")
        # print("image_bgr.shape: ", image_bgr.shape)
        # undistorted_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ###

        # get grayscale image for visualization
        # Let's also make a 1-channel grayscale image
        # image_gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2RGB)
        # print("image_gray.shape: ", image_gray.shape)
        # ...and also a 3-channel RGB image, which we will eventually use for showing FastSAM output on
        # image_gray_rgb = np.stack((image_gray,)*3, axis=-1)
        # print("image_gray_rgb.shape: ", image_gray_rgb.shape)

        # run FastSAM on cv2_img
        everything_results = self.fastSamModel(undistorted_img, device=self.DEVICE, retina_masks=True, imgsz=1024, conf=self.conf, iou=self.iou,)
        prompt_process = FastSAMPrompt(undistorted_img, everything_results, device=self.DEVICE)
        segmask = prompt_process.everything_prompt()

        # get centroid of segmask
        blob_means = self.get_blob_means(segmask, undistorted_img)

        # map the blob_means to the world frame
        positions = self.map_to_world(pose_msg, blob_means)

        # publish the centroid
        self.publish_objarray(positions)
    
    # undistort image
    def undistort_image(self, cv_img):
        h,  w = cv_img.shape[:2]
        scale_factor = 1.0
        img_dim_out =(int(w*scale_factor), int(h*scale_factor))

        # OpenCV fisheye calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        nK = self.K.copy()
        nK[0,0] = self.K[0,0] * 0.3
        nK[1,1] = self.K[1,1] * 0.3

        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, self.R, nK, img_dim_out, cv2.CV_32FC1)
        undistorted_img = cv2.remap(cv_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img
    
    # get the blob means and covs
    def get_blob_means(self, segmask, undistorted_img):
        
        # initialize blob means
        blob_means = []

        # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
        if (len(segmask) > 0):
            segmask = segmask.cpu().numpy()
        else:
            segmask = None

        if (segmask is not None):
            # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
            # of the detected segment
            [numMasks, h, w] = segmask.shape

            # Prepare a mask of IDs where each pixel value corresponds to the mask ID
            segmasks_flat = np.zeros((h,w),dtype=int)

            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segmask[maskId,:,:]

                # From the pixel coordinates of the mask, compute centroid and a covariance matrix
                blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
                
                # if blob is too small or too big, skip
                if abs(blob_cov[0,0]) > self.MAX_BLOB_SIZE or abs(blob_cov[1,1]) > self.MAX_BLOB_SIZE or abs(blob_cov[0,1]) > self.MAX_BLOB_SIZE \
                    or abs(blob_cov[0,0]) < self.MIN_BLOB_SIZE or abs(blob_cov[1,1]) < self.MIN_BLOB_SIZE or abs(blob_cov[0,1]) < self.MIN_BLOB_SIZE:
                    # print("blob too small or too big")
                    continue

                # Store centroids and covariances in lists
                blob_means.append(blob_mean)

                # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
                # segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

                # Using skimage, overlay masked images with colors
                # undistorted_img = skimage.color.label2rgb(segmasks_flat, undistorted_img)
            
            # cv2.imshow("undistorted_img", undistorted_img)
            # cv2.waitKey(10)
    
        return blob_means

    # map the blob_means to the world frame
    def map_to_world(self, pose_msg, blob_means):
        pose = [] # x, y, z, qx, qy, qz, qw
        pose.append(pose_msg.pose.position.x)
        pose.append(pose_msg.pose.position.y)
        pose.append(pose_msg.pose.position.z)
        # pose.append(2.0) # just for testing 
        pose.append(pose_msg.pose.orientation.x)
        pose.append(pose_msg.pose.orientation.y)
        pose.append(pose_msg.pose.orientation.z)
        pose.append(pose_msg.pose.orientation.w)
        print("pose: ", pose)
        return compute_3d_position_of_each_centroid(blob_means, pose, camera=self.camera, K=self.K)

    # publish the centroid
    def publish_objarray(self, positions):

        # create ObjArray
        objarray = motlee_msgs.ObjArray()

        for positoin in positions:

            # create Obj
            obj = motlee_msgs.Obj()
            
            # set obj position
            obj.position.x = positoin[0]
            obj.position.y = positoin[1]
            obj.position.z = 0.0 # flat ground assumption

            # set obj covariance
            obj.covariance = self.mapper_covariance 

            # append obj to objarray
            objarray.objects.append(obj)

        # publish ObjArray
        objarray.header.stamp = rospy.Time.now()
        objarray.header.frame_id = rospy.get_namespace()
        self.pub_objarray.publish(objarray)

    # shutdown callback
    def shutdown_callback(self):
        print(colored("shutting down", 'red'))
        
if __name__ == '__main__':
    rospy.init_node('fastsam')
    fastsam = FastSAM_ROS()
    rospy.spin()