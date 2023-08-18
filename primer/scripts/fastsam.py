#!/usr/bin/env python
# subscribe to T265's raw images and distort it and run it on FastSAM, and get the mean of the blobs
# Author: Kota Kondo
# Atribution: Jouko Kinnari (https://gitlab.com/mit-acl/clipper/uncertain-localization/depth-estimation-experiments/)

import os
import argparse
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import Goal
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
from utils import get_quaternion_from_euler, plotErrorEllipse
import matplotlib.pyplot as plt

class FastSAM_ROS:

    def __init__(self):

        # define params
        self.SAVE_IMAGES = rospy.get_param('~save_images', False)

        # get ROS params
        self.is_sim = rospy.get_param('~is_sim', True)

        # set up sim/hw-specific params
        if self.is_sim:
            # asus camera
            # self.camera_name_topic = "camera/rgb"
            # self.camera = "sim_camera"

            # t265 camera
            self.camera_name_topic = "camera/fisheye1"
            self.camera = rospy.get_param('~camera', "t265_fisheye1")
            self.world_name_topic = "goal" # if you use perfect_tracker in primer, "world" won't be published.
        else:
            # self.camera_name_topic = "t265/fisheye1"
            self.camera_name_topic = "camera/fisheye1"
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
        
        # the object covariance is stretched along the range axis. Here we stretch the x-axis, but we'll rotate 
        # the covariance so the x-axis points from ego position to the object position for each object.
        # We keep 3D because this will naturally result in larger covariances in objects viewed at a similar elevation
        # as the drone.
        # Assume both non-range axes covariances are the same.
        range_std_dev = .5 # 50 cm std dev along range axis
        non_range_std_dev = .2 # 20 cm std dev along other axes (non-range)
        self.mapper_covariance = np.diag([range_std_dev**2, non_range_std_dev**2, non_range_std_dev**2])

        # static variables for image numbering
        self.image_number = 0

        # set up ROS communications
        if self.is_sim:
            # if you use sim (perfect_tracker), "world" won't be published and we need to use "goal" that is published sporadically. So we store those values as world_pose to sync with the image.            rospy.Subscriber(rospy.get_namespace()+'/'+self.world_name_topic, Goal, self.store_world_pose)
            rospy.Subscriber(rospy.get_namespace()+'/'+self.world_name_topic, Goal, self.store_world_pose)
            self.world_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            rospy.Subscriber(f'{self.camera_name_topic}/image_raw', Image, self.store_image)
            self.img = None
            self.pub_world = rospy.Publisher('world', PoseStamped, queue_size=1)
        else: # hw
            subs = []
            subs.append(message_filters.Subscriber(rospy.get_namespace()+'/'+self.world_name_topic, PoseStamped, queue_size=10))
            subs.append(message_filters.Subscriber(f'{self.camera_name_topic}/image_raw', Image, queue_size=1)) # we only need the most recent image
            self.ats = message_filters.ApproximateTimeSynchronizer(subs, queue_size=100, slop=.05)
            self.ats.registerCallback(self.fastsam_cb)

        self.pub_objarray = rospy.Publisher('detections', motlee_msgs.ObjArray, queue_size=1)

    # for simulation, we need to store the image since /goal is not published all the time
    def store_image(self, img):
        self.img = img
        if self.world_pose != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            self.fastsam_cb(self.world_pose, self.img)

    # for simulation, we need to store the pose of the world since /goal is not published all the time
    def store_world_pose(self, pose_goal_msg):
        self.world_pose[0] = pose_goal_msg.p.x
        self.world_pose[1] = pose_goal_msg.p.y
        self.world_pose[2] = pose_goal_msg.p.z
        quaternion = get_quaternion_from_euler(0.0, 0.0, pose_goal_msg.psi)
        self.world_pose[3] = quaternion.x
        self.world_pose[4] = quaternion.y
        self.world_pose[5] = quaternion.z
        self.world_pose[6] = quaternion.w

    # get undistortion params
    def get_undistortion_params(self):
        # get camera matrix and distortion coefficients
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
        if self.is_sim:
            undistorted_img = cv_img # no need to undistort in sim
        else:
            undistorted_img = cv_img
            # undistorted_img = self.undistort_image(cv_img)

        ### debug
        # show undistorted_img
        # print("show undistorted_img")
        # cv2.imshow("undistorted_img", undistorted_img)
        # print("after imshow")
        # cv2.waitKey(1)
        #
        # save undistorted_img with the file name undistorted_{frame_number}.png
        if self.SAVE_IMAGES:
            print("save undistorted_img")
            cv2.imwrite(f"/media/kota/T7/frame/tmp/undistorted_images/undistorted_{self.image_number}.png", undistorted_img)
        #
        # get undistorted_img from file
        # image_bgr = cv2.imread("/media/kota/T7/frame/pngs-csvs/test4-partial/pngs/undistorted_images/t265_fisheye1/frame000500_undistorted.png")
        # print("image_bgr.shape: ", image_bgr.shape)
        # undistorted_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ###

        # get grayscale image for visualization (segmented images)
        # Let's also make a 1-channel grayscale image
        if self.SAVE_IMAGES:
            image_gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
            image_gray_rgb = np.stack((image_gray,)*3, axis=-1)
        # print("image_gray_rgb.shape: ", image_gray_rgb.shape)
        # print("image_gray.shape: ", image_gray.shape)

        # run FastSAM on cv2_img
        everything_results = self.fastSamModel(undistorted_img, device=self.DEVICE, retina_masks=True, imgsz=256, conf=self.conf, iou=self.iou,)
        prompt_process = FastSAMPrompt(undistorted_img, everything_results, device=self.DEVICE)
        segmask = prompt_process.everything_prompt()

        # get centroid of segmask
        blob_means = self.get_blob_means(segmask, image_gray_rgb) if self.SAVE_IMAGES else self.get_blob_means(segmask, undistorted_img) 

        # map the blob_means to the world frame
        positions = self.map_to_world(pose_msg, blob_means)

        # publish the centroid
        self.publish_objarray(pose_msg, positions)
    
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
        blob_covs = []

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
                blob_covs.append(blob_cov)

                if self.SAVE_IMAGES:
                    # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
                    segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

                    # Using skimage, overlay masked images with colors
                    undistorted_img = skimage.color.label2rgb(segmasks_flat, undistorted_img)

            if self.SAVE_IMAGES:
                # Create a matplotlib figure to plot image and ellipsoids on.
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                for m, c in zip(blob_means, blob_covs):
                    plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

                # save image
                ax.imshow(undistorted_img)
                ax.set_xlim(0, undistorted_img.shape[1])
                ax.set_ylim(undistorted_img.shape[0], 0)
                fig.savefig(f"/media/kota/T7/frame/tmp/segmented_images/segmented_{self.image_number}.png")
                plt.close(fig)

        self.image_number += 1

        return blob_means

    # publish pose
    def publish_pose(self, pose):
        pub_pose_msg = PoseStamped()
        pub_pose_msg.header.stamp = rospy.Time.now()
        pub_pose_msg.header.frame_id = "world"
        pub_pose_msg.pose.position.x = pose[0]
        pub_pose_msg.pose.position.y = pose[1]
        pub_pose_msg.pose.position.z = pose[2]
        pub_pose_msg.pose.orientation.x = pose[3]
        pub_pose_msg.pose.orientation.y = pose[4]
        pub_pose_msg.pose.orientation.z = pose[5]
        pub_pose_msg.pose.orientation.w = pose[6]
        self.pub_world.publish(pub_pose_msg)

    # map the blob_means to the world frame
    def map_to_world(self, pose_msg, blob_means):

        if not self.is_sim:
            pose = [] # x, y, z, qx, qy, qz, qw
            pose.append(pose_msg.pose.position.x)
            pose.append(pose_msg.pose.position.y)
            pose.append(pose_msg.pose.position.z)
            # pose.append(2.0) # just for testing 
            pose.append(pose_msg.pose.orientation.x)
            pose.append(pose_msg.pose.orientation.y)
            pose.append(pose_msg.pose.orientation.z)
            pose.append(pose_msg.pose.orientation.w)
        else: # if sim (in sim pose_msg is actually a list)
            pose = pose_msg.copy()
            # publish pose (because goal is only sporadically published, we need to republish it)
            self.publish_pose(pose)

        print("pose: ", pose)

        return compute_3d_position_of_each_centroid(blob_means, pose, camera=self.camera, K=self.K)

    # publish the centroid
    def publish_objarray(self, pose_msg, positions):
        if not self.is_sim:
            ego_position = np.array([pose_msg.pose.position.x,
                                    pose_msg.pose.position.y,
                                    pose_msg.pose.position.z])
        else:
            ego_position = np.array(pose_msg[:3])

        # create ObjArray
        objarray = motlee_msgs.ObjArray()

        for position in positions:

            # create Obj
            obj = motlee_msgs.Obj()
            
            # set obj position
            obj.position.x = position[0]
            obj.position.y = position[1]
            obj.position.z = 0.0 # flat ground assumption

            # build object covariance - 
            # previously stretched x axis will point toward object from ego pose
            obj_position = np.array([position[0], position[1], 0.])
            cov_R_x = (obj_position - ego_position) / np.linalg.norm(obj_position - ego_position)
            # y and z axis just need to be orthonormal (symmetric about x axis)
            # first check if R_x and z are parallel to compute the second axis
            if np.allclose(np.cross(cov_R_x, np.array([0., 0., 1.])), np.zeros(3)):
                cov_R_y_nonunit = np.cross(cov_R_x, np.array([0., 1., 0.]))
            else:
                cov_R_y_nonunit = np.cross(cov_R_x, np.array([0., 0., 1.]))
            cov_R_y = cov_R_y_nonunit / np.linalg.norm(cov_R_y_nonunit)
            cov_R_z = np.cross(cov_R_x, cov_R_y)
            cov_R = np.c_[cov_R_x, cov_R_y, cov_R_z]

            obj.covariance = (cov_R @ self.mapper_covariance @ cov_R.T).reshape(-1)

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