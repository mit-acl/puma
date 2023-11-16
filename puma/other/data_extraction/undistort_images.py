#!/usr/bin/env python
# undistort images

import os
import argparse
import cv2
import numpy as np
import rosbag
from PIL import Image

def undistort_images_voxl(input_dir, output_dir, K, D):
    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # undistort images
    for filename in img_list:
        img = cv2.imread(os.path.join(input_dir, filename))
        h,  w = img.shape[:2]
        scale_factor = 1.0 
        balance = 1.0
        img_dim_out =(int(w*scale_factor), int(h*scale_factor))

        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img_dim_out, np.eye(3), balance=balance)
        # print("K_new: ", K_new)
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, img_dim_out, cv2.CV_32FC1)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        output_filename = filename[:-4] + '_undistorted.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), undistorted_img)

def undistort_images_rs(input_dir, output_dir, K, D, R, P):

    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # undistort images
    for filename in img_list:
        img = cv2.imread(os.path.join(input_dir, filename))
        h,  w = img.shape[:2]
        scale_factor = 1.0
        balance = 1.0
        img_dim_out =(int(w*scale_factor), int(h*scale_factor))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        nK = K.copy()
        nK[0,0] = K[0,0] * 0.3
        nK[1,1] = K[1,1] * 0.3

        # print("nK - need in compute_3d_position_of_centroid(): ", nK)

        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        output_filename = filename[:-4] + '_undistorted.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), undistorted_img)

def pad_images(input_dir, output_dir):

    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # pad images
    for filename in img_list:
        img = Image.open(os.path.join(input_dir, filename))
        width, height = img.size
        
        right = 100
        bottom = 100
        left = 100
        top = 100
        
        new_width = width + right + left
        new_height = height + top + bottom

        result = Image.new(img.mode, (new_width, new_height), 0)
        result.paste(img, (left, top))
        output_filename = filename[:-4] + '_padded.png'
        result.save(os.path.join(output_dir, output_filename))

def main():
    """Undistort images"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Undistort images.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("test_dir", help="Test Directory.")
    args = parser.parse_args()

    # check if output folder exists and create it if not
    input_dir = os.path.join(args.test_dir, "pngs/raw_images")
    output_dir = os.path.join(args.test_dir, "pngs/undistorted_images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "voxl")):
        os.makedirs(os.path.join(output_dir, "voxl"))
    if not os.path.exists(os.path.join(output_dir, "t265_fisheye1")):
        os.makedirs(os.path.join(output_dir, "t265_fisheye1"))
    if not os.path.exists(os.path.join(output_dir, "t265_fisheye2")):
        os.makedirs(os.path.join(output_dir, "t265_fisheye2"))

    # get camera matrix and distortion coefficients
    # for realsense
    bag = rosbag.Bag(args.bag_file, "r")
    t265_fisheye1_camera_info_topic = "/t265/fisheye1/camera_info"
    for topic, msg, t in bag.read_messages(topics=[t265_fisheye1_camera_info_topic]):
        # save K_fe1, D_fe1
        if topic == t265_fisheye1_camera_info_topic:
            K_fe1 = np.array(msg.K).reshape(3,3)
            D_fe1 = np.array(msg.D)
            R_fe1 = np.array(msg.R).reshape(3,3)
            P_fe1 = np.array(msg.P).reshape(3,4)
            break
    
    t265_fisheye2_camera_info_topic = "/t265/fisheye2/camera_info"
    for topic, msg, t in bag.read_messages(topics=[t265_fisheye2_camera_info_topic]):
        # save K_fe2, D_fe2
        if topic == t265_fisheye2_camera_info_topic:
            K_fe2 = np.array(msg.K).reshape(3,3)
            D_fe2 = np.array(msg.D)
            R_fe2 = np.array(msg.R).reshape(3,3)
            P_fe2 = np.array(msg.P).reshape(3,4)
            break
        
    # for VOXL (from calibration - see /data/modalai/opencv_tracking_intrinsics.yml)
    K_voxl = np.array([[273.90235382142345, 0.0, 315.12271705027996], [0., 274.07405600616045, 241.28160498854680], [0.0, 0.0, 1.0]])
    D_voxl = np.array([[6.4603799803546918e-04, 2.0604787401502832e-03, 0., 0. ]])

    print("K_voxl: ", K_voxl)
    print("D_voxl: ", D_voxl)
    print("K_fe1: ", K_fe1)
    print("D_fe1: ", D_fe1)
    print("R_fe1: ", R_fe1)
    print("P_fe1: ", P_fe1)
    print("K_fe2: ", K_fe2)
    print("D_fe2: ", D_fe2)
    print("R_fe2: ", R_fe2)
    print("P_fe2: ", P_fe2)
    
    # pad images (not working well)
    # if not os.path.exists(os.path.join(input_dir, "voxl_padded")):
    #     os.makedirs(os.path.join(input_dir, "voxl_padded"))
    # if not os.path.exists(os.path.join(input_dir, "t265_fisheye1_padded")):
    #     os.makedirs(os.path.join(input_dir, "t265_fisheye1_padded"))
    # if not os.path.exists(os.path.join(input_dir, "t265_fisheye2_padded")):
    #     os.makedirs(os.path.join(input_dir, "t265_fisheye2_padded"))
    # pad_images(os.path.join(input_dir, "voxl"), os.path.join(input_dir, "voxl_padded"))
    # pad_images(os.path.join(input_dir, "t265_fisheye1"), os.path.join(input_dir, "t265_fisheye1_padded"))
    # pad_images(os.path.join(input_dir, "t265_fisheye2"), os.path.join(input_dir, "t265_fisheye2_padded"))

    # undistort images
    undistort_images_voxl(os.path.join(input_dir, "voxl"), os.path.join(output_dir, "voxl"), K_voxl, D_voxl)
    undistort_images_rs(os.path.join(input_dir, "t265_fisheye1"), os.path.join(output_dir, "t265_fisheye1"), K_fe1, D_fe1, R_fe1, P_fe1)
    undistort_images_rs(os.path.join(input_dir, "t265_fisheye2"), os.path.join(output_dir, "t265_fisheye2"), K_fe2, D_fe2, R_fe2, P_fe2)

    return

if __name__ == '__main__':
    main()