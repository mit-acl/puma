#!/usr/bin/env python
# undistort images

import os
import argparse
import cv2
import numpy as np
import rosbag
from PIL import Image

def undistort_images_rs_scaling_test(input_dir, output_dir, K, D, R, P):

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
        scaling_factor = 1.0
        balance = 1.0
        img_dim_out =(int(w*scaling_factor), int(h*scaling_factor))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        nK = K.copy()

        for i in range(1, 10):
            scaling_factor = 0.1 * i

            nK[0,0] = K[0,0] * scaling_factor
            nK[1,1] = K[1,1] * scaling_factor

            # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            sf = str(round(scaling_factor,2)).replace('.','')
            output_filename = filename[:-4] + f'_undistorted_sf_{sf}.png'
            cv2.imwrite(os.path.join(output_dir, output_filename), undistorted_img)

def undistort_images_rs_resolution_test(input_dir, output_dir, K, D, R, P):

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
        size_scale = 1.0
        balance = 1.0
        img_dim_out =(int(w*size_scale), int(h*size_scale))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        nK = K.copy()

        for i in range(1, 10):
            scaling_factor = 0.1 * i

            nK[0,0] = K[0,0] * scaling_factor
            nK[1,1] = K[1,1] * scaling_factor

            # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            sf = str(round(scaling_factor,2)).replace('.','')
            output_filename = filename[:-4] + f'_undistorted_sf_{sf}.png'
            cv2.imwrite(os.path.join(output_dir, output_filename), undistorted_img)

def main():
    """Undistort images"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Undistort images.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("test_dir", help="Test Directory.")
    args = parser.parse_args()

    # check if output folder exists and create it if not
    input_dir = os.path.join(args.test_dir, "pngs/raw_images")
    output_dir = os.path.join(args.test_dir, "pngs/scaling_test")

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
    
    print("K_fe1: ", K_fe1)
    print("D_fe1: ", D_fe1)
    print("R_fe1: ", R_fe1)
    print("P_fe1: ", P_fe1)
    
    # undistort images
    undistort_images_rs_scaling_test(os.path.join(input_dir, "t265_fisheye1"), os.path.join(output_dir, "t265_fisheye1"), K_fe1, D_fe1, R_fe1, P_fe1)

    return

if __name__ == '__main__':
    main()