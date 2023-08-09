#!/usr/bin/env python
# extract pngs/timestamps/pose from a rosbag

import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

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


def get_camera_intrinsic(bag_text, camera_topic):

    K = None; D = None; R = None; P = None
    bag = rosbag.Bag(bag_text, "r")
    for topic, msg, t in bag.read_messages(topics=[camera_topic]):
        # save K and D
        if topic == camera_topic:
            K = np.array(msg.K).reshape(3,3)
            D = np.array(msg.D)
            R = np.array(msg.R).reshape(3,3)
            P = np.array(msg.P).reshape(3,4)
            break
    
    return K, D, R, P

def write_csvs(args, t_list_images, bag, pose_topic, test_num, camera="voxl"):
    
    # save timestamps
    t_list_pose = []
    msg_list_pose = []
    for topic, msg, t in bag.read_messages(topics=pose_topic):
        # if there is a pose that corresponds to the image, record it in a csv
        t_list_pose.append(t.to_sec())
        msg_list_pose.append(msg)

    # get the index of the first pose that corresponds to the first image
    pose_index_list = []
    for t_image in t_list_images:
        pose_index_list.append(min(range(len(t_list_pose)), key=lambda i: abs(t_list_pose[i]-t_image)))

    # write the csvs
    for count, idx in enumerate(pose_index_list):
        msg = msg_list_pose[idx]
        with open(os.path.join(args.output_dir, f'{test_num}/csvs/{camera}/frame%06i.csv' % count), 'a') as f:
            f.write(str(msg.pose.position.x) + ',' + str(msg.pose.position.y) + ',' + str(msg.pose.position.z) + ',' + str(msg.pose.orientation.x) + ',' + str(msg.pose.orientation.y) + ',' + str(msg.pose.orientation.z) + ',' + str(msg.pose.orientation.w) + '\n')

def main():
    """Extract a folder of images from a rosbag"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("input_rosbag_dir", help="Input ROS bag directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("veh_name", help="Name of vehicle.")
    args = parser.parse_args()

    # get the bags
    bag_files = []
    for file in os.listdir(args.input_rosbag_dir):
        if file.endswith(".bag"):
            bag_files.append(os.path.join(args.input_rosbag_dir, file))

    # loop through the bags
    for bag in bag_files:

        test_num = bag.split('/')[-1].split('_')[0]
        print("test_num: ", test_num)

        # check if output folder exists and create it if not
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images/voxl')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images/voxl'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images/t265_fisheye1')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images/t265_fisheye1'))
        if not os.path.exists(os.path.join(args.output_dir , f'{test_num}/pngs/raw_images/t265_fisheye2')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/raw_images/t265_fisheye2'))
        
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/csvs')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/csvs'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/csvs/voxl')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/csvs/voxl'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/csvs/t265_fisheye1')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/csvs/t265_fisheye1'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/csvs/t265_fisheye2')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/csvs/t265_fisheye2'))

        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images/voxl')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images/voxl'))
        if not os.path.exists(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images/t265_fisheye1')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images/t265_fisheye1'))
        if not os.path.exists(os.path.join(args.output_dir , f'{test_num}/pngs/undistorted_images/t265_fisheye2')):
            os.makedirs(os.path.join(args.output_dir, f'{test_num}/pngs/undistorted_images/t265_fisheye2'))

        bridge = CvBridge()
        
        # topic name
        # image_topic = f'/{args.veh_name}/qvio_overlay'
        voxl_image_topic = f'/{args.veh_name}/tracking'
        t265_fisheye1_image_topic = f'/{args.veh_name}/t265/fisheye1/image_raw'
        t265_fisheye2_image_topic = f'/{args.veh_name}/t265/fisheye2/image_raw'
        pose_topic = f'/{args.veh_name}/world'

        # save images
        voxl_t_list_images = []
        fisheye1_t_list_images = []
        fisheye2_t_list_images = []
        voxl_count = 0
        fisheye1_count = 0
        fisheye2_count = 0
        test_bag = rosbag.Bag(bag, "r")
        for topic, msg, t in test_bag.read_messages(topics=[voxl_image_topic, t265_fisheye1_image_topic, t265_fisheye2_image_topic]):
            # save images
            if topic == voxl_image_topic:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv2.imwrite(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/voxl/frame%06i.png" % voxl_count), cv_img)
                voxl_t_list_images.append(t.to_sec())
                voxl_count += 1
            elif topic == t265_fisheye1_image_topic:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv2.imwrite(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/t265_fisheye1/frame%06i.png" % fisheye1_count), cv_img)
                fisheye1_t_list_images.append(t.to_sec())
                fisheye1_count += 1
            elif topic == t265_fisheye2_image_topic:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv2.imwrite(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/t265_fisheye2/frame%06i.png" % fisheye2_count), cv_img)
                fisheye2_t_list_images.append(t.to_sec())
                fisheye2_count += 1

        # write csvs
        write_csvs(args, voxl_t_list_images, test_bag, pose_topic, test_num, camera="voxl")
        write_csvs(args, fisheye1_t_list_images, test_bag, pose_topic, test_num, camera="t265_fisheye1")
        write_csvs(args, fisheye2_t_list_images, test_bag, pose_topic, test_num, camera="t265_fisheye2")

        # close the test_bag
        test_bag.close()

        # undistort images
        # get camera intrinsics
        K_voxl = np.array([[273.90235382142345, 0.0, 315.12271705027996], [0., 274.07405600616045, 241.28160498854680], [0.0, 0.0, 1.0]])
        D_voxl = np.array([[6.4603799803546918e-04, 2.0604787401502832e-03, 0., 0. ]])
        K_fe1, D_fe1, R_fe1, P_fe1 = get_camera_intrinsic(bag, f"/{args.veh_name}/t265/fisheye1/camera_info")
        K_fe2, D_fe2, R_fe2, P_fe2 = get_camera_intrinsic(bag, f"/{args.veh_name}/t265/fisheye2/camera_info")

        # undistort images
        undistort_images_voxl(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/voxl"), os.path.join(args.output_dir, f"{test_num}/pngs/undistorted_images/voxl"), K_voxl, D_voxl)
        if K_fe1 is not None:
            undistort_images_rs(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/t265_fisheye1"), os.path.join(args.output_dir, f"{test_num}/pngs/undistorted_images/t265_fisheye1"), K_fe1, D_fe1, R_fe1, P_fe1)
        if K_fe2 is not None:
            undistort_images_rs(os.path.join(args.output_dir, f"{test_num}/pngs/raw_images/t265_fisheye2"), os.path.join(args.output_dir, f"{test_num}/pngs/undistorted_images/t265_fisheye2"), K_fe2, D_fe2, R_fe2, P_fe2)

    return

if __name__ == '__main__':
    main()