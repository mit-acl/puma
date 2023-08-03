#!/usr/bin/env python
# extract pngs/timestamps/pose from a rosbag

import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def write_csvs(args, t_list_images, bag, pose_topic, camera="voxl"):
    
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
        with open(os.path.join(args.output_dir, f'csvs/{camera}/frame%06i.csv' % count), 'a') as f:
            f.write(str(msg.pose.position.x) + ',' + str(msg.pose.position.y) + ',' + str(msg.pose.position.z) + ',' + str(msg.pose.orientation.x) + ',' + str(msg.pose.orientation.y) + ',' + str(msg.pose.orientation.z) + ',' + str(msg.pose.orientation.w) + '\n')

def main():
    """Extract a folder of images from a rosbag"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("veh_name", help="Name of vehicle.")
    args = parser.parse_args()

    # check if output folder exists and create it if not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'pngs')):
        os.makedirs(os.path.join(args.output_dir, 'pngs/raw_images'))
    if not os.path.exists(os.path.join(args.output_dir, 'pngs/raw_images/voxl')):
        os.makedirs(os.path.join(args.output_dir, 'pngs/raw_images/voxl'))
    if not os.path.exists(os.path.join(args.output_dir, 'pngs/raw_images/t265_fisheye1')):
        os.makedirs(os.path.join(args.output_dir, 'pngs/raw_images/t265_fisheye1'))
    if not os.path.exists(os.path.join(args.output_dir, 'pngs/raw_images/t265_fisheye2')):
        os.makedirs(os.path.join(args.output_dir, 'pngs/raw_images/t265_fisheye2'))
    if not os.path.exists(os.path.join(args.output_dir, 'csvs')):
        os.makedirs(os.path.join(args.output_dir, 'csvs'))
    if not os.path.exists(os.path.join(args.output_dir, 'csvs/voxl')):
        os.makedirs(os.path.join(args.output_dir, 'csvs/voxl'))
    if not os.path.exists(os.path.join(args.output_dir, 'csvs/t265_fisheye1')):
        os.makedirs(os.path.join(args.output_dir, 'csvs/t265_fisheye1'))
    if not os.path.exists(os.path.join(args.output_dir, 'csvs/t265_fisheye2')):
        os.makedirs(os.path.join(args.output_dir, 'csvs/t265_fisheye2'))

    # get the bag
    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    
    # topic name
    # image_topic = f'/{args.veh_name}/qvio_overlay'
    voxl_image_topic = f'/{args.veh_name}/tracking'
    t265_fisheye1_image_topic = f'/t265/fisheye1/image_raw'
    t265_fisheye2_image_topic = f'/t265/fisheye2/image_raw'
    pose_topic = f'/{args.veh_name}/world'

    # save images
    voxl_t_list_images = []
    fisheye1_t_list_images = []
    fisheye2_t_list_images = []
    voxl_count = 0
    fisheye1_count = 0
    fisheye2_count = 0
    for topic, msg, t in bag.read_messages(topics=[voxl_image_topic, t265_fisheye1_image_topic, t265_fisheye2_image_topic]):
        # save images
        if topic == voxl_image_topic:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(args.output_dir, "pngs/raw_images/voxl/frame%06i.png" % voxl_count), cv_img)
            voxl_t_list_images.append(t.to_sec())
            voxl_count += 1
        elif topic == t265_fisheye1_image_topic:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(args.output_dir, "pngs/raw_images/t265_fisheye1/frame%06i.png" % fisheye1_count), cv_img)
            fisheye1_t_list_images.append(t.to_sec())
            fisheye1_count += 1
        elif topic == t265_fisheye2_image_topic:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(args.output_dir, "pngs/raw_images/t265_fisheye2/frame%06i.png" % fisheye2_count), cv_img)
            fisheye2_t_list_images.append(t.to_sec())
            fisheye2_count += 1

    # write csvs
    write_csvs(args, voxl_t_list_images, bag, pose_topic, camera="voxl")
    write_csvs(args, fisheye1_t_list_images, bag, pose_topic, camera="t265_fisheye1")
    write_csvs(args, fisheye2_t_list_images, bag, pose_topic, camera="t265_fisheye2")

    # close the bag
    bag.close()

    return

if __name__ == '__main__':
    main()