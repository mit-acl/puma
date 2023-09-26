#!/usr/bin/env python
# Author: Kota Kondo

import os
import argparse
import cv2
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import font_manager
import matplotlib.ticker as ticker

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-b", "--objects_location_bag", help="Input bag.", default="/media/kota/T7/frame/hw/day4/flat_object_location_2023-09-13-15-13-22.bag")
    args = parser.parse_args()

    # hardware ground truth (day3 test1)
    objects_names = []
    print("Processing bag: ", args.objects_location_bag)
    bag = rosbag.Bag(args.objects_location_bag, "r")
    
    # get ros topic names
    for topic, msg, t in bag.read_messages():
        # if topic has OBJ in it, then it is an object or if the topic is not in objects_names
        if topic.find("OBJ") != -1 and topic.split("/")[1] not in objects_names:
            objects_names.append(topic.split("/")[1])

    object_gt = {object_name: [] for object_name in objects_names}

    # create world topic names
    object_gt_world_topic_names = []
    for object_name in objects_names:
        object_gt_world_topic_names.append("/" + object_name + "/world")

    # get object ground truth
    for topic, msg, t in bag.read_messages(topics=object_gt_world_topic_names):
        # if topic has OBJ in it, then it is an object
        if topic.find("OBJ") != -1 and topic.find("world") != -1:
            object_gt[topic.split("/")[1]].append([msg.pose.position.x, msg.pose.position.y])

    # get the mean of the object ground truth
    object_gt_mean = {object_name: [] for object_name in objects_names}
    for object_name in objects_names:
        object_gt_mean[object_name] = np.mean(object_gt[object_name], axis=0)
    
    # print out the mean of the object ground truth nicely
    for object_name in objects_names:
        print(object_name, ": ", object_gt_mean[object_name].tolist())
if __name__ == '__main__':
    main()