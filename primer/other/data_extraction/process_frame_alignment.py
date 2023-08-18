#!/usr/bin/env python
# Author: Kota Kondo

import os
import argparse
import rosbag
import rospy
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from tf_bag import BagTfTransformer
from scipy.interpolate import make_interp_spline
from statistics import mean
import datetime

def smooth_data(x, y):
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(min(x), max(x), 10)
    Y_ = X_Y_Spline(X_)
    return X_, Y_

def get_transformation(bag, veh_pair):

    # ego name
    ego_name = veh_pair[0]
    # other name
    other_name = veh_pair[1]

    # topic name
    tpn_frame_align = f'/{ego_name}/frame_align'

    # get euler angles from data_frame_align (transofrmation matrix from ego to other)
    euler_frame_align = []
    offsets_frame_align = []
    t_frame_align = []
    for topic, msg, t in bag.read_messages(topics=[tpn_frame_align]):
        if topic == tpn_frame_align and msg.frame_src == other_name and msg.frame_dest == ego_name:
            t_frame_align.append(t.to_sec())
            rotation_matrix = [[msg.transform[0], msg.transform[1], msg.transform[2]], [msg.transform[4], msg.transform[5], msg.transform[6]], [msg.transform[8], msg.transform[9], msg.transform[10]]]
            r = R.from_matrix(rotation_matrix)
            euler_frame_align.append(r.as_euler('xyz', degrees=True))
            offsets_frame_align.append([msg.transform[3], msg.transform[7], msg.transform[11]])
    
    return euler_frame_align, offsets_frame_align, t_frame_align

def get_ground_truth_transformation(bag, frame_name, t_frame_align):

    # get bag's start and end time
    bag_start_time = bag.get_start_time()
    bag_end_time = bag.get_end_time()
    time_discreted = np.linspace(bag_start_time, bag_end_time, len(t_frame_align)*5)

    # bag transformer
    bag_transformer = BagTfTransformer(bag)

    # get euler_actual_drift and offsets_actual_drift 
    euler_actual_drift = []
    offsets_actual_drift = []
    for t in time_discreted:
        translation, quaternion = bag_transformer.lookupTransform("world", frame_name, rospy.Time.from_sec(t))
        r = R.from_quat(quaternion)
        euler_actual_drift.append(r.as_euler('xyz', degrees=True))
        offsets_actual_drift.append(translation)
    
    return euler_actual_drift, offsets_actual_drift, time_discreted

def get_drift(bag, veh_name):

    # topic name
    tpn_drift = f'/{veh_name}/drift'

    # get drift from data_drift
    drift = []
    t_drift = []
    for topic, msg, t in bag.read_messages(topics=[tpn_drift]):
        if topic == tpn_drift:
            t_drift.append(t.to_sec())
            drift.append(msg)
    
    return drift, t_drift

def get_clipper_start_time_index(euler_frame_align, offsets_frame_align):

    # get the time when the euler_frame_align and offsets_frame_align are not zeros
    threshold = 0.0001
    for i in range(len(euler_frame_align)):
        if not (euler_frame_align[i][0] <= threshold and euler_frame_align[i][1] <= threshold and euler_frame_align[i][2] <= threshold):
            t_start_index_euler = i
            break

    for j in range(len(offsets_frame_align)):
        if not (offsets_frame_align[j][0] <= threshold and offsets_frame_align[j][1] <= threshold and offsets_frame_align[j][2] <= threshold):
            t_start_index_translational = j
            break
    
    t_start_index = max(t_start_index_euler, t_start_index_translational)
    return t_start_index

def get_actual_drift_element(idx, t_frame_align, euler_actual_drift_roll, t_actual):
    # get the index of the closest element in t_actual
    t_actual_idx = np.abs(np.array(t_actual) - t_frame_align[idx]).argmin()
    return euler_actual_drift_roll[t_actual_idx]

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_folder", help="Input directory.")
    args = parser.parse_args()
    VEH_NAMES = ["SQ01s", "SQ02s"]

    # get bags from input directory
    bags = []
    for file in os.listdir(args.bag_folder):
        if file.endswith(".bag"):
            bags.append(os.path.join(args.bag_folder, file))

    # sort bags by time
    bags.sort()

    # make lists to store data across bags
    euler_diff_roll_mean = []
    euler_diff_roll_std = []
    euler_diff_pitch_mean = []
    euler_diff_pitch_std = []
    euler_diff_yaw_mean = []
    euler_diff_yaw_std = []
    translational_diff_x_mean = []
    translational_diff_x_std = []
    translational_diff_y_mean = []
    translational_diff_y_std = []

    # for loop
    for bag_text in bags:

        # open the bag
        bag = rosbag.Bag(bag_text, "r")

        # get permutations of veh_names
        # veh_name_permutations = permutations(VEH_NAMES)

        # get combinations of veh_names
        veh_name_combinations = combinations(VEH_NAMES,2)

        euler_frame_align = []
        offsets_frame_align = []
        t_frame_align = []
        for veh_pair in veh_name_combinations:
             euler_frame_align, offsets_frame_align, t_frame_align = get_transformation(bag, veh_pair)

        # get drift
        drift, t_actual = get_drift(bag, VEH_NAMES[0])
        # get euler_actual_drift and offsets_actual_drift
        euler_actual_drift_roll = []
        euler_actual_drift_pitch = []
        euler_actual_drift_yaw = []
        offsets_actual_drift_x = []
        offsets_actual_drift_y = []
        for i in range(len(drift)):
            euler_actual_drift_roll.append(np.rad2deg(drift[i].drift_euler[0]))
            euler_actual_drift_pitch.append(np.rad2deg(drift[i].drift_euler[1]))
            euler_actual_drift_yaw.append(np.rad2deg(drift[i].drift_euler[2]))
            offsets_actual_drift_x.append(drift[i].drift_pos[0])
            offsets_actual_drift_y.append(drift[i].drift_pos[1])

        # close the bag
        bag.close()

        # analysis
        # find the first time CLIPPER found a solution
        t_clipper_start_index = get_clipper_start_time_index(euler_frame_align, offsets_frame_align)

        # get the difference between the actual drift and the predicted drift
        euler_frame_align = np.array(euler_frame_align)[t_clipper_start_index:]
        offsets_frame_align = np.array(offsets_frame_align)[t_clipper_start_index:]
        t_frame_align = t_frame_align[t_clipper_start_index:]
        euler_actual_drift_roll = np.array(euler_actual_drift_roll)
        euler_actual_drift_pitch = np.array(euler_actual_drift_pitch)
        euler_actual_drift_yaw = np.array(euler_actual_drift_yaw)

        # initialize the arrays to store the difference between the actual drift and the predicted drift
        euler_diff_roll = []
        euler_diff_pitch = []
        euler_diff_yaw = []
        translational_diff_x = []
        translational_diff_y = []

        for idx, euler in enumerate(euler_frame_align):
            # sync the euler_frame_align and euler_actual_drift
            eadr = get_actual_drift_element(idx, t_frame_align, euler_actual_drift_roll, t_actual)
            eadp = get_actual_drift_element(idx, t_frame_align, euler_actual_drift_pitch, t_actual)
            eady = get_actual_drift_element(idx, t_frame_align, euler_actual_drift_yaw, t_actual)
            euler_diff_roll.append(euler[0] - eadr)
            euler_diff_pitch.append(euler[1] - eadp)
            euler_diff_yaw.append(euler[2] - eady)
        
        for idx, offset in enumerate(offsets_frame_align):
            # sync the offsets_frame_align and offsets_actual_drift
            ofadx = get_actual_drift_element(idx, t_frame_align, offsets_actual_drift_x, t_actual)
            ofady = get_actual_drift_element(idx, t_frame_align, offsets_actual_drift_y, t_actual)
            translational_diff_x.append(offset[0] - ofadx)
            translational_diff_y.append(offset[1] - ofady)

        # get the mean and std of the difference between the actual drift and the predicted drift
        euler_diff_roll_mean.append(np.mean(euler_diff_roll))
        euler_diff_roll_std.append(np.std(euler_diff_roll))
        euler_diff_pitch_mean.append(np.mean(euler_diff_pitch))
        euler_diff_pitch_std.append(np.std(euler_diff_pitch))
        euler_diff_yaw_mean.append(np.mean(euler_diff_yaw))
        euler_diff_yaw_std.append(np.std(euler_diff_yaw))
        translational_diff_x_mean.append(np.mean(translational_diff_x))
        translational_diff_x_std.append(np.std(translational_diff_x))
        translational_diff_y_mean.append(np.mean(translational_diff_y))
        translational_diff_y_std.append(np.std(translational_diff_y))

    # end of for loop

    # save the data to text file

    d_string     = f"date                      :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    sf_string    = f"simulation folder         :{args.bag_folder}"
    edym_string  = f"euler_diff_yaw_mean       :{str(round(mean(euler_diff_yaw_mean),2))}"
    edys_string  = f"euler_diff_yaw_std        :{str(round(mean(euler_diff_yaw_std),2))}"
    tdxm_string  = f"translational_diff_x_mean :{str(round(mean(translational_diff_x_mean),2))}"
    tdxsd_string = f"translational_diff_x_std  :{str(round(mean(translational_diff_x_std),2))}"
    tdym_string  = f"translational_diff_y_mean :{str(round(mean(translational_diff_y_mean),2))}"
    tdysd_string = f"translational_diff_y_std  :{str(round(mean(translational_diff_y_std),2))}"
    edrm_string  = f"euler_diff_roll_mean      :{str(round(mean(euler_diff_roll_mean),2))}"
    edrsd_string = f"euler_diff_roll_std       :{str(round(mean(euler_diff_roll_std),2))}"
    edpm_string  = f"euler_diff_pitch_mean     :{str(round(mean(euler_diff_pitch_mean),2))}"
    edpsd_string = f"euler_diff_pitch_std      :{str(round(mean(euler_diff_pitch_std),2))}"

    with open(os.path.join(args.bag_folder, 'data.txt'), 'a') as f:
        f.write("\n")
        f.write("********************************************************************************\n")
        f.write(d_string + '\n')
        f.write(sf_string + '\n')
        f.write(edym_string + '\n')
        f.write(edys_string + '\n')
        f.write(tdxm_string + '\n')
        f.write(tdxsd_string + '\n')
        f.write(tdym_string + '\n')
        f.write(tdysd_string + '\n')
        f.write(edrm_string + '\n')
        f.write(edrsd_string + '\n')
        f.write(edpm_string + '\n')
        f.write(edpsd_string + '\n')
        f.write("********************************************************************************\n")

if __name__ == '__main__':
    main()