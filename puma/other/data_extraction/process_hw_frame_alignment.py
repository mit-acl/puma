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
from plot_frame_alignment import get_estimate_euler_and_offset, get_transformation, wrap_angle
from plot_hw_frame_alignment import filter_estimate

def smooth_data(x, y):
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(min(x), max(x), 10)
    Y_ = X_Y_Spline(X_)
    return X_, Y_

# def get_transformation(bag, veh_pair):

#     # ego name
#     ego_name = veh_pair[0]
#     # other name
#     other_name = veh_pair[1]

#     # topic name
#     tpn_frame_align = f'/{ego_name}/frame_align'

#     # get euler angles from data_frame_align (transofrmation matrix from ego to other)
#     euler_est_drift = []
#     offsets_est_drift = []
#     t_est_drift = []
#     for topic, msg, t in bag.read_messages(topics=[tpn_frame_align]):
#         if topic == tpn_frame_align and msg.frame_src == other_name and msg.frame_dest == ego_name:
#             t_est_drift.append(t.to_sec())
#             rotation_matrix = [[msg.transform[0], msg.transform[1], msg.transform[2]], [msg.transform[4], msg.transform[5], msg.transform[6]], [msg.transform[8], msg.transform[9], msg.transform[10]]]
#             r = R.from_matrix(rotation_matrix)
#             euler_est_drift.append(r.as_euler('xyz', degrees=True))
#             offsets_est_drift.append([msg.transform[3], msg.transform[7], msg.transform[11]])
    
#     return euler_est_drift, offsets_est_drift, t_est_drift

def get_ground_truth_transformation(bag, frame_name, t_est_drift):

    # get bag's start and end time
    bag_start_time = bag.get_start_time()
    bag_end_time = bag.get_end_time()
    time_discreted = np.linspace(bag_start_time, bag_end_time, len(t_est_drift)*5)

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
            t_drift.append(msg.header.stamp.to_sec())
            drift.append(msg)
    
    return drift, t_drift

def get_clipper_start_time_index(transformation_matrix, t_frame_align, t_est_drift):

    # get the time when the euler_est_drift and offsets_est_drift are not zeros
    threshold = 0.00000000000001
    t_start_index = 0

    for idx, trans in enumerate(transformation_matrix):
        if not (abs(trans[0][0] - 1) <= threshold and abs(trans[1][1] - 1) <= threshold and abs(trans[2][2] - 1) <= threshold):
            t_start_index = idx
            break

    # clipper start time 
    t_clipper_start = t_frame_align[t_start_index]

    # get start index in t_frame_align
    t_start_index = np.abs(np.array(t_est_drift) - t_clipper_start).argmin()

    print("t_est_drift[t_start_index]: ", t_est_drift[t_start_index])

    return t_start_index

def get_actual_drift_element(idx, t_est_drift, euler_actual_drift_roll, t_actual):
    # get the index of the closest element in t_actual
    t_actual_idx = np.abs(np.array(t_actual) - t_est_drift[idx]).argmin()
    return euler_actual_drift_roll[t_actual_idx]

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-d", "--sim_dir", help="Input directory.")
    parser.add_argument("-s", "--use_success_rate", help="Use success rate.", default="true")
    args = parser.parse_args()
    # VEH_NAMES = ["SQ01s", "SQ02s"]
    VEH_NAMES = ["NX08", "NX04"]

    # get folders from input directory
    folders = []
    for file in os.listdir(args.sim_dir):
        if os.path.isdir(os.path.join(args.sim_dir, file)):
            folders.append(os.path.join(args.sim_dir, file))
        
    # sort folders by time
    # folders.sort(key=os.path.getctime)
    folders.sort()

    # loop through folders
    for folder in folders:

        print("Processing folder: ", folder)

        # get bags from input directory
        bags = []
        for file in os.listdir(folder):
            if file.endswith(".bag"):
                bags.append(os.path.join(folder, file))

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
        success_rate = []

        # for loop
        for bag_text in bags:

            print("Processing bag: ", bag_text)

            # if bag_text has "NX04" in it, skip
            if "NX04" in bag_text:
                continue

            # open the bag
            bag = rosbag.Bag(bag_text, "r")

            # get permutations of veh_names
            # veh_name_permutations = permutations(VEH_NAMES)

            # get combinations of veh_names
            veh_name_combinations = combinations(VEH_NAMES,2)

            euler_est_drift = []
            offsets_est_drift = []
            t_est_drift = []
            # for veh_pair in veh_name_combinations:
                # euler_est_drift, offsets_est_drift, t_est_drift = get_transformation(bag, veh_pair)
            transformation_matrix_frame_align, t_frame_align, t_clipper_start = get_transformation(bag, VEH_NAMES)
            euler_est_drift, offsets_est_drift, t_est_drift  = get_estimate_euler_and_offset(bag, VEH_NAMES, transformation_matrix_frame_align, t_frame_align, t_clipper_start)

            # wrap euler_estimate
            euler_est_drift[:,0] = wrap_angle(euler_est_drift[:,0])
            euler_est_drift[:,1] = wrap_angle(euler_est_drift[:,1])
            euler_est_drift[:,2] = wrap_angle(euler_est_drift[:,2])

            # get actual drift (which is 0 because we used vicon)
            euler_actual_drift_roll = np.zeros(len(t_est_drift))
            euler_actual_drift_pitch = np.zeros(len(t_est_drift))
            euler_actual_drift_yaw = np.zeros(len(t_est_drift))
            offsets_actual_drift_x = np.zeros(len(t_est_drift))
            offsets_actual_drift_y = np.zeros(len(t_est_drift))
            t_actual = np.zeros(len(t_est_drift))

            # wrap euler_actual_drift
            euler_actual_drift_roll = wrap_angle(euler_actual_drift_roll)
            euler_actual_drift_pitch = wrap_angle(euler_actual_drift_pitch)
            euler_actual_drift_yaw = wrap_angle(euler_actual_drift_yaw)

            # close the bag
            bag.close()

            # analysis
            # find the first time CLIPPER found a solution
            t_clipper_start_index = get_clipper_start_time_index(transformation_matrix_frame_align, t_frame_align, t_est_drift)

            if t_clipper_start == 0.0:
                print(f"CLIPPER did not find a solution in {bag_text}")
                success_rate.append(0)
                continue
            else:
                print(f"CLIPPER found a solution in {bag_text}")
                success_rate.append(1)

            # get the difference between the actual drift and the predicted drift
            euler_est_drift = np.array(euler_est_drift)[t_clipper_start_index:]

            print("euler_est_drift: ", euler_est_drift)

            # filter out the euler and offsets that changed too much from the previous one
            euler_estimate_filtered = []
            offsets_estimate_filtered = []
            euler_estimate_filtered, offsets_estimate_filtered= filter_estimate(euler_est_drift, offsets_est_drift)

            offsets_est_drift = np.array(offsets_est_drift)[t_clipper_start_index:]
            t_est_drift = t_est_drift[t_clipper_start_index:]
            euler_actual_drift_roll = np.array(euler_actual_drift_roll)
            euler_actual_drift_pitch = np.array(euler_actual_drift_pitch)
            euler_actual_drift_yaw = np.array(euler_actual_drift_yaw)

            # initialize the arrays to store the difference between the actual drift and the predicted drift
            euler_diff_roll = []
            euler_diff_pitch = []
            euler_diff_yaw = []
            translational_diff_x = []
            translational_diff_y = []

            for idx, euler in enumerate(euler_estimate_filtered):
                # sync the euler_esttimate_fitlered and euler_actual_drift
                eadr = get_actual_drift_element(idx, t_est_drift, euler_actual_drift_roll, t_actual)
                eadp = get_actual_drift_element(idx, t_est_drift, euler_actual_drift_pitch, t_actual)
                eady = get_actual_drift_element(idx, t_est_drift, euler_actual_drift_yaw, t_actual)
                euler_diff_roll.append(euler[0] - eadr)
                euler_diff_pitch.append(euler[1] - eadp)
                euler_diff_yaw.append(euler[2] - eady)

            for idx, offset in enumerate(offsets_estimate_filtered):
                # sync the offsets_estimate_filtered and offsets_actual_drift
                ofadx = get_actual_drift_element(idx, t_est_drift, offsets_actual_drift_x, t_actual)
                ofady = get_actual_drift_element(idx, t_est_drift, offsets_actual_drift_y, t_actual)
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

        d_string        = f"date                           :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        sf_string       = f"simulation folder              :{folder}"
        success_string  = f"success rate        [%]        :{str(round(mean(success_rate)*100,2))}"
        edym_string     = f"euler_diff_yaw_mean [deg]      :{str(round(mean(euler_diff_yaw_mean),2))}"
        edys_string     = f"euler_diff_yaw_std  [deg]      :{str(round(mean(euler_diff_yaw_std),2))}"
        tdxm_string     = f"translational_diff_x_mean [m]  :{str(round(mean(translational_diff_x_mean),2))}"
        tdxsd_string    = f"translational_diff_x_std  [m]  :{str(round(mean(translational_diff_x_std),2))}"
        tdym_string     = f"translational_diff_y_mean [m]  :{str(round(mean(translational_diff_y_mean),2))}"
        tdysd_string    = f"translational_diff_y_std  [m]  :{str(round(mean(translational_diff_y_std),2))}"
        edrm_string     = f"euler_diff_roll_mean  [deg]    :{str(round(mean(euler_diff_roll_mean),2))}"
        edrsd_string    = f"euler_diff_roll_std   [deg]    :{str(round(mean(euler_diff_roll_std),2))}"
        edpm_string     = f"euler_diff_pitch_mean [deg]    :{str(round(mean(euler_diff_pitch_mean),2))}"
        edpsd_string    = f"euler_diff_pitch_std  [deg]    :{str(round(mean(euler_diff_pitch_std),2))}"
        
        with open(os.path.join(folder, 'data.txt'), 'w') as f:
            f.write("\n")
            f.write("********************************************************************************\n")
            f.write(d_string + '\n')
            f.write(sf_string + '\n')
            f.write(success_string + '\n')
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
        
        with open(os.path.join(args.sim_dir, 'all_data.txt'), 'a') as f:
            f.write("\n")
            f.write("********************************************************************************\n")
            f.write(d_string + '\n')
            f.write(sf_string + '\n')
            f.write(success_string + '\n')
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