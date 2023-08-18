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

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-b", "--bag_folder", help="Input directory.")
    args = parser.parse_args()
    VEH_NAMES = ["SQ01s", "SQ02s"]

    # get bags from input directory
    bags = []
    for file in os.listdir(args.bag_folder):
        if file.endswith(".bag"):
            bags.append(os.path.join(args.bag_folder, file))

    # sort bags by time
    bags.sort()

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
        drift, t_plot = get_drift(bag, VEH_NAMES[0])
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

        # get state (or world)
        tpn_state1 = f'/{VEH_NAMES[0]}/state'
        tpn_state2 = f'/{VEH_NAMES[1]}/state'
        state1 = []
        t_state1 = []
        state2 = []
        t_state2 = []
        for topic, msg, t in bag.read_messages(topics=[tpn_state1, tpn_state2]):
            if topic == tpn_state1:
                state1.append([msg.pos.x, msg.pos.y])
                t_state1.append(t.to_sec())
            if topic == tpn_state2:
                state2.append([msg.pos.x, msg.pos.y])
                t_state2.append(t.to_sec())

        # get corrupted world
        tpn_cw1 = f'/{VEH_NAMES[0]}/corrupted_world'
        tpn_cw2 = f'/{VEH_NAMES[1]}/corrupted_world'
        cw1 = []
        t_cw1 = []
        cw2 = []
        t_cw2 = []
        for topic, msg, t in bag.read_messages(topics=[tpn_cw1, tpn_cw2]):
            if topic == tpn_cw1:
                cw1.append([msg.pose.position.x, msg.pose.position.y])
                t_cw1.append(t.to_sec())
            if topic == tpn_cw2:
                cw2.append([msg.pose.position.x, msg.pose.position.y])
                t_cw2.append(t.to_sec())
        
        # close the bag
        bag.close()

        # plot the state and corrupted world (which is state + drift)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(np.array(state1)[:, 0], np.array(state1)[:, 1], label='state1')
        # axs.plot(np.array(state2)[:, 0], np.array(state2)[:, 1], label='state2')
        axs.plot(np.array(cw1)[:, 0], np.array(cw1)[:, 1], label='corrupted world1')
        # axs.plot(np.array(cw2)[:, 0], np.array(cw2)[:, 1], label='corrupted world2')
        axs.set_xlabel('x [m]')
        axs.set_ylabel('y [m]')
        axs.legend()
        axs.set_aspect('equal', 'box')
        axs.grid(True)
        plt.tight_layout()
        plt.show()

        return
        

        # plot frame alignment
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(t_frame_align, np.array(euler_frame_align)[:, 0], label='pred roll drift')
        axs[0].plot(t_frame_align, np.array(euler_frame_align)[:, 1], label='pred pitch drift')
        axs[0].plot(t_frame_align, np.array(euler_frame_align)[:, 2], label='pred yaw drift')
        axs[0].plot(t_plot, euler_actual_drift_roll, label='actual roll drift')
        axs[0].plot(t_plot, euler_actual_drift_pitch, label='actual pitch drift')
        axs[0].plot(t_plot, euler_actual_drift_yaw, label='actual yaw drift')
        axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('euler angles [deg]')
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(t_frame_align, np.array(offsets_frame_align)[:, 0], label='pred x drift')
        axs[1].plot(t_frame_align, np.array(offsets_frame_align)[:, 1], label='pred y drift')
        # axs[1].plot(t_frame_align, np.array(offsets_frame_align)[:, 2], label='pred z drift')
        axs[1].plot(t_plot, offsets_actual_drift_x, label='actual x drift')
        axs[1].plot(t_plot, offsets_actual_drift_y, label='actual y drift')
        # axs[1].plot(t_frame_align, np.array(offsets_actual_drift)[:, 2], label='actual z drift')
        axs[1].set_xlabel('time [s]')
        axs[1].set_ylabel('offsets [m]')
        axs[1].legend()
        axs[1].grid()
        plt.tight_layout()
        plt.savefig(os.path.join(args.bag_folder, os.path.splitext(os.path.basename(bag_text))[0] + '.png'))
        plt.show()

if __name__ == '__main__':
    main()