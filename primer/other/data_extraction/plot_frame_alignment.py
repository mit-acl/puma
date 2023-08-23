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
from motlee.utils.transform import transform

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
    transformation_matrix_frame_align = []
    t_frame_align = []
    for topic, msg, t in bag.read_messages(topics=[tpn_frame_align]):
        if topic == tpn_frame_align and msg.frame_src == other_name and msg.frame_dest == ego_name:
            t_frame_align.append(msg.header.stamp.to_sec())
            transformation_matrix = [[msg.transform[0], msg.transform[1], msg.transform[2], msg.transform[3]],\
                                        [msg.transform[4], msg.transform[5], msg.transform[6], msg.transform[7]],\
                                        [msg.transform[8], msg.transform[9], msg.transform[10], msg.transform[11]],\
                                        [msg.transform[12], msg.transform[13], msg.transform[14], msg.transform[15]]]
            transformation_matrix_frame_align.append(transformation_matrix)
    return transformation_matrix_frame_align, t_frame_align

def get_estimate_euler_and_offset(bag, veh_pair, transformation_matrix_frame_align, t_frame_align):

    # ego name
    ego_name = veh_pair[0]
    # other name
    other_name = veh_pair[1]

    # topic name
    tpn_state = f'/{ego_name}/state'

    # get euler angles from data_frame_align (transformation matrix from ego to other)
    euler_estimate = []
    offsets_estimate = []
    t_estimate = []
    for topic, msg, t in bag.read_messages(topics=[tpn_state]):
        if topic == tpn_state:

            t_estimate.append(msg.header.stamp.to_sec())

            # get the most recent transformation matrix
            transformation_matrix_idx = np.argmin(np.abs(np.array(t_frame_align) - msg.header.stamp.to_sec()))
            if t_frame_align[transformation_matrix_idx] > msg.header.stamp.to_sec():
                transformation_matrix_idx -= 1
            transformation_matrix = transformation_matrix_frame_align[transformation_matrix_idx]

            # get transformation matrix from state to world
            state_pos = np.array([msg.pos.x, msg.pos.y, msg.pos.z])
            state_r = R.from_quat([msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w])
            
            # apply transformation matrix from state to world
            state_transformation_matrix = np.eye(4)
            state_transformation_matrix[:3,:3] = state_r.as_matrix()
            state_transformation_matrix[:3,3] = state_pos[:3]

            estimate_transformation_matrix = transformation_matrix @ state_transformation_matrix

            # get euler angles and offsets
            r = R.from_matrix(estimate_transformation_matrix[:3,:3])
            euler_estimate.append(r.as_euler('xyz', degrees=True) - state_r.as_euler('xyz', degrees=True))
            offsets_estimate.append(estimate_transformation_matrix[:3,3] - state_pos[:3])

    return np.array(euler_estimate), np.array(offsets_estimate), np.array(t_estimate)

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
            t_drift.append(msg.header.stamp.to_sec())
            drift.append(msg)
    
    return drift, t_drift

def sync_data(t_state1, cw1, t_cw1):

    # get state1 and state2
    cw1_synced = []
    for t in t_state1:
        cw1_idx = np.argmin(np.abs(np.array(t_cw1) - t))
        cw1_synced.append(cw1[cw1_idx])

    return cw1_synced

def wrap_angle(angles):
    # wrap angle to a range of [-180, 180]
    new_angles = []
    for angle in angles:
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        new_angles.append(angle)
    return new_angles

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-d", "--sim_dir", help="Input directory.")
    args = parser.parse_args()
    VEH_NAMES = ["SQ01s", "SQ02s"]

    # get folders from input directory
    folders = []
    for file in os.listdir(args.sim_dir):
        if os.path.isdir(os.path.join(args.sim_dir, file)):
            folders.append(os.path.join(args.sim_dir, file))

    # sort folders by time
    folders.sort()

    for folder in folders:

        print("Processing folder: {}".format(folder))

        # get bags from input directory
        bags = []
        for file in os.listdir(folder):
            if file.endswith(".bag"):
                bags.append(os.path.join(folder, file))

        # sort bags by time
        bags.sort()

        # for loop
        for bag_text in bags:

            # open the bag
            bag = rosbag.Bag(bag_text, "r")

            # get permutations of veh_names
            # veh_name_permutations = permutations(VEH_NAMES)

            # get combinations of veh_names
            # veh_name_combinations = combinations(VEH_NAMES,2)

            # get transformation_matrix_frame_align and t_frame_align
            transformation_matrix_frame_align = []
            t_frame_align = []
            # for veh_pair in veh_name_combinations:
            transformation_matrix_frame_align, t_frame_align = get_transformation(bag, ["SQ01s", "SQ02s"])

            # get euler_estimate and offsets_estimate that is synced with t_frame_align
            euler_estimate = []
            offsets_estimate = []
            euler_estimate, offsets_estimate, t_estimate = get_estimate_euler_and_offset(bag, ["SQ01s", "SQ02s"], transformation_matrix_frame_align, t_frame_align)
            
            # wrap euler_estimate
            euler_estimate[:,0] = wrap_angle(euler_estimate[:,0])
            euler_estimate[:,1] = wrap_angle(euler_estimate[:,1])
            euler_estimate[:,2] = wrap_angle(euler_estimate[:,2])

            # get drift
            drift, t_plot = get_drift(bag, VEH_NAMES[0])
            # get euler_actual_drift and offsets_actual_drift
            euler_actual_drift_roll = []
            euler_actual_drift_pitch = []
            euler_actual_drift_yaw = []
            offsets_actual_drift_x = []
            offsets_actual_drift_y = []
            for i in range(len(drift)):
                euler_actual_drift_roll.append(drift[i].drift_euler[0])
                euler_actual_drift_pitch.append(drift[i].drift_euler[1])
                euler_actual_drift_yaw.append(drift[i].drift_euler[2])
                offsets_actual_drift_x.append(drift[i].drift_pos[0])
                offsets_actual_drift_y.append(drift[i].drift_pos[1])

            # wrap euler_actual_drift
            euler_actual_drift_roll = wrap_angle(euler_actual_drift_roll)
            euler_actual_drift_pitch = wrap_angle(euler_actual_drift_pitch)
            euler_actual_drift_yaw = wrap_angle(euler_actual_drift_yaw)


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
                    t_state1.append(msg.header.stamp.to_sec())
                if topic == tpn_state2:
                    state2.append([msg.pos.x, msg.pos.y])
                    t_state2.append(msg.header.stamp.to_sec())
            
            ## get relative distance
            # first get synced state1 and state2
            if len(state1) > len(state2):
                state1_synced = sync_data(t_state2, state1, t_state1)
                state2_synced = state2.copy()
                t_rd_plot = t_state2
            else:
                state2_synced = sync_data(t_state1, state2, t_state2)
                state1_synced = state1.copy()
                t_rd_plot = t_state1
            print("len(state1_synced): ", len(state1_synced))
            print("len(state2_synced): ", len(state2_synced))
            relative_distance = []
            for i in range(len(t_rd_plot)):
                relative_distance.append(np.linalg.norm(np.array(state1_synced[i]) - np.array(state2_synced[i])))

            # # get corrupted world (debug)
            # tpn_cw1 = f'/{VEH_NAMES[0]}/corrupted_world'
            # tpn_cw2 = f'/{VEH_NAMES[1]}/corrupted_world'
            # cw1 = []
            # t_cw1 = []
            # cw2 = []
            # t_cw2 = []
            # for topic, msg, t in bag.read_messages(topics=[tpn_cw1, tpn_cw2]):
            #     if topic == tpn_cw1:
            #         cw1.append([msg.pose.position.x, msg.pose.position.y])
            #         t_cw1.append(msg.header.stamp.to_sec())
            #     if topic == tpn_cw2:
            #         cw2.append([msg.pose.position.x, msg.pose.position.y])
            #         t_cw2.append(msg.header.stamp.to_sec())
            
            # cw1_synced = sync_data(t_state1, cw1, t_cw1)

            # # get the difference between state1 and cw1_synced
            # diff_x = []
            # diff_y = []
            # for i in range(len(state1)):
            #     diff_x.append(cw1_synced[i][0] - state1[i][0])
            #     diff_y.append(cw1_synced[i][1] - state1[i][1])

            # close the bag
            bag.close()

            # plot the state and corrupted world (which is state + drift)
            # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            # axs.plot(np.array(state1)[:, 0], np.array(state1)[:, 1], label='state1')
            # # axs.plot(np.array(state2)[:, 0], np.array(state2)[:, 1], label='state2')
            # axs.plot(np.array(cw1)[:, 0], np.array(cw1)[:, 1], label='corrupted world1')
            # # axs.plot(np.array(cw2)[:, 0], np.array(cw2)[:, 1], label='corrupted world2')
            # axs.set_xlabel('x [m]')
            # axs.set_ylabel('y [m]')
            # axs.legend()
            # axs.set_aspect('equal', 'box')
            # axs.grid(True)
            # plt.tight_layout()
            # plt.show()

            # plot frame alignment
            fig, axs = plt.subplots(3, 1, figsize=(10, 10))

            ## plot relative distance
            axs[0].plot(t_rd_plot, relative_distance, label='relative distance')
            axs[0].set_xlabel('time [s]')
            axs[0].set_ylabel('relative distance [m]')
            axs[0].legend()
            axs[0].grid()

            ## plot euler angles
            axs[1].plot(t_estimate, euler_estimate[:, 0], label='pred roll drift')
            axs[1].plot(t_estimate, euler_estimate[:, 1], label='pred pitch drift')
            axs[1].plot(t_estimate, euler_estimate[:, 2], label='pred yaw drift')
            axs[1].plot(t_plot, euler_actual_drift_roll, label='actual roll drift')
            axs[1].plot(t_plot, euler_actual_drift_pitch, label='actual pitch drift')
            axs[1].plot(t_plot, euler_actual_drift_yaw, label='actual yaw drift')
            axs[1].set_xlabel('time [s]')
            axs[1].set_ylabel('euler angles [deg]')
            axs[1].legend()
            axs[1].grid()

            ## plot offsets
            axs[2].plot(t_estimate, offsets_estimate[:, 0], label='pred x drift')
            axs[2].plot(t_estimate, offsets_estimate[:, 1], label='pred y drift')
            # axs[2].plot(t_estimate, np.array(offsets_estimate)[:, 2], label='pred z drift')
            # axs[2].plot(t_state1, diff_x, label='diff x')
            # axs[2].plot(t_state1, diff_y, label='diff y')
            axs[2].plot(t_plot, offsets_actual_drift_x, label='actual x drift')
            axs[2].plot(t_plot, offsets_actual_drift_y, label='actual y drift')
            # axs[2].plot(t_estimate, np.array(offsets_actual_drift)[:, 2], label='actual z drift')
            axs[2].set_xlabel('time [s]')
            axs[2].set_ylabel('offsets [m]')
            axs[2].legend()
            axs[2].grid()

            plt.tight_layout()
            plt.savefig(os.path.join(folder, os.path.splitext(os.path.basename(bag_text))[0] + '.png'))
            # plt.show()

if __name__ == '__main__':
    main()