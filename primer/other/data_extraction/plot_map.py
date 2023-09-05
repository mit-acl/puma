#!/usr/bin/env python
# Author: Kota Kondo
# example use: python plot_animation.py /media/kota/T7/frame/real_time_fastsam_exp/run.bag NX04 /media/kota/T7/frame/real_time_fastsam_exp

import os
import argparse
import cv2
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import font_manager

def get_data_sync_maps_and_data_sync_world(bag, veh_name):

    # topic name
    tpn_detections = f'/{veh_name}/detections'
    tpn_maps = f'/{veh_name}/map/poses_only'
    tpn_world = f'/{veh_name}/state'

    # get the data
    data_detections = []
    data_maps = []
    t_maps = []
    data_world = []
    t_world = []

    for topic, msg, t in bag.read_messages(topics=[tpn_detections, tpn_maps, tpn_world]):

        # first 1 min of data is not useful for run.bag (Aug 2023)
        # if t.to_sec() < MIN_TIME or t.to_sec() > MAX_TIME:
            # print("skipping data: ", t.to_sec())
            # continue

        if topic == tpn_detections:
            data_detections.append(msg)
        elif topic == tpn_maps:
            data_maps.append(msg)
            t_maps.append(t.to_sec())
        elif topic == tpn_world:
            data_world.append(msg)
            t_world.append(t.to_sec())

    # sync the data
    world_index_list = []
    for t in t_maps:
        # print("t: ", t)
        world_index_list.append(min(range(len(t_world)), key=lambda i: abs(t_world[i]-t)))

    data_sync_world = []
    for idx in world_index_list:
        data_sync_world.append(data_world[idx])

    # clean up data_maps
    data_sync_maps = []
    for msg in data_maps:
        tmp_x = []
        tmp_y = []
        for object_pose in msg.poses:
            tmp_x.append(object_pose.position.x)
            tmp_y.append(object_pose.position.y)
        data_sync_maps.append([tmp_x, tmp_y])

    # sanity check for length of data_sync_maps and data_sync_world
    if len(data_sync_maps) != len(data_sync_world):
        print("length of data_sync_maps and data_sync_world are different.")
        exit()
    
    return data_sync_maps, data_sync_world

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-d", "--sim_dir", help="Input directory.", default="/media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper")
    args = parser.parse_args()

    # cut off time for run.bag (Aug 2023)
    MIN_TIME = 1691525005
    MAX_TIME = 1691525018

    # hardware ground truth (Aug 2023)
    # object_gt = [[1.30538948174781, 0.08792017608525951], [1.7556711854938247, 1.5301845738388788], [-2.970445795397385, -0.017968445918466327], \
    #              [3.470787181274709, 4.078329613986586], [2.168708267646973, -1.2237931460359912], [-3.9456521452453295, -1.5780622937245332], \
    #                 [-2.4715796031824846, 4.221399753581286], [4.441561003442656, -1.692115998046444], [4.255669637763099, 2.300721891392908], \
    #                     [-1.2788058555668842, 0.8623606354570972]]

    # for simulations (need to be synced with floor_objects_env.py)
    np.random.seed(10)
    xy_min = [-5, -5]
    xy_max = [5, 5]
    object_gt = np.random.uniform(low=xy_min, high=xy_max, size=(30,2))

    # get folders from input directory
    folders = []
    for file in os.listdir(args.sim_dir):
        if os.path.isdir(os.path.join(args.sim_dir, file)):
            folders.append(os.path.join(args.sim_dir, file))
        
    # sort folders by time
    folders.sort()
    folders = [folder for folder in folders if os.path.isdir(folder)]

    for folder in folders:

        print("Processing folder: ", folder)

        for subfolder in os.listdir(folder):

            # if subfolder is not a directory, then skip
            if not os.path.isdir(os.path.join(folder, subfolder)):
                continue

            print("Processing subfolder: ", subfolder)

            # get bags from input directory
            bags = []
            for file in os.listdir(os.path.join(folder, subfolder)):
                if file.endswith(".bag"):
                    bags.append(os.path.join(folder, subfolder, file))

            # sort bags by time
            bags.sort()

            # for loop
            for bag_text in bags:

                bag = rosbag.Bag(bag_text, "r")

                veh_names = ["SQ01s", "SQ02s"]
                # veh_names = ["SQ01s"]
                data_sync_world = {}
                data_sync_maps = {}
                for veh_name in veh_names:
                    data_sync_maps[veh_name], data_sync_world[veh_name] = get_data_sync_maps_and_data_sync_world(bag, veh_name)

                # get corrupted world (debug)
                tpn_cw1 = f'/SQ01s/corrupted_world'
                tpn_cw2 = f'/SQ02s/corrupted_world'
                cw1 = []
                t_cw1 = []
                cw2 = []
                t_cw2 = []
                for topic, msg, t in bag.read_messages(topics=[tpn_cw1, tpn_cw2]):
                    if topic == tpn_cw1:
                        cw1.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                        t_cw1.append(msg.header.stamp.to_sec())
                    if topic == tpn_cw2:
                        cw2.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                        t_cw2.append(msg.header.stamp.to_sec())            

                # close the bag
                bag.close()

                # make sure the data has the same length
                if len(veh_names) > 1:
                    shorter_length = min(len(data_sync_maps[veh_names[0]]), len(data_sync_maps[veh_names[1]]))
                    for veh_name in veh_names:
                        data_sync_maps[veh_name] = data_sync_maps[veh_name][:shorter_length]
                        data_sync_world[veh_name] = data_sync_world[veh_name][:shorter_length]

                        # get the last 5 data points
                        data_sync_maps[veh_name] = data_sync_maps[veh_name][-5:]
                        data_sync_world[veh_name] = data_sync_world[veh_name][-5:]

                # font
                font = font_manager.FontProperties()
                font.set_family('serif')
                plt.rcParams.update({"text.usetex": True})
                plt.rcParams["font.family"] = "Times New Roman"
                font.set_size(16)

                # plot the data (one for 3D trajectory, one for 2D map)
                fig = plt.figure(figsize=(12, 6))
                
                ax = fig.add_subplot(121, projection='3d')
                ax.set(xlabel='y [m]', ylabel='x [m]', zlabel='z [m]') # to make it look consistent with the map
                ax.set_xlim3d(-6, 6)
                ax.set_ylim3d(-6, 6)
                ax.set_zlim3d(0, 2.5)
                ax.set_aspect('equal', 'box')
                ax.grid()
                ax.view_init(30, 45)
                ax.set_title('3D Trajectory and Map', fontproperties=font)
                cw1_x = np.array(cw1)[:,0]
                cw1_y = np.array(cw1)[:,1]
                cw1_z = np.array(cw1)[:,2]
                ax.plot3D(cw1_y, cw1_x, cw1_z, label='vehicle 1 traj', color='b', linestyle='-', linewidth=3)
                cw2_x = np.array(cw2)[:,0]
                cw2_y = np.array(cw2)[:,1]
                cw2_z = np.array(cw2)[:,2]
                ax.plot3D(cw2_y, cw2_x, cw2_z, label='vehicle 2 traj', color='red', linestyle='-', linewidth=1)
                ax.invert_xaxis()

                # plot (state)
                ax.scatter(data_sync_maps[veh_names[0]][-1][1], data_sync_maps[veh_names[0]][-1][0], 0, label=f'vehicle 1 map', color='b', marker='s')
                ax.scatter(data_sync_maps[veh_names[1]][-1][1], data_sync_maps[veh_names[1]][-1][0], 0, label=f'vehicle 2 map', color='red')
                ax.scatter([x[1] for x in object_gt], [x[0] for x in object_gt], [0 for _ in object_gt], c='g', marker='x', label=f'objects')
                ax.legend()
                ax.set_aspect('equal', 'box')
                ax.grid()

                # plot (map)
                # add another plot
                ax = fig.add_subplot(122)
                ax.plot(cw1_x, cw1_y, label='vehicle 1 traj', color='b', linestyle='-', linewidth=3)
                ax.plot(cw2_x, cw2_y, label='vehicle 2 traj', color='red', linestyle='-', linewidth=1)
                ax.scatter(data_sync_maps[veh_names[0]][-1][0], data_sync_maps[veh_names[0]][-1][1], label=f'vehicle 1 map', color='b', marker='s')
                ax.scatter(data_sync_maps[veh_names[1]][-1][0], data_sync_maps[veh_names[1]][-1][1], label=f'vehicle 2 map', color='red')
                ax.scatter([x[0] for x in object_gt], [x[1] for x in object_gt], c='g', marker='x', label=f'objects')
                ax.set(xlim=[-6, 6], ylim=[-6, 6], xlabel='x [m]', ylabel='y [m]')
                ax.legend()
                ax.set_aspect('equal', 'box')
                ax.set_title('2D Map', fontproperties=font)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.grid()
                # plt.tight_layout()
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_map.pdf'), dpi=300)
                plt.close()

if __name__ == '__main__':
    main()