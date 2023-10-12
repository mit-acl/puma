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

def get_data_sync_maps_and_data_sync_world(veh_name):

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
    print("length of data_sync_maps: ", len(data_sync_maps))
    print("length of data_sync_world: ", len(data_sync_world))
    if len(data_sync_maps) != len(data_sync_world):
        print("length of data_sync_maps and data_sync_world are different.")
        exit()
    
    return data_sync_maps, data_sync_world, t_maps

def get_data_sync_between_agents(data_sync_maps, data_sync_world, veh_names, t_maps):
    """
    when animating the data should be synced between agents too
    """

    t_maps_0 = t_maps[veh_names[0]]
    t_maps_1 = t_maps[veh_names[1]]
    times = t_maps_0 if t_maps_0[-1] < t_maps_1[-1] else t_maps_1
    
    new_data_sync_maps = {veh_names[0]: [], veh_names[1]: []}
    new_data_sync_world = {veh_names[0]: [], veh_names[1]: []}

    for time in times:
        # get the index of the closest time
        idx_0 = min(range(len(t_maps_0)), key=lambda i: abs(t_maps_0[i]-time))
        idx_1 = min(range(len(t_maps_1)), key=lambda i: abs(t_maps_1[i]-time))

        # get the data
        new_data_sync_maps[veh_names[0]].append(data_sync_maps[veh_names[0]][idx_0])
        new_data_sync_maps[veh_names[1]].append(data_sync_maps[veh_names[1]][idx_1])
        new_data_sync_world[veh_names[0]].append(data_sync_world[veh_names[0]][idx_0])
        new_data_sync_world[veh_names[1]].append(data_sync_world[veh_names[1]][idx_1])
    
    return new_data_sync_maps, new_data_sync_world



# data extraction from bag file
# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
parser.add_argument("-d", "--sim_dir", help="Input directory.", default="/media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos")
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
# folders.sort(key=os.path.getctime)
folders.sort()

for folder in folders:

    print("Processing folder: ", folder)

    for subfolder in os.listdir(folder):

        print("Processing subfolder: ", subfolder)

        # get bags from input directory
        bags = []
        for file in os.listdir(os.path.join(folder,subfolder)):
            if file.endswith(".bag"):
                bags.append(os.path.join(folder, subfolder, file))

        # sort bags by time
        bags.sort()

        # for loop
        for bag_text in bags:

            print("Processing bag: ", bag_text)

            bag = rosbag.Bag(bag_text, "r")

            veh_names = ["SQ01s", "SQ02s"]
            # veh_names = ["SQ01s"]
            data_sync_world = {}
            data_sync_maps = {}
            t_maps = {}
            for veh_name in veh_names:
                data_sync_maps[veh_name], data_sync_world[veh_name], t_maps[veh_name] = get_data_sync_maps_and_data_sync_world(veh_name)

            # close the bag
            bag.close()

            
            ## when animating the data should be synced between agents too
            data_sync_maps, data_sync_world = get_data_sync_between_agents(data_sync_maps, data_sync_world, veh_names, t_maps)
            
            # make sure the data has the same length
            if len(veh_names) > 1:
                shorter_length = min(len(data_sync_maps[veh_names[0]]), len(data_sync_maps[veh_names[1]]))
                for veh_name in veh_names:
                    data_sync_maps[veh_name] = data_sync_maps[veh_name][:shorter_length]
                    data_sync_world[veh_name] = data_sync_world[veh_name][:shorter_length]

                    # get the last 5 data points
                    # data_sync_maps[veh_name] = data_sync_maps[veh_name][-5:]
                    # data_sync_world[veh_name] = data_sync_world[veh_name][-5:]


            # plot the data
            font = font_manager.FontProperties()
            font.set_family('serif')
            plt.rcParams.update({"text.usetex": True})
            plt.rcParams["font.family"] = "Times New Roman"
            font.set_size(10)

            fig, ax = plt.subplots()

            # plot (state)
            world0 = ax.scatter(data_sync_world[veh_names[0]][0].pos.x, data_sync_world[veh_names[0]][0].pos.y, label=f'vehicle1', color='orange', marker='s')
            maps0 = ax.scatter(data_sync_maps[veh_names[0]][0][0], data_sync_maps[veh_names[0]][0][1], label=f'map1', color='orange', marker='s')
            line0, = ax.plot(data_sync_world[veh_names[0]][0].pos.x, data_sync_world[veh_names[0]][0].pos.y, label=f'path1', color='orange', alpha=0.3)
            if len(veh_names) > 1:
                world1 = ax.scatter(data_sync_world[veh_names[1]][0].pos.x, data_sync_world[veh_names[1]][0].pos.y, label=f'vehicle2', color='red')
                maps1 = ax.scatter(data_sync_maps[veh_names[1]][0][0], data_sync_maps[veh_names[1]][0][1], label=f'map2', color='red')
                line1, = ax.plot(data_sync_world[veh_names[1]][0].pos.x, data_sync_world[veh_names[1]][0].pos.y, label=f'path2', color='red', alpha=0.3)
            objects = ax.scatter([x[0] for x in object_gt], [x[1] for x in object_gt], c='g', marker='x', label=f'objects')
            ax.set(xlim=[-6, 6], ylim=[-6, 6], xlabel='x [m]', ylabel='y [m]')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid()
            ax.set_aspect('equal', 'box')
            x_line0 = []
            x_line1 = []
            y_line0 = []
            y_line1 = []

            def update(frame):
                # for each frame, update the data stored on each artist.
            
                # agent 0
                x = data_sync_world[veh_names[0]][frame].pos.x
                y = data_sync_world[veh_names[0]][frame].pos.y
                data = np.stack([x, y]).T
                world0.set_offsets(data)

                # agent 1
                if len(veh_names) > 1:
                    x = data_sync_world[veh_names[1]][frame].pos.x
                    y = data_sync_world[veh_names[1]][frame].pos.y
                    data = np.stack([x, y]).T
                    world1.set_offsets(data)

                # update the line plot:
                # agent 0
                x_map = data_sync_maps[veh_names[0]][frame][0]
                y_map = data_sync_maps[veh_names[0]][frame][1]
                data_map = np.stack([x_map, y_map]).T
                maps0.set_offsets(data_map)
                
                # agent 1
                if len(veh_names) > 1:
                    x_map = data_sync_maps[veh_names[1]][frame][0]
                    y_map = data_sync_maps[veh_names[1]][frame][1]
                    data_map = np.stack([x_map, y_map]).T
                    maps1.set_offsets(data_map)

                # plot history of path
                # agent 0
                x_line0.append(data_sync_world[veh_names[0]][frame].pos.x)
                y_line0.append(data_sync_world[veh_names[0]][frame].pos.y)
                line0.set_data(x_line0, y_line0)

                # agent 1
                if len(veh_names) > 1:
                    x_line1.append(data_sync_world[veh_names[1]][frame].pos.x)
                    y_line1.append(data_sync_world[veh_names[1]][frame].pos.y)
                    line1.set_data(x_line1, y_line1)

                if len(veh_names) > 1:
                    return world0, maps0, line0, world1, maps1, line1
                else:
                    return world0, maps0, line0

            animation_text = 'map_animation_' + bag_text.split('/')[-1][4:-4] + '.mp4'
            ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_sync_world[veh_names[0]]), interval=100/len(data_sync_world[veh_names[0]]), blit=True)
            FFwriter = animation.FFMpegWriter(fps=len(data_sync_world[veh_names[0]])/100, extra_args=['-vcodec', 'libx264'], bitrate=50000)
            ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)

            # low res
            # animation_text = 'low_res_map_animation_' + bag_text.split('/')[-1][4:-4] + '.mp4'
            # ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_sync_world[veh_names[0]]), interval=100/len(data_sync_world[veh_names[0]]), blit=True)
            # FFwriter = animation.FFMpegWriter(fps=len(data_sync_world[veh_names[0]])/100, extra_args=['-vcodec', 'libx264'], bitrate=3000)
            # ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)