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
    if len(veh_names) > 1:
        t_maps_1 = t_maps[veh_names[1]]
        times = t_maps_0 if t_maps_0[-1] < t_maps_1[-1] else t_maps_1
        new_data_sync_maps = {veh_names[0]: [], veh_names[1]: []}
        new_data_sync_world = {veh_names[0]: [], veh_names[1]: []}
    else:
        times = t_maps_0
        new_data_sync_maps = {veh_names[0]: []}
        new_data_sync_world = {veh_names[0]: []}
    
    for time in times:
        # get the index of the closest time
        idx_0 = min(range(len(t_maps_0)), key=lambda i: abs(t_maps_0[i]-time))
        if len(veh_names) > 1:
            idx_1 = min(range(len(t_maps_1)), key=lambda i: abs(t_maps_1[i]-time))

        # get the data
        new_data_sync_maps[veh_names[0]].append(data_sync_maps[veh_names[0]][idx_0])
        new_data_sync_world[veh_names[0]].append(data_sync_world[veh_names[0]][idx_0])
        if len(veh_names) > 1:
            new_data_sync_maps[veh_names[1]].append(data_sync_maps[veh_names[1]][idx_1])
            new_data_sync_world[veh_names[1]].append(data_sync_world[veh_names[1]][idx_1])
    
    return new_data_sync_maps, new_data_sync_world



# data extraction from bag file
# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
parser.add_argument("-d", "--sim_dir", help="Input directory.")
args = parser.parse_args()

# cut off time for run.bag (Aug 2023)
MIN_TIME = 1691525005
MAX_TIME = 1691525018

# hardware ground truth (Aug 2023)
object_gt = [[-3.0372080186368464, 0.22397204552242447], \
            [-3.6061494895920125, 2.76184386797755], \
            [-2.1282758244199154, 4.57715987383919], \
            [-0.9346650869955624, -1.313118935224775], \
            [-1.90873055926488, -0.06169130408194741], \
            [0.4392078157270155, -4.009257676954879], \
            [0.9354266630735132, -0.791290723645069], \
            [-2.616311428294339, -2.7901442777178205], \
            [0.747514991829383, 1.718877648210209], \
            [-1.7908125219980169, 3.1645007576400763], \
            [-5.739352268678843, 1.556202256677097], \
            [0.8793082430586796, -4.56679974371163], \
            [4.47415173097812, -4.03246772011779], \
            [-3.5087011180809156, -2.714360427048964], \
            [2.873283235168597, 3.196413394884821], \
            [-0.1825862018472673, 1.5285239404712456], \
            [4.188151321087452, -1.7826952725556753], \
            [3.672397674630294, -0.0654974132670579], \
            [-0.9919950175308176, 1.0853073396056716]]

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

        if not os.path.isdir(os.path.join(folder,subfolder)):
            continue

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

            # if bag is not NX08_test_2023-09-14-00-47-15.bag then skip
            if bag_text.split('/')[-1] != 'NX08_test_2023-09-14-00-47-15.bag':
                continue

            bag = rosbag.Bag(bag_text, "r")


            veh_names = ["NX08"]
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

            # high res
            animation_text = 'high_res_map_animation_' + bag_text.split('/')[-1][4:-4] + '.mp4'
            ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_sync_world[veh_names[0]]), interval=100/len(data_sync_world[veh_names[0]]), blit=True)
            FFwriter = animation.FFMpegWriter(fps=len(data_sync_world[veh_names[0]])/100, extra_args=['-vcodec', 'libx264'], bitrate=10000)
            ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)

            # low res
            animation_text = 'low_res_map_animation_' + bag_text.split('/')[-1][4:-4] + '.mp4'
            ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_sync_world[veh_names[0]]), interval=100/len(data_sync_world[veh_names[0]]), blit=True)
            FFwriter = animation.FFMpegWriter(fps=len(data_sync_world[veh_names[0]])/100, extra_args=['-vcodec', 'libx264'], bitrate=3000)
            ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)