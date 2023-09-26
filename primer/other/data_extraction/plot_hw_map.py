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
            data_world.append([msg.pos.x, msg.pos.y, msg.pos.z])
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
    parser.add_argument("-d", "--sim_dir", help="Input directory.", default="/media/kota/T7/frame/")
    args = parser.parse_args()

    # hardware ground truth (day3)
    # object_gt = [ [-3.5087559564774855, -2.7145703611626755], \
    #               [4.188334498407577, -1.7830872595230933], \
    #               [-3.604313029916783, 2.7605422921599017], \
    #               [0.7990991621291876, -0.8264659551352469], \
    #               [2.7546678541049077, 1.2412523666448423], \
    #               [-2.6161002931014763, -2.790130537818463], \
    #               [2.836452446611966, -1.9914397117720533], \
    #               [1.6552945518589852, 2.1039764105667764], \
    #               [2.8733158518411894, 3.1966556205230354], \
    #               [-0.7826376463754213, 1.7155747457928945], \
    #               [3.671620893097389, -0.06494449557483611], \
    #               [-1.7911375768430426, 3.1642848524613876], \
    #               [-3.037674863865118, 0.2220354197878991], \
    #               [-1.29218932153881, 0.9020045555061555], \
    #               [-0.9397570201621559, -1.30470121410826] ]
    
    # hardware ground truth (day4)
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

                print("Processing bag: ", bag_text)
                bag = rosbag.Bag(bag_text, "r")

                veh_names = ["NX01"]
                if bag_text.find(veh_names[0]) == -1:
                    continue

                data_sync_world = {}
                data_sync_maps = {}
                for veh_name in veh_names:
                    data_sync_maps[veh_name], data_sync_world[veh_name] = get_data_sync_maps_and_data_sync_world(bag, veh_name)  

                # close the bag
                bag.close()

                # make sure the data has the same length
                if len(veh_names) > 1:
                    shorter_length = min(len(data_sync_maps[veh_names[0]]), len(data_sync_maps[veh_names[1]]))
                    for veh_name in veh_names:
                        data_sync_maps[veh_name] = data_sync_maps[veh_name][:shorter_length]
                        data_sync_world[veh_name] = data_sync_world[veh_name][:shorter_length]

                        # get the last 5 data points

                        data_sync_maps[veh_name] = data_sync_maps[veh_name][-100:-99]
                        data_sync_world[veh_name] = data_sync_world[veh_name][-100:-99]

                # font
                font = font_manager.FontProperties()
                font.set_family('serif')
                plt.rcParams.update({"text.usetex": True})
                plt.rcParams["font.family"] = "Times New Roman"
                font.set_size(10)

                # plot the data (one for 3D trajectory, one for 2D map)
                fig = plt.figure(figsize=(6, 6))
                
                ax = fig.add_subplot(111, projection='3d')
                ax.set(xlabel='y [m]', ylabel='x [m]', zlabel='z [m]') # to make it look consistent with the map
                ax.set_xlim3d(-6, 6)
                ax.set_ylim3d(-6, 6)
                ax.set_zlim3d(0, 3)
                ax.zaxis.set_ticks(np.arange(0, 3, 1))
                ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
                ax.set_aspect('equal', 'box')
                ax.grid()
                ax.view_init(30, 45)
                # ax.set_title('3D Trajectory and Map', fontproperties=font)

                world1_x = np.array(data_sync_world[veh_names[0]])[:,0]
                world1_y = np.array(data_sync_world[veh_names[0]])[:,1]
                world1_z = np.array(data_sync_world[veh_names[0]])[:,2]
                ax.plot3D(world1_y, world1_x, world1_z, label='vehicle 1 traj', color='b', linestyle='-', linewidth=3, alpha=0.8)
                if len(veh_names) > 1:
                    world2_x = np.array(data_sync_world[veh_names[1]])[:,0]
                    world2_y = np.array(data_sync_world[veh_names[1]])[:,1]
                    world2_z = np.array(data_sync_world[veh_names[1]])[:,2]
                    ax.plot3D(world2_y, world2_x, world2_z, label='vehicle 2 traj', color='red', linestyle='-', linewidth=3, alpha=0.8)
                ax.invert_xaxis()

                # plot (state)
                ax.scatter(data_sync_maps[veh_names[0]][-1][1], data_sync_maps[veh_names[0]][-1][0], 0, label=f'vehicle 1 map', color='b', marker='s')
                if len(veh_names) > 1:
                    ax.scatter(data_sync_maps[veh_names[1]][-1][1], data_sync_maps[veh_names[1]][-1][0], 0, label=f'vehicle 2 map', color='red')
                ax.scatter([x[1] for x in object_gt], [x[0] for x in object_gt], [0 for _ in object_gt], c='g', marker='x', label=f'objects')
                ax.legend()
                ax.set_aspect('equal', 'box')
                ax.grid()
                
                ax.set_xlabel('x [m]', fontproperties=font, labelpad=1)
                ax.set_ylabel('y [m]', fontproperties=font, labelpad=1)
                ax.set_zlabel('z [m]', fontproperties=font, labelpad=1)
                ax.set_zlim(0, 3)

                # plt.tight_layout()
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_3Dmap.pdf'), dpi=300)
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_3Dmap.png'), dpi=300)
                plt.close()

                fig = plt.figure(figsize=(6, 6))
                # plot (map)
                # add another plot
                ax = fig.add_subplot(111)
                ax.plot(world1_x, world1_y, label='vehicle 1 traj', color='b', linestyle='-', linewidth=3, alpha=0.8)
                if len(veh_names) > 1:
                    ax.plot(world2_x, world2_y, label='vehicle 2 traj', color='red', linestyle='-', linewidth=3, alpha=0.8)
                ax.scatter(data_sync_maps[veh_names[0]][-1][0], data_sync_maps[veh_names[0]][-1][1], label=f'vehicle 1 map', color='b', marker='s')
                if len(veh_names) > 1:
                    ax.scatter(data_sync_maps[veh_names[1]][-1][0], data_sync_maps[veh_names[1]][-1][1], label=f'vehicle 2 map', color='red')
                ax.scatter([x[0] for x in object_gt], [x[1] for x in object_gt], c='g', marker='x', label=f'objects')
                ax.set(xlim=[-6, 6], ylim=[-6, 6], xlabel='x [m]', ylabel='y [m]')
                ax.legend()
                ax.set_aspect('equal', 'box')
                # ax.set_title('2D Map', fontproperties=font)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_2Dmap.pdf'), dpi=300)
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_2Dmap.png'), dpi=300)
                plt.close()

if __name__ == '__main__':
    main()