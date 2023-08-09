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

# data extraction from bag file
# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
parser.add_argument("bag_file", help="Input ROS bag.")
parser.add_argument("veh_name", help="Name of vehicle.")
parser.add_argument("output_dir", help="Output directory.")
args = parser.parse_args()

# cut off time for run.bag (Aug 2023)
MIN_TIME = 1691525005
MAX_TIME = 1691525018

# ground truth (Aug 2023)
object_gt = [[1.30538948174781, 0.08792017608525951], [1.7556711854938247, 1.5301845738388788], [-2.970445795397385, -0.017968445918466327], [3.470787181274709, 4.078329613986586], [2.168708267646973, -1.2237931460359912], [-3.9456521452453295, -1.5780622937245332], [-2.4715796031824846, 4.221399753581286], [4.441561003442656, -1.692115998046444], [4.255669637763099, 2.300721891392908], [-1.2788058555668842, 0.8623606354570972], [-2.0180484955869553, 2.902511955121203], [-0.27154462548986463, 2.8820569403751874], [-1.8699706004964698, -2.008375434619125], [2.7665396650365186, 0.1135119037351044], [-0.7926963921536332, -0.6462376920580662], [0.3326033986672144, 2.029590565125389], [-4.188906698042496, 3.617683236245243], [-1.73699333120812, -2.7400201356279594], [4.0859296012985356, 0.4219624323470336]]

# get the bag
bag = rosbag.Bag(args.bag_file, "r")

# topic name
tpn_detections = f'/{args.veh_name}/detections'
tpn_maps = f'/{args.veh_name}/map/poses_only'
tpn_world = f'/{args.veh_name}/world'

# get the data
data_detections = []
data_maps = []
t_maps = []
data_world = []
t_world = []

for topic, msg, t in bag.read_messages(topics=[tpn_detections, tpn_maps, tpn_world]):

    # first 1 min of data is not useful for run.bag (Aug 2023)
    if t.to_sec() < MIN_TIME or t.to_sec() > MAX_TIME:
        # print("skipping data: ", t.to_sec())
        continue

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


# close the bag
bag.close()

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
# print("length of data_sync_maps: ", len(data_sync_maps))
# print("length of data_sync_world: ", len(data_sync_world))
if len(data_sync_maps) != len(data_sync_world):
    print("length of data_sync_maps and data_sync_world are different.")
    exit()

# plot the data
fig, ax = plt.subplots()
world = ax.scatter(data_sync_world[0].pose.position.x, data_sync_world[0].pose.position.y, label=f'vehicle')
maps = ax.scatter(data_sync_maps[0][0], data_sync_maps[0][1], label=f'map')
objects = ax.scatter([x[0] for x in object_gt], [x[1] for x in object_gt], c='g', marker='x', label=f'objects')
line, = ax.plot(data_sync_world[0].pose.position.x, data_sync_world[0].pose.position.y, label=f'path')
ax.set(xlim=[-6, 6], ylim=[-6, 6], xlabel='x [m]', ylabel='y [m]')
ax.legend()
ax.grid()
x_line = []
y_line = []

def update(frame):
    # for each frame, update the data stored on each artist.
    x = data_sync_world[frame].pose.position.x
    y = data_sync_world[frame].pose.position.y
    data = np.stack([x, y]).T
    world.set_offsets(data)
    # update the line plot:
    x_map = data_sync_maps[frame][0]
    y_map = data_sync_maps[frame][1]
    data_map = np.stack([x_map, y_map]).T
    maps.set_offsets(data_map)
    # plot history of path
    x_line.append(data_sync_world[frame].pose.position.x)
    y_line.append(data_sync_world[frame].pose.position.y)
    line.set_data(x_line, y_line)
    return world, maps, line

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_sync_world), interval=300)
# ani.save(os.path.join(args.output_dir, 'animation.gif'), writer='imagemagick')
plt.show()