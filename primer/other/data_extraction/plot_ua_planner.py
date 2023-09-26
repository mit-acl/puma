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
import matplotlib.font_manager as font_manager
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text
from matplotlib.collections import LineCollection

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
    is_initialized = False
    for topic, msg, t in bag.read_messages(topics=[tpn_frame_align]):
        if topic == tpn_frame_align and msg.frame_src == other_name and msg.frame_dest == ego_name:

            # check if Clipper found a transformation matrix successfully
            threshold = 0.000001
            if np.abs(msg.transform[0] - 1.0) < threshold and np.abs(msg.transform[5] - 1.0) < threshold and np.abs(msg.transform[10] - 1.0) < threshold:
                continue
            elif is_initialized == False:
                t_clipper_start = msg.header.stamp.to_sec()
                is_initialized = True

            transformation_matrix = [[msg.transform[0], msg.transform[1], msg.transform[2], msg.transform[3]],\
                                        [msg.transform[4], msg.transform[5], msg.transform[6], msg.transform[7]],\
                                        [msg.transform[8], msg.transform[9], msg.transform[10], msg.transform[11]],\
                                        [msg.transform[12], msg.transform[13], msg.transform[14], msg.transform[15]]]
            t_frame_align.append(msg.header.stamp.to_sec())
            transformation_matrix_frame_align.append(transformation_matrix)
    return transformation_matrix_frame_align, t_frame_align, t_clipper_start

def get_estimate_euler_and_offset(bag, veh_pair, transformation_matrix_frame_align, t_frame_align, t_clipper_start):

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

            # check if the time is after the clipper start time
            if msg.header.stamp.to_sec() < t_clipper_start:
                continue

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
            pos_drift_estimate = estimate_transformation_matrix[:3,3] - state_pos[:3]
            quat_drift_estimate = R.from_matrix(estimate_transformation_matrix[:3,:3].T @ state_r.as_matrix()[:3,:3]).as_quat()
            offsets_estimate.append(np.concatenate([pos_drift_estimate, quat_drift_estimate]))

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

def sync_data(t_result, data_orig, t_orig):

    # get state1 and state2
    data_synced = []
    for t in t_result:
        data_idx = np.argmin(np.abs(np.array(t_orig) - t))
        data_synced.append(data_orig[data_idx])

    return data_synced

def linear_interpolate_data(t_result, data_orig, t_orig):

    # linearly interpolate data_orig to match t_result
    data_synced_x = []
    data_synced_y = []
    data_synced_z = []
    for t in t_result:
        
        # if t_result is longer than t_orig, add the last element to data_synced
        if t > t_orig[-1] + 0.01:
            data_synced_x.append(0)
            data_synced_y.append(0)
            data_synced_z.append(0)
            continue

        # get the index of the smaller closest time
        data_idx = np.argmin(np.abs(np.array(t_orig) - t))
        if t_orig[data_idx] + 0.01 > t:
            left_data_idx = data_idx - 1
            right_data_idx = data_idx
        else:
            if data_idx + 1 > len(t_orig):
                left_data_idx = data_idx -1 
                right_data_idx = data_idx
            else:
                left_data_idx = data_idx
                right_data_idx = data_idx + 1 
        
        # linearly interpolate

        left_data = np.array(data_orig[left_data_idx,:])
        right_data = np.array(data_orig[right_data_idx,:])
        left_t = t_orig[left_data_idx]
        right_t = t_orig[right_data_idx]
        interpolated_data = left_data + (right_data - left_data) * (t - left_t) / (right_t - left_t)
        data_synced_x.append(interpolated_data.tolist()[0])
        data_synced_y.append(interpolated_data.tolist()[1])
        data_synced_z.append(interpolated_data.tolist()[2])
    
    data_synced = np.array([data_synced_x, data_synced_y, data_synced_z]).T

    return data_synced

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

def get_correct_t(t_plot):
    # make sure the t_rd_plot, t_estimate, t_plot's last element is 100 [s]
    t_plot = np.array(t_plot)
    if t_plot[-1] > 100:
        t_plot = t_plot - t_plot[-1] + 100
    return t_plot

def plot_estimate_and_gt(t_rd_plot, relative_distance, font, t_estimate, euler_estimate, t_plot, euler_actual_drift_yaw, \
                         offsets_estimate, offsets_actual_drift_x, offsets_actual_drift_y, folder, subfolder, bag_text, traj_type):
    
    # make sure the t_rd_plot, t_estimate, t_plot's last element is 100 [s]
    t_rd_plot = get_correct_t(t_rd_plot)
    t_estimate = get_correct_t(t_estimate)
    t_plot = get_correct_t(t_plot)

    # make fig and axs
    fig = plt.figure(figsize=(20, 20))
    # add title
    # fig.suptitle(bag_text.split("/")[-3] + '-' + bag_text.split("/")[-2], fontsize=20)

    # if it's venn diagram, plot relative distance
    if not traj_type == "circle":
        ## plot relative distance
        ax = fig.add_axes([0.1, 0.725, 0.7197, 0.175]) # to match the width with other plots
        ax.plot(t_rd_plot, relative_distance, label='relative distance', linewidth=3)
        ax.set_xlabel('Time [s]', fontproperties=font)
        ax.set_ylabel('Relative Distance [m]', fontproperties=font)
        ax.legend()
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.xaxis.set_tick_params(labelsize=40)
        ax.yaxis.set_tick_params(labelsize=40)
        ax.autoscale()

    # plot euler angle drift
    ax = fig.add_axes([0.1, 0.65, 0.9, 0.25]) if traj_type == "circle" else fig.add_axes([0.1, 0.5, 0.9, 0.175])
    # sync euler_estimate and euler_actual_drift_yaw
    euler_actual_drift_yaw_synced = sync_data(t_estimate, euler_actual_drift_yaw, t_plot)

    err_norm = np.abs(np.array(euler_estimate)[:,2] - euler_actual_drift_yaw_synced)
    points = np.array([t_estimate, np.array(euler_estimate)[:,2]]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1) # to make the line smoother https://stackoverflow.com/questions/47851492/plot-curve-with-blending-line-colors-with-matplotlib-pyplot/47856091#47856091
    norm = plt.Normalize(vmin=0.0, vmax=5.0)
    reversed_cmap = plt.get_cmap('coolwarm')
    lc = LineCollection(segments, cmap=reversed_cmap, norm=norm, label="yaw drift estimate", color=reversed_cmap(norm(err_norm)), linewidths=3)
    lc.set_array(err_norm)
    line = ax.add_collection(lc)
    ax.plot(t_plot, euler_actual_drift_yaw, label='actual yaw drift', color='k', linewidth=2, linestyle='-.', alpha=0.5)
    ax.set_xlabel('Time [s]', fontproperties=font)
    ax.set_ylabel('Yaw Angle Estimate [deg]', fontproperties=font)
    ax.legend(fontsize=20)
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    min_bound = min(min(euler_estimate[:,2])-5, min(euler_actual_drift_yaw)-5)
    max_bound = max(max(euler_estimate[:,2])+5, max(euler_actual_drift_yaw)+5)
    ax.set_ylim([min_bound, max_bound])
    ax.grid()
    cbar = fig.colorbar(line, ax=ax, location='right')
    cbar.ax.tick_params(labelsize=40)
    cbar.set_label("Estimate Error [deg]", fontsize=20)

    # plot translational drift
    ax = fig.add_axes([0.1, 0.35, 0.9, 0.25]) if traj_type == "circle" else fig.add_axes([0.1, 0.275, 0.9, 0.175])
    # sync offsets_estimate and offsets_actual_drift
    offsets_actual_drift_x_synced = sync_data(t_estimate, offsets_actual_drift_x, t_plot)

    # x axis 
    err_norm = np.abs(np.array(offsets_estimate)[:,0] - offsets_actual_drift_x_synced)
    x_points = np.array([t_estimate, np.array(offsets_estimate)[:,0]]).T.reshape(-1, 1, 2)
    # x_segments = np.concatenate([x_points[:-1], x_points[1:]], axis=1)
    x_segments = np.concatenate([x_points[:-2], x_points[1:-1], x_points[2:]], axis=1) # to make the line smoother https://stackoverflow.com/questions/47851492/plot-curve-with-blending-line-colors-with-matplotlib-pyplot/47856091#47856091
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    reversed_cmap = plt.get_cmap('RdYlGn').reversed()
    lc_x = LineCollection(x_segments, cmap=reversed_cmap, norm=norm, label="x drift estimate", color=reversed_cmap(norm(err_norm)), linewidths=3)
    lc_x.set_array(err_norm)
    ax.plot(t_plot, offsets_actual_drift_x, label='actual x drift', color='k', linewidth=2, linestyle='-.', alpha=0.5)
    ax.set_xlabel('Time [s]', fontproperties=font)
    ax.set_ylabel('X Estimate [m]', fontproperties=font)
    line_x = ax.add_collection(lc_x)
    cbar = fig.colorbar(line_x, ax=ax, location='right')
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label("X Estimate Error [m]", fontsize=20)
    ax.legend(fontsize=20)
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    min_bound = min(min(offsets_estimate[:,0])-0.3, min(offsets_actual_drift_x)-0.3)
    max_bound = max(max(offsets_estimate[:,0])+0.3, max(offsets_actual_drift_x)+0.3)
    ax.set_ylim([min_bound, max_bound])
    ax.grid()

    ax = fig.add_axes([0.1, 0.05, 0.9, 0.25]) if traj_type == "circle" else fig.add_axes([0.1, 0.05, 0.9, 0.175])
    # sync offsets_estimate and offsets_actual_drift
    offsets_actual_drift_y_synced = sync_data(t_estimate, offsets_actual_drift_y, t_plot)

    # y axis
    err_norm = np.abs(np.array(offsets_estimate)[:,1] - offsets_actual_drift_y_synced)
    y_points = np.array([t_estimate, np.array(offsets_estimate)[:,1]]).T.reshape(-1, 1, 2)
    # y_segments = np.concatenate([y_points[:-1], y_points[1:]], axis=1)
    y_segments = np.concatenate([y_points[:-2], y_points[1:-1], y_points[2:]], axis=1) # to make the line smoother https://stackoverflow.com/questions/47851492/plot-curve-with-blending-line-colors-with-matplotlib-pyplot/47856091#47856091
    lc_y = LineCollection(y_segments, cmap=reversed_cmap, norm=norm, label="y drift estimate", color=reversed_cmap(norm(err_norm)), linewidths=3)
    lc_y.set_array(err_norm) 
    ax.plot(t_plot, offsets_actual_drift_y, label='actual y drift', color='k', linewidth=2, linestyle='-.', alpha=0.3)
    ax.set_xlabel('Time [s]', fontproperties=font)
    ax.set_ylabel('Y Estimate [m]', fontproperties=font)
    line_y = ax.add_collection(lc_y)
    cbar = fig.colorbar(line_y, ax=ax, location='right')
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label("Y Estimate Error [m]", fontsize=20)
    ax.legend(fontsize=20)
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    min_bound = min(min(offsets_estimate[:,1])-0.3, min(offsets_actual_drift_y)-0.3)
    max_bound = max(max(offsets_estimate[:,1])+0.3, max(offsets_actual_drift_y)+0.3)
    ax.set_ylim([min_bound, max_bound])
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_tracking.png'))

def plot_3d_traj_with_error_color_map(state1, t_state1, cw1, t_cw1, offsets_estimate, euler_offsets_estimate, t_estimate, font, folder, subfolder, bag_text):

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # sync estimate and cw1
    cw = sync_data(t_estimate, cw1, t_cw1) # synced cw1
    state = sync_data(t_estimate, state1, t_state1) # synced state1

    # get the estimate trajectory
    pos_estimate = np.array(state)[:,:3] + np.array(offsets_estimate)[:,:3]

    # get the estimate euler angle
    euler_estimate = R.from_quat(np.array(state)[:,3:]).as_euler('xyz', degrees=True) + np.array(euler_offsets_estimate)

    # get the error norm between cw1 and estimate
    err_norm = np.linalg.norm(pos_estimate[:,:3]-np.array(cw)[:, :3], axis=1)

    # plot the ground truth trajectory (cw)
    x_cw = np.array(cw)[:, 0]
    y_cw = np.array(cw)[:, 1]
    z_cw = np.array(cw)[:, 2]
    ax.plot3D(x_cw, y_cw, z_cw, label='Ground Truth', color="k", linewidth=15, alpha=0.3)

    # plot the estimate trajectory (estimate) with color map
    # Create a continuous norm to map from data points to colors
    x_estimate = np.array(pos_estimate)[:, 0]
    y_estimate = np.array(pos_estimate)[:, 1]
    z_estimate = np.array(pos_estimate)[:, 2]
    points = np.array([x_estimate, y_estimate, z_estimate]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # norm = plt.Normalize(vmin=0.0, vmax=err_norm.max())
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    # lc = Line3DCollection(segments, cmap='viridis', norm=norm, label="Estimate")
    orig_cmap = plt.get_cmap('RdYlGn')
    reversed_cmap = orig_cmap.reversed()
    lc = Line3DCollection(segments, cmap=reversed_cmap, norm=norm, label="Estimate", color=reversed_cmap(norm(err_norm)), linewidths=3)
    
    # Set the values used for colormapping
    lc.set_array(err_norm)

    # lc.set_linewidth(3)
    line = ax.add_collection3d(lc)
    cbar = fig.colorbar(line, ax=ax, shrink=0.5, location='bottom', pad=-0.1, anchor=(0.5, 0.5))
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.set_title("Translation Estimate Error [m]", fontsize=20)
    
    # add attitude https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    setattr(Axes3D, 'arrow3D', _arrow3D)

    # get random color for each arrow
    num_arrow = int(len(cw) / 1000) + 1
    dots_colors = plt.cm.jet(np.linspace(0, 1, num_arrow))

    # get color for each orientation
    # if the error is small, the color is green, and if the error is large, the color is red
    # use the same cmap as the line
    euler_angle_err_max = 5.0 # [deg]
    cmap = plt.get_cmap('coolwarm')
    # norm = BoundaryNorm(np.linspace(0, euler_angle_err_max, 100), cmap.N, clip=True)
    norm = plt.Normalize(vmin=0.0, vmax=euler_angle_err_max)
    # set color bar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, location='bottom', pad=-0.05)
    cbar.ax.tick_params(labelsize=40)
    # make the title bigger
    cbar.ax.set_title("Yaw Estimate Error [deg]", fontsize=20)

    # plot orientation
    for i in range(len(cw)):
        if i % 500 == 0:

            # plot orientation for cw
            r_cw = R.from_quat([cw[i][3], cw[i][4], cw[i][5], cw[i][6]]).as_matrix()[:3,:3]
            for j in range(3):
                # plot arrow
                rx = r_cw[0,j] * 0.3; ry = r_cw[1,j] * 0.3; rz = r_cw[2,j] * 0.3
                ax.arrow3D(cw[i][0], cw[i][1], cw[i][2], rx, ry, rz, mutation_scale=10, arrowstyle="-|>", color='k', linewidth=3)
            # plot a marker
            # ax.scatter(cw[i][0], cw[i][1], cw[i][2], color=dots_colors[int(i/1000)], marker='o', s=60)
            ax.scatter(cw[i][0], cw[i][1], cw[i][2], color='deepskyblue', marker='o', s=800, zorder=1, alpha=0.3)

            # plot orientation for estimate
            r_euler_estimate = R.from_euler("xyz", euler_estimate[i], degrees=True).as_matrix()[:3,:3]
            euler_actual_offset = R.from_matrix(r_euler_estimate).as_euler('xyz', degrees=True) - R.from_quat([cw[i][3], cw[i][4], cw[i][5], cw[i][6]]).as_euler('xyz', degrees=True)
            for j in range(3):
                rx = r_euler_estimate[0,j] * 0.3; ry = r_euler_estimate[1,j] * 0.3; rz = r_euler_estimate[2,j] * 0.3
                # calculate the euler angle error between cw and estimate
                _, _, yaw = euler_actual_offset
                yaw = wrap_angle([yaw])[0]
                print("actual yaw offset: ", abs(yaw))
                ax.arrow3D(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], rx, ry, rz, mutation_scale=10, arrowstyle="-|>", color=cmap(norm(abs(yaw))), linewidth=3)
            # ax.scatter(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], color=dots_colors[int(i/1000)], marker='o', s=60)
            ax.scatter(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], color='deepskyblue', marker='o', s=800, zorder=1, alpha=0.3)

            # connect two dots with a line
            # ax.plot([cw[i][0], pos_estimate[i][0]], [cw[i][1], pos_estimate[i][1]], [cw[i][2], pos_estimate[i][2]], color=dots_colors[int(i/1000)], linewidth=1)
            ax.plot([cw[i][0], pos_estimate[i][0]], [cw[i][1], pos_estimate[i][1]], [cw[i][2], pos_estimate[i][2]], color='deepskyblue', linewidth=2)
            # add text at 1.2 times the distance from the origin
            # ax.text(cw[i][0]*1.2, cw[i][1]*1.2, cw[i][2]*1, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=15)
            if cw[i][0] > 0:
                ax.text((cw[i][0]-1)*1.05+1, (cw[i][1]-1)*1.2+1, cw[i][2]*1, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=15)
            else:
                ax.text((cw[i][0]-1)*1.2+1, (cw[i][1]-1)*1.2+1, cw[i][2]*1.1, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=15)

    # axis labels
    ax.set_xlabel('x [m]', fontproperties=font)
    ax.set_ylabel('y [m]', fontproperties=font)
    ax.set_zlabel('z [m]', fontproperties=font)
    ax.set_zlim(0, 3)

    # set axis tick size
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    ax.zaxis.set_tick_params(labelsize=40)

    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9))
    ax.legend(fontsize=20)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_3d_traj.png'))

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

def plot_state_and_cw(state1, cw1, state1_quat):

    # plot the state and corrupted world (which is state + drift)
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.plot(np.array(state1)[:, 0], np.array(state1)[:, 1], label='state1')
    # axs.plot(np.array(state2)[:, 0], np.array(state2)[:, 1], label='state2')
    axs.plot(np.array(cw1)[:, 0], np.array(cw1)[:, 1], label='corrupted world1')
    # axs.plot(np.array(cw2)[:, 0], np.array(cw2)[:, 1], label='corrupted world2')
    # plot orientation
    for i in range(len(state1_quat)):
        if i % 100 == 0:
            r = R.from_quat([state1_quat[i][0], state1_quat[i][1], state1_quat[i][2], state1_quat[i][3]])
            x, y, z = r.as_euler('xyz', degrees=False)
            axs.quiver(state1[i][0], state1[i][1], np.cos(z), np.sin(z), color='b', width=0.005)
    for i in range(len(cw1)):
        if i % 100 == 0:
            r = R.from_quat([cw1[i][3], cw1[i][4], cw1[i][5], cw1[i][6]])
            x, y, z = r.as_euler('xyz', degrees=False)
            axs.quiver(cw1[i][0], cw1[i][1], np.cos(z), np.sin(z), color='r', width=0.005)
    axs.set_xlabel('x [m]')
    axs.set_ylabel('y [m]')
    axs.legend()
    axs.set_aspect('equal', 'box')
    axs.grid(True)
    plt.tight_layout()
    # plt.show()

def main():

    # data extraction from bag file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-d", "--sim_dir", help="Input directory.", default="/media/kota/T7/ua-planner/single-sims/used_in_icra_paper/primer")
    parser.add_argument("-p", "--plot_type", help="Plot type.", default="3d")
    args = parser.parse_args()

    VEH_NAME = "SQ01s"
    PLOT_TYPE = args.plot_type

    # get folders from input directory
    folders = []
    for file in os.listdir(args.sim_dir):
        if os.path.isdir(os.path.join(args.sim_dir, file)):
            folders.append(os.path.join(args.sim_dir, file))

    # sort folders and if not a directory then remove it from the list
    folders.sort()
    folders = [folder for folder in folders if os.path.isdir(folder)]

    for folder in folders:

        print("Processing folder: {}".format(folder))

        for subfolder in os.listdir(folder):

            # if subfolder is not a directory, then skip
            if not os.path.isdir(os.path.join(folder, subfolder)):
                continue

            print("Processing subfolder: {}".format(subfolder))

            # get bags from input directory
            bags = []
            for file in os.listdir(os.path.join(folder, subfolder)):
                if file.endswith(".bag"):
                    bags.append(os.path.join(folder, subfolder, file))

            # sort bags by time
            bags.sort()

            # for loop
            for bag_text in bags:

                # open the bag
                bag = rosbag.Bag(bag_text, "r")

                # get state (or world)
                tpn_state1 = f'/{VEH_NAME}/state'
                state1 = []; t_state1 = []
                for topic, msg, t in bag.read_messages(topics=[tpn_state1]):
                    if topic == tpn_state1:
                        state1.append([msg.pos.x, msg.pos.y, msg.pos.z, msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w])
                        t_state1.append(msg.header.stamp.to_sec())
                
                # get uncertainty_related values
                tpn_mv_sigma = f'/{VEH_NAME}/primer/moving_direction_sigma_values'
                tpn_mv_uncertainty = f'/{VEH_NAME}/primer/moving_direction_uncertainty_values'
                tpn_mv_times = f'/{VEH_NAME}/primer/moving_direction_uncertainty_times'
                tpn_obs_sigma = f'/{VEH_NAME}/primer/obstacle_sigma_values'
                tpn_obs_uncertainty = f'/{VEH_NAME}/primer/obstacle_uncertainty_values'
                tpn_obs_times = f'/{VEH_NAME}/primer/obstacle_uncertainty_times'
                tpn_alpha = f'/{VEH_NAME}/primer/alpha'
                mv_sigma = []; mv_uncertainty = []; mv_times = []
                obs_sigma = []; obs_uncertainty = []; obs_times = []
                replan_start_time = []
                alphas = []
                for topic, msg, t in bag.read_messages(topics=[tpn_mv_sigma, tpn_mv_uncertainty, tpn_mv_times, tpn_obs_sigma, tpn_obs_uncertainty, tpn_obs_times, tpn_alpha]):
                    if topic == tpn_mv_sigma:
                        mv_sigma.append(msg.data)
                    elif topic == tpn_mv_uncertainty:
                        replan_start_time.append(t.to_sec())
                        mv_uncertainty.append(msg.data)
                        mv_uncertainty_steps = msg.layout.dim[0].size
                        mv_uncertainty_dim = msg.layout.dim[1].size
                    elif topic == tpn_mv_times:
                        mv_times.append(msg.data)
                    elif topic == tpn_obs_sigma:
                        obs_sigma.append(msg.data)
                    elif topic == tpn_obs_uncertainty:
                        obs_uncertainty.append(msg.data)
                        obs_uncertainty_steps = msg.layout.dim[0].size
                        obs_uncertainty_dim = msg.layout.dim[1].size
                    elif topic == tpn_obs_times:
                        obs_times.append(msg.data)
                    elif topic == tpn_alpha:
                        alphas.append(msg.data)
                
                # close the bag
                bag.close()

                
                # my_times are normalized by alpha, so need to multiply alpha to get the actual time
                mv_times = np.array(mv_times) * np.array(alphas)
                print("mv_times: ", mv_times)

                # make replan_start_time start at 0
                replan_start_time = np.array(replan_start_time) - replan_start_time[0]
                
                # number of replan
                num_replan = len(replan_start_time)

                print(num_replan)

                # convert all the list to numpy array
                state1 = np.array(state1)
                t_state1 = np.array(t_state1)
                mv_sigma = np.array(mv_sigma)
                mv_uncertainty = np.array(mv_uncertainty) #[x, y, z] for each steps for each replanning
                mv_times = np.array(mv_times).reshape(num_replan, mv_uncertainty_steps) #t for each steps for each replanning
                obs_sigma = np.array(obs_sigma) 
                obs_uncertainty = np.array(obs_uncertainty) #[x, y, z] for each steps for each replanning
                obs_times = np.array(obs_times).reshape(num_replan, obs_uncertainty_steps) # for each steps for each replanning

                # reshape the data

                # mv
                mv_uncertainty_new = np.zeros((num_replan, mv_uncertainty_steps,  mv_uncertainty_dim))
                for i in range(mv_uncertainty.shape[1]):
                    replan_idx = int(mv_uncertainty.shape[0] / mv_uncertainty_steps)
                    dim_idx = i % mv_uncertainty_dim
                    step_idx = int(i / mv_uncertainty_dim)
                    mv_uncertainty_new[replan_idx, step_idx, dim_idx] = mv_uncertainty[0, i]

                # obs
                obs_uncertainty_new = np.zeros((num_replan, obs_uncertainty_steps,  obs_uncertainty_dim))
                for i in range(obs_uncertainty.shape[1]):
                    replan_idx = int(obs_uncertainty.shape[0] / obs_uncertainty_steps)
                    dim_idx = i % obs_uncertainty_dim
                    step_idx = int(i / obs_uncertainty_dim)
                    obs_uncertainty_new[replan_idx, step_idx, dim_idx] = obs_uncertainty[0, i]

                # total uncertainty 
                mv_uncertainty_synced = np.zeros((num_replan, obs_uncertainty_steps, mv_uncertainty_dim))
                for i in range(num_replan): # for each replanning
                    mv_uncertainty_synced[i,:,:] = linear_interpolate_data(obs_times[i], mv_uncertainty_new[i,:,:], mv_times[i])
                
                total_uncertainty = mv_uncertainty_synced + obs_uncertainty_new

                # font
                font = font_manager.FontProperties()
                font.set_family('serif')
                plt.rcParams.update({"text.usetex": True})
                plt.rcParams["font.family"] = "Times New Roman"
                font.set_size(40)

                ### plot the uncertainty propagated in horizon ###
                fig = plt.figure(figsize=(25, 14))

                # plot total uncertainty (moving direction + obstacle)
                # ax = fig.add_axes(311)

                # # plot the total uncertainty
                # for i in range(num_replan): # for each replanning
                #     ax.plot(obs_times[i, :] + replan_start_time[i], total_uncertainty[i, :, 0], label=f'x in step {i}', linewidth=3, color='r')
                #     ax.plot(obs_times[i, :] + replan_start_time[i], total_uncertainty[i, :, 1], label=f'y in step {i}', linewidth=3, color='g')
                #     ax.plot(obs_times[i, :] + replan_start_time[i], total_uncertainty[i, :, 2], label=f'z in step {i}', linewidth=3, color='b')
                # ax.set_xlabel('Time [s]', fontproperties=font)
                # ax.set_ylabel('Total Uncertainty [??]', fontproperties=font)
                # ax.grid(True)
                # ax.legend(fontsize=20)
                # ax.xaxis.set_tick_params(labelsize=40)
                # # plt.xlim(0, 8)
                # ax.yaxis.set_tick_params(labelsize=40)

                ## plot moving direction uncertainty
                ax = fig.add_axes(211)

                print("mv_uncertainty_synced.shape: ", mv_uncertainty_synced.shape)

                for i in range(num_replan): # for each replanning
                    ax.plot(obs_times[i, :] + replan_start_time[i], mv_uncertainty_synced[i, :, 0], label=f'Direction of motion uncertainty', linewidth=10, color='k')
                    # ax.plot(obs_times[i, :] + replan_start_time[i], mv_uncertainty_synced[i, :, 1], label=f'y in step {i}', linewidth=10, color='g')
                    # ax.plot(obs_times[i, :] + replan_start_time[i], mv_uncertainty_synced[i, :, 2], label=f'z in step {i}', linewidth=10, color='b')
                    # points = np.array([obs_times[i, :] + replan_start_time[i], mv_uncertainty_synced[i, :, 0]]).T.reshape(-1, 1, 2)
                    # segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1) # to make the line smoother https://stackoverflow.com/questions/47851492/plot-curve-with-blending-line-colors-with-matplotlib-pyplot/47856091#47856091
                    # norm = plt.Normalize(vmin=0.35, vmax=0.5)
                    # reversed_cmap = plt.get_cmap('RdYlGn').reversed()
                    # lc = LineCollection(segments, cmap=reversed_cmap, norm=norm, label="Uncertainty [m]", color=reversed_cmap(norm(mv_uncertainty_synced[i, :, 0])), linewidths=10)
                    # lc.set_array(mv_uncertainty_synced[i, :, 0])
                    # line = ax.add_collection(lc)
                    # cbar = fig.colorbar(line, ax=ax, location='right')
                    # cbar.ax.tick_params(labelsize=40)
                    # cbar.set_label("Uncertainty [m]", fontsize=40)

                ax.set_xlabel('Time [s]', fontproperties=font)
                ax.set_ylabel('Standard Deviatoin [m]', fontproperties=font)
                ax.grid(True)
                ax.legend(fontsize=40)
                ax.xaxis.set_tick_params(labelsize=40)
                ax.yaxis.set_tick_params(labelsize=40)
                plt.ylim(0.33, 0.52)
                plt.xlim(0, 4.2)

                # looking at the moving direction
                x1 = 0.1; y1 = 6.0; x2 = 2.2; y2 = 0; x3 = 3.45; y3 = 6.0; x4 = 4.0; y4 = 3.0; 
                rectangle1 = plt.Rectangle((x2,y2), x3-x2, y3-y2, fc='forestgreen', alpha=0.1)
                ax.add_patch(rectangle1)

                # add text to indicate the agent's behavior (specific to sim_002_2023-09-12-16-57-33.bag) with textlinewidth=3
                ax.text(2.33, 0.5, 'Looking at direction of motion', fontsize=35, color='forestgreen', fontweight='bold')

                ## plot obstacle uncertainty
                ax = fig.add_axes(212)
                for i in range(num_replan): # for each replanning
                    ax.plot(obs_times[i, :] + replan_start_time[i], obs_uncertainty_new[i, :, 0], label=f'Known obstacle uncertainty', linewidth=10, color='k')
                    # ax.plot(obs_times[i, :] + replan_start_time[i], obs_uncertainty_new[i, :, 1], label=f'y in step {i}', linewidth=10, color='g')
                    # ax.plot(obs_times[i, :] + replan_start_time[i], obs_uncertainty_new[i, :, 2], label=f'z in step {i}', linewidth=10, color='b')
                    # points = np.array([obs_times[i, :] + replan_start_time[i], obs_uncertainty_new[i, :, 0]]).T.reshape(-1, 1, 2)
                    # segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1) # to make the line smoother https://stackoverflow.com/questions/47851492/plot-curve-with-blending-line-colors-with-matplotlib-pyplot/47856091#47856091
                    # norm = plt.Normalize(vmin=3.5, vmax=5.0)
                    # reversed_cmap = plt.get_cmap('RdYlGn').reversed()
                    # lc = LineCollection(segments, cmap=reversed_cmap, norm=norm, label="Uncertainty [m]", color=reversed_cmap(norm(obs_uncertainty_new[i, :, 0])), linewidths=10)
                    # lc.set_array(obs_uncertainty_new[i, :, 0])
                    # line = ax.add_collection(lc)
                    # cbar = fig.colorbar(line, ax=ax, location='right')
                    # cbar.ax.tick_params(labelsize=40)
                    # cbar.set_label("Uncertainty [m]", fontsize=40)

                ax.set_xlabel('Time [s]', fontproperties=font)
                ax.set_ylabel('Standard Deviatoin [m]', fontproperties=font)
                ax.set_ylim([3.3, 5.2])
                plt.xlim(0, 4.2)

                ## plot squares to indicate the agent's behavior (specific to sim_002_2023-09-12-16-57-33.bag)

                # looking at obst
                x1 = 0.1; y1 = 6.0; x2 = 1.6; y2 = 0; x3 = 3.45; y3 = 6.0; x4 = 4.0; y4 = 3.0; 
                rectangle1 = plt.Rectangle((x1,y1), x2-x1, y2-y1, fc='forestgreen', alpha=0.1)
                rectangle2 = plt.Rectangle((x3,y3), x4-x3, y4-y3, fc='forestgreen', alpha=0.1)
                ax.add_patch(rectangle1)
                ax.add_patch(rectangle2)

                # add text to indicate the agent's behavior (specific to sim_002_2023-09-12-16-57-33.bag) with textlinewidth=3
                ax.text(2.1, 5.05, 'Looking at the obstacle', fontsize=35, color='forestgreen', fontweight='bold')

                # add line to indicate the agent's behavior (specific to sim_002_2023-09-12-16-57-33.bag)
                ax.arrow(1.9, 5.08, -0.3, 0, color='forestgreen', head_width=0.05, linewidth=5)
                ax.arrow(3.15, 5.08, 0.3, 0, color='forestgreen', head_width=0.05, linewidth=5)

                ax.grid(True)
                ax.legend(fontsize=40)
                ax.xaxis.set_tick_params(labelsize=40)
                # plt.xlim(0, 8)
                ax.xaxis.limit_range_for_scale(0, 10)
                ax.yaxis.set_tick_params(labelsize=40)
                
                plt.tight_layout()
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_uncertainty.png'))
                plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_uncertainty.pdf'))
                plt.show()
if __name__ == '__main__':
    main()