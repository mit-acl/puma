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
import matplotlib.ticker as ticker
import matplotlib.animation as animation

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

    t_clipper_start = 0.0

    # get euler angles from data_frame_align (transofrmation matrix from ego to other)
    transformation_matrix_frame_align = []
    t_frame_align = []
    is_initialized = False
    for topic, msg, t in bag.read_messages(topics=[tpn_frame_align]):
        if topic == tpn_frame_align and msg.frame_src == other_name and msg.frame_dest == ego_name:

            # check if Clipper found a transformation matrix successfully
            threshold = 0.00000000000001
            if np.abs(msg.transform[0] - 1.0) < threshold and np.abs(msg.transform[5] - 1.0) < threshold and np.abs(msg.transform[10] - 1.0) < threshold:
                pass
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
                         offsets_estimate, offsets_actual_drift_x, offsets_actual_drift_y, folder, subfolder, bag_text):
    
    # make sure the t_rd_plot, t_estimate, t_plot's last element is 100 [s]
    t_rd_plot = get_correct_t(t_rd_plot)
    t_estimate = get_correct_t(t_estimate)
    t_plot = get_correct_t(t_plot)

    # the size is too big? sparse the data
    t_estimate = t_estimate[::10]
    euler_estimate = euler_estimate[::10]

    ## plot animation
    print("plotting animation...")

    # plot the data
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # plot (state)
    line0, = axes[0].plot(t_estimate[0], euler_estimate[0,2], label='Euler Estimate', linewidth=3)
    axes[0].plot(t_plot, euler_actual_drift_yaw, label='Actual Yaw Drift', color='k', linewidth=2, linestyle='-.', alpha=0.5)
    axes[0].set(xlim=[0, 100], ylim=[-10, 10], xlabel='Time [s]', ylabel='Euler Estimte [deg]')
    axes[0].legend()
    axes[0].grid(True)
    line1, = axes[1].plot(t_estimate[0], offsets_estimate[0,0], label='X Estimate', linewidth=3)
    axes[1].plot(t_plot, offsets_actual_drift_x, label='Actual X Drift', color='k', linewidth=2, linestyle='-.', alpha=0.5)
    axes[1].set(xlim=[0, 100], ylim=[-2, 2], xlabel='Time [s]', ylabel='X Estimate [m]')
    axes[1].legend()
    axes[1].grid(True)
    line2, = axes[2].plot(t_estimate[0], offsets_estimate[0,1], label='Y Estimate', linewidth=3)
    axes[2].plot(t_plot, offsets_actual_drift_y, label='Actual Y Drift', color='k', linewidth=2, linestyle='-.', alpha=0.5)
    axes[2].set(xlim=[0, 100], ylim=[-2, 2], xlabel='Time [s]', ylabel='Y Estimate [m]')
    axes[2].legend()
    axes[2].grid(True)
    # ax.set_aspect('equal', 'box')
    t_ani = []
    euler_ani = []
    x_offset_ani = []
    y_offset_ani = []
    def update(frame):
        # euler
        t_ani.append(t_estimate[frame])
        euler_ani.append(euler_estimate[frame,2])
        line0.set_data(t_ani, euler_ani)
        # x offset
        x_offset_ani.append(offsets_estimate[frame,0])
        line1.set_data(t_ani, x_offset_ani)
        # y offset
        y_offset_ani.append(offsets_estimate[frame,1])
        line2.set_data(t_ani, y_offset_ani)
        return line0, line1, line2

    # high res
    animation_text = 'animation_' + bag_text.split('/')[-1][4:-4] + '_2d_high_res.mp4'
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t_estimate), interval=100/len(t_estimate), blit=True)
    FFwriter = animation.FFMpegWriter(fps=len(t_estimate)/100, extra_args=['-vcodec', 'libx264'], bitrate=10000)
    ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)

    # low res
    animation_text = 'animation_' + bag_text.split('/')[-1][4:-4] + '_2d_low_res.mp4'
    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(t_estimate), interval=100/len(t_estimate), blit=True)
    FFwriter = animation.FFMpegWriter(fps=len(t_estimate)/100, extra_args=['-vcodec', 'libx264'], bitrate=3000)
    ani.save(os.path.join(folder, subfolder, animation_text), writer=FFwriter)

def plot_3d_traj_with_error_color_map(state1, t_state1, cw1, t_cw1, offsets_estimate, euler_offsets_estimate, t_estimate, font, folder, subfolder, bag_text):

    fig = plt.figure(figsize=(50, 50))
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
    ax.plot3D(x_cw, y_cw, z_cw, label='Ground Truth', color="k", linewidth=40, alpha=0.3)

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
    lc = Line3DCollection(segments, cmap=reversed_cmap, norm=norm, label="Estimate", color=reversed_cmap(norm(err_norm)), linewidths=10)
    
    # Set the values used for colormapping
    lc.set_array(err_norm)

    # lc.set_linewidth(3)
    line = ax.add_collection3d(lc)
    cbar = fig.colorbar(line, ax=ax, shrink=0.5, location='top', pad=-0.1, anchor=(0.2, -0.1))
    cbar.ax.tick_params(labelsize=80)
    cbar.set_label(label="Translation Estimate Error [m]", fontsize=80, labelpad=50)
    
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, location='top', pad=-0.05, anchor=(0.2, -0.6))
    cbar.ax.tick_params(labelsize=80)
    # make the title bigger
    cbar.set_label("Yaw Estimate Error [deg]", fontsize=80, labelpad=50)

    arrow_factor = 0.8

    # plot orientation
    for i in range(len(cw)):
        if i % 500 == 0:

            # plot orientation for cw
            r_cw = R.from_quat([cw[i][3], cw[i][4], cw[i][5], cw[i][6]]).as_matrix()[:3,:3]
            for j in range(3):
                # plot arrow
                rx = r_cw[0,j] * arrow_factor; ry = r_cw[1,j] * arrow_factor; rz = r_cw[2,j] * arrow_factor
                ax.arrow3D(cw[i][0], cw[i][1], cw[i][2], rx, ry, rz, mutation_scale=10, arrowstyle="-|>", color='k', linewidth=10, alpha=0.6)
            # plot a marker
            # ax.scatter(cw[i][0], cw[i][1], cw[i][2], color=dots_colors[int(i/1000)], marker='o', s=60)
            ax.scatter(cw[i][0], cw[i][1], cw[i][2], color='deepskyblue', marker='o', s=10000, zorder=1, alpha=0.3)

            # plot orientation for estimate
            r_euler_estimate = R.from_euler("xyz", euler_estimate[i], degrees=True).as_matrix()[:3,:3]
            euler_actual_offset = R.from_matrix(r_euler_estimate).as_euler('xyz', degrees=True) - R.from_quat([cw[i][3], cw[i][4], cw[i][5], cw[i][6]]).as_euler('xyz', degrees=True)
            for j in range(3):
                rx = r_euler_estimate[0,j] * arrow_factor; ry = r_euler_estimate[1,j] * arrow_factor; rz = r_euler_estimate[2,j] * arrow_factor
                # calculate the euler angle error between cw and estimate
                _, _, yaw = euler_actual_offset
                yaw = wrap_angle([yaw])[0]
                ax.arrow3D(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], rx, ry, rz, mutation_scale=10, arrowstyle="-|>", color=cmap(norm(abs(yaw))), linewidth=10)
            # ax.scatter(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], color=dots_colors[int(i/1000)], marker='o', s=60)
            ax.scatter(pos_estimate[i][0], pos_estimate[i][1], pos_estimate[i][2], color='deepskyblue', marker='o', s=10000, zorder=1, alpha=0.3)

            # connect two dots with a line
            # ax.plot([cw[i][0], pos_estimate[i][0]], [cw[i][1], pos_estimate[i][1]], [cw[i][2], pos_estimate[i][2]], color=dots_colors[int(i/1000)], linewidth=1)
            ax.plot([cw[i][0], pos_estimate[i][0]], [cw[i][1], pos_estimate[i][1]], [cw[i][2], pos_estimate[i][2]], color='deepskyblue', linewidth=10)
            # add text at 1.2 times the distance from the origin
            # ax.text(cw[i][0]*1.2, cw[i][1]*1.2, cw[i][2]*1, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=85)
            # if cw[i][1] > 0:
            #     ax.text((cw[i][0]-1), (cw[i][1]-1), cw[i][2]*1.2, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=85)
            # else:
            #     ax.text((cw[i][0]-1), (cw[i][1]-1), cw[i][2]*0.8, f"{round(t_cw1[i], 2)} [s]", color='k', fontsize=85)

    # axis labels
    ax.set_xlabel('x [m]', fontproperties=font, labelpad=120)
    ax.set_ylabel('y [m]', fontproperties=font, labelpad=120)
    ax.set_zlabel('z [m]', fontproperties=font, labelpad=100)
    ax.set_zlim(0, 3)
    ax.zaxis.set_ticks(np.arange(0, 3, 1))
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.view_init(30, 45)

    # set axis tick size
    ax.xaxis.set_tick_params(labelsize=80, pad=50)
    ax.yaxis.set_tick_params(labelsize=80, pad=50)
    ax.zaxis.set_tick_params(labelsize=80, pad=50)

    ax.invert_xaxis()

    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9))
    ax.legend(fontsize=80)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(folder, subfolder, os.path.splitext(os.path.basename(bag_text))[0] + '_3d_traj.pdf'))
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
    parser.add_argument("-d", "--sim_dir", help="Input directory.", default="/media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos")
    parser.add_argument("-p", "--plot_type", help="Plot type.", default="both")
    args = parser.parse_args()

    VEH_NAMES = ["SQ01s", "SQ02s"]
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

                # get permutations of veh_names
                # veh_name_permutations = permutations(VEH_NAMES)

                # get combinations of veh_names
                # veh_name_combinations = combinations(VEH_NAMES,2)

                # get transformation_matrix_frame_align and t_frame_align
                transformation_matrix_frame_align = []
                t_frame_align = []
                # for veh_pair in veh_name_combinations:
                transformation_matrix_frame_align, t_frame_align, t_clipper_start = get_transformation(bag, ["SQ01s", "SQ02s"])

                # get euler_estimate and offsets_estimate that is synced with t_frame_align
                euler_estimate = []
                offsets_estimate = []
                euler_estimate, offsets_estimate, t_estimate = get_estimate_euler_and_offset(bag, ["SQ01s", "SQ02s"], transformation_matrix_frame_align, t_frame_align, t_clipper_start)
                
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
                state1_quat = []
                t_state1 = []
                state2 = []
                t_state2 = []
                for topic, msg, t in bag.read_messages(topics=[tpn_state1, tpn_state2]):
                    if topic == tpn_state1:
                        state1.append([msg.pos.x, msg.pos.y, msg.pos.z, msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w])
                        state1_quat.append([msg.quat.x, msg.quat.y, msg.quat.z, msg.quat.w])
                        t_state1.append(msg.header.stamp.to_sec())
                    if topic == tpn_state2:
                        state2.append([msg.pos.x, msg.pos.y, msg.pos.z])
                        t_state2.append(msg.header.stamp.to_sec())
                
                # # get corrupted world (debug)
                tpn_cw1 = f'/{VEH_NAMES[0]}/corrupted_world'
                tpn_cw2 = f'/{VEH_NAMES[1]}/corrupted_world'
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
                
                # font
                font = font_manager.FontProperties()
                font.set_family('serif')
                plt.rcParams.update({"text.usetex": True})
                plt.rcParams["font.family"] = "Times New Roman"
                font.set_size(80)

                ### plot the data
                print("plot_type: ", PLOT_TYPE)
                if PLOT_TYPE == "state_and_cw":
                    ## (0) plot state and corrupted world
                    plot_state_and_cw(state1, cw1, state1_quat)
                elif PLOT_TYPE == "both":
                    ## get relative distance
                    # first get synced state1 and state2
                    if len(state1) > len(state2):
                        state1_synced = sync_data(t_state2, state1, t_state1)
                        state2_synced = state2.copy()
                        t_rd_plot = t_state2.copy()
                    else:
                        state2_synced = sync_data(t_state1, state2, t_state2)
                        state1_synced = state1.copy()
                        t_rd_plot = t_state1.copy()
                    relative_distance = []
                    for i in range(len(t_rd_plot)):
                        relative_distance.append(np.linalg.norm(np.array(state1_synced[i])[:3] - np.array(state2_synced[i])[:3]))

                    ## (1) plot relative position, euler angles, and offsets
                    plot_estimate_and_gt(t_rd_plot, relative_distance, font, t_estimate, euler_estimate, t_plot, euler_actual_drift_yaw, \
                                offsets_estimate, offsets_actual_drift_x, offsets_actual_drift_y, folder, subfolder, bag_text)
                    
                    ## (2) plot ground truth position and estimate position in 3D
                    # the estimate traj's red color means the error is big and if it's small is should be green 
                    plot_3d_traj_with_error_color_map(state1, t_state1, cw1, t_cw1, offsets_estimate, euler_estimate, t_estimate, font, folder, subfolder, bag_text)

                elif PLOT_TYPE == "tracking":

                    ## get relative distance
                    # first get synced state1 and state2
                    if len(state1) > len(state2):
                        state1_synced = sync_data(t_state2, state1, t_state1)
                        state2_synced = state2.copy()
                        t_rd_plot = t_state2.copy()
                    else:
                        state2_synced = sync_data(t_state1, state2, t_state2)
                        state1_synced = state1.copy()
                        t_rd_plot = t_state1.copy()
                    relative_distance = []
                    for i in range(len(t_rd_plot)):
                        relative_distance.append(np.linalg.norm(np.array(state1_synced[i])[:3] - np.array(state2_synced[i])[:3]))
                    
                    ## (1) plot relative position, euler angles, and offsets
                    plot_estimate_and_gt(t_rd_plot, relative_distance, font, t_estimate, euler_estimate, t_plot, euler_actual_drift_yaw, \
                                offsets_estimate, offsets_actual_drift_x, offsets_actual_drift_y, folder, subfolder, bag_text)
                    
                elif PLOT_TYPE == "3d":
                    ## (2) plot ground truth position and estimate position in 3D
                    # the estimate traj's red color means the error is big and if it's small is should be green 
                    plot_3d_traj_with_error_color_map(state1,t_state1, cw1, t_cw1, offsets_estimate, euler_estimate, t_estimate, font, folder, subfolder, bag_text)

if __name__ == '__main__':
    main()