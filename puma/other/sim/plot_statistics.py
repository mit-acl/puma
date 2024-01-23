#!/usr/bin/env python3

import argparse
import rosbag
import matplotlib.pyplot as plt
import yaml
import os

def main():

    # arguments
    parser = argparse.ArgumentParser(description='Plot statistics from bag file')
    parser.add_argument('--bag', type=str, default="flight_2024-01-23-12-11-02.bag", help='bag file name')
    args = parser.parse_args()

    # get bag file name
    bag_file = "/media/jtorde/T7/gdp/bags/" + args.bag

    # extract data from bag file
    bag = rosbag.Bag(bag_file)
    t, pos, vel, accel, jerk, psi, dpsi= [], [], [], [], [], [], []
    for topic, msg, _ in bag.read_messages(topics=['/SQ01s/goal']):
        if topic == "/SQ01s/goal":
            t.append(msg.header.stamp.to_sec())
            pos.append([msg.p.x, msg.p.y, msg.p.z])
            vel.append([msg.v.x, msg.v.y, msg.v.z])
            accel.append([msg.a.x, msg.a.y, msg.a.z])
            jerk.append([msg.j.x, msg.j.y, msg.j.z])
            psi.append(msg.psi)
            dpsi.append(msg.dpsi)
    bag.close()

    # get constraints
    yaml_file = "/home/jtorde/Research/puma_ws/src/puma/puma/param/puma.yaml"
    with open(yaml_file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dpsi_max = params["ydot_max"]
    v_max = params["v_max"][0]
    a_max = params["a_max"][0]
    j_max = params["j_max"][0]

    # plot subplots of data
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
    fig.suptitle('Goal Data')
    
    # pos
    labels = ["x", "y", "z"]
    axs[0].plot(t, pos, label=labels)
    axs[0].set_ylabel('pos')
    
    # vel
    axs[1].plot(t, vel, label=labels)
    axs[1].set_ylabel('vel')
    axs[1].axhline(y=v_max, color='r', linestyle='--', label="constraints")
    axs[1].axhline(y=-v_max, color='r', linestyle='--')

    # accel
    axs[2].plot(t, accel, label=labels)
    axs[2].set_ylabel('accel')
    axs[2].axhline(y=a_max, color='r', linestyle='--', label="constraints")
    axs[2].axhline(y=-a_max, color='r', linestyle='--')

    # jerk
    axs[3].plot(t, jerk, label=labels)
    axs[3].set_ylabel('jerk')
    axs[3].axhline(y=j_max, color='r', linestyle='--', label="constraints")
    axs[3].axhline(y=-j_max, color='r', linestyle='--')

    # psi
    axs[4].plot(t, psi, label="psi")
    axs[4].set_ylabel('psi')

    # dpsi
    axs[5].plot(t, dpsi, label="dpsi")
    axs[5].set_ylabel('dpsi')
    axs[5].axhline(y=dpsi_max, color='r', linestyle='--', label="constraints")
    axs[5].axhline(y=-dpsi_max, color='r', linestyle='--')

    # time
    axs[5].set_xlabel('time (s)')

    # legend
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    axs[5].legend()
    
    plt.show()

if __name__ == '__main__':
    main()