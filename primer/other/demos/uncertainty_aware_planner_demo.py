#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Kota Kondo
#  * See LICENSE file for the license information
#  *
#  * Input: record_bag (bool)
#  *        output_dir (string)
#  *        use_rviz (bool)
#  *
#  * Output: rosbag
#  *         data
#  *
#  * Comment: This script demonstrates our uncertainty-aware planner.
#  *
#  * Developer Notes: This script is based on primer/other/sim/run_ua_sims.py
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
from snapstack_msgs.msg import State
import subprocess
import numpy as np
import argparse

def get_start_end_state():

    x_start_list = []
    y_start_list = []
    z_start_list = []
    yaw_start_list = []

    x_goal_list = []
    y_goal_list = []
    z_goal_list = []

    for i in range(1):
        x_start_list.append(-10)
        y_start_list.append(0)
        z_start_list.append(3)
        yaw_start_list.append(0)

        x_goal_list.append(5)
        y_goal_list.append(0)
        z_goal_list.append(3)

    return x_start_list, y_start_list, z_start_list, yaw_start_list, x_goal_list, y_goal_list, z_goal_list

def check_goal_reached():
    try:
        is_goal_reached = subprocess.check_output(['rostopic', 'echo', '/sim_all_agents_goal_reached', '-n', '1'], timeout=2).decode()
        print("Goal Reached")
        return True 
    except:
        print("Not Goal Reached Yet")
        return False  

def main():

    ##
    ## Parameters
    ##

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("-b", "--record_bag", help="Whether to record bags.", default=False, type=bool)
    parser.add_argument("-o", "--output_dir", help="Directory to save bags.", default="./data/bags")
    parser.add_argument("-v", "--use_rviz", help="Whether to use rviz.", default=True, type=bool)
    args = parser.parse_args()

    NUM_OF_SIMS = 1
    NUM_OBS = 1
    USE_PERFECT_CONTROLLER = "true"
    USE_PERFECT_PREDICTION = "true"
    SIM_DURATION = 100 # in seconds
    OUTPUT_DIR = args.output_dir
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f primer & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"
    TOPICS_TO_RECORD = "/{}/primer/alpha /{}/goal /{}/state /tf /tf_static /{}/primer/fov /obstacles_mesh /{}/primer/pause_sim /{}/primer/best_solution_expert /{}/primer/best_solution_student /{}/term_goal /{}/primer/actual_traj /clock /trajs /sim_all_agents_goal_reached /{}/primer/is_ready /{}/primer/log /{}/primer/obstacle_uncertainty /{}/primer/obstacle_uncertainty_values /{}/primer/obstacle_sigma_values /{}/primer/obstacle_uncertainty_times /{}/primer/moving_direction_uncertainty_values /{}/primer/moving_direction_sigma_values /{}/primer/moving_direction_uncertainty_times"
    USE_RVIZ = args.use_rviz
    AGENTS_TYPES = ["primer"]

    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## comment out some parameters in primer.yaml to overwrite them
    ##

    os.system("sed -i '/uncertainty_aware:/s/^/#/g' $(rospack find primer)/param/primer.yaml")

    ##
    ## simulation loop
    ##

    for agent_type in AGENTS_TYPES:

        ##
        ## set up folders
        ##

        folder_bags = OUTPUT_DIR + f"/{agent_type}/"

        if args.record_bag:
            if not os.path.exists(folder_bags):
                os.makedirs(folder_bags)
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            os.system(f'echo "\n{date}" >> '+folder_bags+'/status.txt')

        for s in range(NUM_OF_SIMS):

            ##
            ## commands list initialized
            ##

            commands = []
            time_sleep = 0.2
            time_sleep_goal = 3.0

            ##
            ## simulation set up
            ##

            ## roscore
            commands.append("roscore")

            ## sim_basestation
            commands.append(f"roslaunch --wait primer sim_base_station.launch num_of_obs:={NUM_OBS} rviz:={USE_RVIZ} gui_mission:=false")
            
            ## set up parameters depending on agent types
            agent_name = "SQ01s"
            if agent_type == "parm_star":
                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/primer/uncertainty_aware false")
            elif agent_type == "primer":
                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/primer/uncertainty_aware true")

            ## sim_onboard
            x_start_list, y_start_list, z_start_list, yaw_start_list, x_goal_list, y_goal_list, z_goal_list = get_start_end_state()
            for i, (x, y, z, yaw) in enumerate(zip(x_start_list, y_start_list, z_start_list, yaw_start_list)):
                agent_name = "SQ01s"
                commands.append(f"sleep 5.0 && roslaunch --wait primer sim_onboard.launch quad:={agent_name} use_downward_camera:=false perfect_controller:={USE_PERFECT_CONTROLLER} perfect_prediction:={USE_PERFECT_PREDICTION} x:={x} y:={y} z:={z} yaw:={yaw} 2> >(grep -v -e TF_REPEATED_DATA -e buffer)")

            ## rosbag record
            if args.record_bag:
                recorded_topics = TOPICS_TO_RECORD.format(*[agent_name for i in range(19)])
                sim_name = f"sim_{str(s).zfill(3)}"
                sim_bag_recorder = sim_name
                commands.append('sleep '+str(time_sleep)+' && cd '+folder_bags+' && rosbag record '+recorded_topics+' -o '+sim_name+' __name:='+sim_bag_recorder)
            
            ## goal checker
            commands.append(f"sleep {time_sleep} && roslaunch --wait primer goal_reached_checker_ua.launch num_of_agents:={1}")

            ## publish goal
            commands.append(f"sleep "+str(time_sleep_goal)+f" && roslaunch --wait primer pub_goal.launch x_goal_list:=\"{x_goal_list}\" y_goal_list:=\"{y_goal_list}\" z_goal_list:=\"{z_goal_list}\"")

            ##
            ## tmux & sending commands
            ##

            session_name="run_many_sims_multiagent_session"
            os.system("tmux kill-session -t" + session_name)
            os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

            for i in range(len(commands)):
                os.system('tmux split-window ; tmux select-layout tiled')
            
            for i in range(len(commands)):
                os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ commands[i]+'" '+' C-m')

            print("commands sent")
            time.sleep(3.0)

            ##
            ## wait until the goal is reached
            ##

            is_goal_reached = False
            tic = time.perf_counter()
            toc = time.perf_counter()

            while (toc - tic < SIM_DURATION and not is_goal_reached):
                toc = time.perf_counter()
                if(check_goal_reached()):
                    print('all the agents reached the goal')
                    is_goal_reached = True
                time.sleep(0.1)

            if (not is_goal_reached):
                if args.record_bag:
                    os.system(f'echo "simulation {s}: not goal reached" >> '+folder_bags+'/status.txt')
                print("Goal is not reached, killing the bag node")
            else:
                if args.record_bag:
                    os.system(f'echo "simulation {s}: goal reached" >> '+folder_bags+'/status.txt')
                print("Goal is reached, killing the bag node")

            if args.record_bag:
                os.system("rosnode kill "+sim_bag_recorder)
                time.sleep(0.5)
            print("Killing the rest")
            os.system(KILL_ALL)

            time.sleep(3.0)

    ## 
    ## uncomment delay_check param
    ##

    ## process data
    print("Processing data")

    ##
    ## tmux & sending commands
    ##

    proc_commands = []
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/process_ua_planner.py -d /media/kota/T7/ua-planner/single-sims/bags -s true")

    session_name="processing"
    os.system("tmux kill-session -t" + session_name)
    os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

    for i in range(len(proc_commands)):
        os.system('tmux split-window ; tmux select-layout tiled')
    
    for i in range(len(proc_commands)):
        os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ proc_commands[i]+'" '+' C-m')

    print("proc_commands sent")

    os.system("sed -i '/uncertainty_aware:/s/^#//g' $(rospack find primer)/param/primer.yaml")    

if __name__ == '__main__':
    main()