#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Input: directory to save bags (if not given the default is  "/media/kota/T7/deep-panther/bags")
#  * Output: bags with the name "sim_{sim_number}_{agent_name}_{date}.bag"
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
from snapstack_msgs.msg import State
import subprocess
import argparse
import numpy as np

def get_drift(is_constant_drift, is_linear_drift):
    
    # constant drift params
    constant_drift_x=0.0
    constant_drift_y=0.0
    constant_drift_z=0.0
    constant_drift_roll=0.0
    constant_drift_pitch=0.0
    constant_drift_yaw=0.0
    
    # linear drift params
    linear_drift_rate_x=0.0
    linear_drift_rate_y=0.0
    linear_drift_rate_z=0.0
    linear_drift_rate_roll=0.0
    linear_drift_rate_pitch=0.0
    linear_drift_rate_yaw=0.0
    
    if is_constant_drift:
        constant_drift_x=1.0
        constant_drift_y=1.0

    if is_linear_drift:
        linear_drift_rate_x=0.01
        linear_drift_rate_y=0.01
        linear_drift_rate_yaw=np.deg2rad(0.1)
    
    return constant_drift_x, constant_drift_y, constant_drift_z, constant_drift_roll, constant_drift_pitch, constant_drift_yaw, linear_drift_rate_x, linear_drift_rate_y, linear_drift_rate_z, linear_drift_rate_roll, linear_drift_rate_pitch, linear_drift_rate_yaw

def agent_dependent_topics(commands, agent_name, other_agent_name, time_fastsam_and_mot, kappa_mot, time_traj_gen, time_takeoff, time_move_to_start, time_start_traj, is_constant_drift, is_linear_drift):
    """ Add topics that are agent dependent to commands """

    ## sim_onboard
    commands.append(f"roslaunch --wait primer sim_onboard.launch quad:={agent_name} veh:={agent_name[:2]} num:={agent_name[2:4]} x:={0.0} y:={0.0} z:=0.0 rviz:=false")

    ## fastsam
    commands.append(f"sleep "+str(time_fastsam_and_mot)+f" && roslaunch --wait primer fastsam.launch quad:={agent_name} is_sim:=false")

    ## mot
    commands.append(f"sleep "+str(time_fastsam_and_mot)+f" && roslaunch --wait motlee_ros mapper.launch quad:={agent_name} kappa:={kappa_mot}")

    ## frame alignment
    commands.append(f"sleep "+str(time_fastsam_and_mot)+f" && roslaunch --wait motlee_ros frame_aligner.launch quad1:={agent_name} quad2:={other_agent_name}")

    ## pose corrupter
    constant_drift_x, constant_drift_y, constant_drift_z, constant_drift_roll, constant_drift_pitch, constant_drift_yaw, \
        linear_drift_rate_x, linear_drift_rate_y, linear_drift_rate_z, linear_drift_rate_roll, linear_drift_rate_pitch, linear_drift_rate_yaw = get_drift(is_constant_drift, is_linear_drift)
    commands.append(f"roslaunch --wait primer pose_corrupter.launch quad:={agent_name} is_constant_drift:={is_constant_drift} constant_drift_x:={constant_drift_x} \
        constant_drift_y:={constant_drift_y} constant_drift_z:={constant_drift_z} constant_drift_roll:={constant_drift_roll} constant_drift_pitch:={constant_drift_pitch} \
        constant_drift_yaw:={constant_drift_yaw} is_linear_drift:={is_linear_drift} linear_drift_rate_x:={linear_drift_rate_x} linear_drift_rate_y:={linear_drift_rate_y} \
        linear_drift_rate_z:={linear_drift_rate_z} linear_drift_rate_roll:={linear_drift_rate_roll} linear_drift_rate_pitch:={linear_drift_rate_pitch} \
        linear_drift_rate_yaw:={linear_drift_rate_yaw}")

    ## trajectory generator onboard
    commands.append(f"sleep "+str(time_traj_gen)+f" &&roslaunch --wait trajectory_generator quad:={agent_name} onboard.launch")

    ## takeoff
    commands.append(f"sleep "+str(time_takeoff)+f" && rostopic pub -1 /{agent_name}/globalflightmode snapstack_msgs/QuadFlightMode '"+"{header: auto, mode: 4}'")

    ## move to the starting position trajectory generator
    commands.append(f"sleep "+str(time_move_to_start)+f" && rostopic pub -1 /{agent_name}/globalflightmode snapstack_msgs/QuadFlightMode '"+"{header: auto, mode: 4}'")

    ## start trajectory generator
    commands.append(f"sleep "+str(time_start_traj)+f" && rostopic pub -1 /{agent_name}/globalflightmode snapstack_msgs/QuadFlightMode '"+"{header: auto, mode: 4}'")

    return commands

def main():

    ##
    ## Arguments
    ##

    parser = argparse.ArgumentParser(description="Run simulations for frame alignment.")
    parser.add_argument("--output_dir", help="Directory to save bags.", default="/media/kota/T7/frame/sim")
    parser.add_argument("--use_rviz", help="Whether to use rviz.", default=True)
    parser.add_argument("--num_of_objects", help="Number of objects.", default=10, type=int)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    USE_RVIZ = args.use_rviz

    ##
    ## Simulation parameters
    ##

    # for dicts
    NUM_OF_AGENTS = [2]
    NUM_OF_OBJECTS = [args.num_of_objects]
    OBJECTS_TYPE = ["pads"] # ["pads", "random"]

    # others
    NUM_OF_SIMS = 1
    SIM_DURATION = 60  # seconds
    IS_CONSTANT_DRIFT = True
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f primer & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"

    ##
    ## MOT parameters
    ##
    
    KAPPA_MOT = 400

    ##
    ## Trajectory generator parameters
    ##

    TIME_TRAJ_GEN = 3.0 # if you don't wait until snap_sim is ready, it will generate an error "Could not read parameters."
    TIME_TAKEOFF = 10.0
    TIME_MOVE_TO_START = 25.0
    TIME_START_TRAJ = 35.0
    TIME_FASTSAM_AND_MOT = TIME_START_TRAJ - 7.0 # takes a bit while to start fastsam and mot

    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## simulation loop
    ##

    # create a dictionary (cause we don't want a nested for loop)
    DICTS = [ {"num_of_agents": num_of_agents, "num_of_objects": num_of_objects, "objects_type": objects_type} \
              for num_of_agents in NUM_OF_AGENTS for num_of_objects in NUM_OF_OBJECTS for objects_type in OBJECTS_TYPE]

    print("DICTS=", DICTS)

    # loop over the dictionary
    for d in DICTS:

        ##
        ## set up folders
        ##

        output_folder = os.path.join(OUTPUT_DIR, "{}agents_{}objects_{}shape".format(*list(d.values())))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ##
        ## loop over the number of simulations
        ##
                
        for s in range(NUM_OF_SIMS):

            ##
            ## commands list initialized
            ##

            commands = []

            ##
            ## simulation set up
            ##

            ## roscore
            commands.append("roscore")

            ## sim_base_station
            commands.append(f"roslaunch --wait primer sim_base_station_fastsam.launch rviz:={USE_RVIZ} num_of_obs:={d['num_of_objects']} gui_mission:=false objects_type:={d['objects_type']}")
            
            ## for each agent we add topics for onboard fastsam/mot/trajectory_generator
            topics_to_record = ""
            AGENTS_NAMES = [] # create a list of agent names
            for i in range(1, d["num_of_agents"]+1):
                AGENTS_NAMES.append(f"SQ0{i}s")
            for agent_name in AGENTS_NAMES:
                # return every other agent name
                other_agent_names = [x for x in AGENTS_NAMES if x != agent_name]
                if agent_name == "SQ01s":
                    is_constant_drift = False
                    is_linear_drift = True
                elif agent_name == "SQ02s":
                    is_constant_drift = False
                    is_linear_drift = False
                commands = agent_dependent_topics(commands, agent_name, other_agent_names, \
                                                  TIME_FASTSAM_AND_MOT, KAPPA_MOT, TIME_TRAJ_GEN, TIME_TAKEOFF, \
                                                    TIME_MOVE_TO_START, TIME_START_TRAJ, is_constant_drift, is_linear_drift)
                topics_to_record = topics_to_record + "/{}/world /{}/detections /{}/map/poses_only /{}/frame_align /{}/corrupted_world /{}/drift ".format(*[agent_name]*6)

            ## rosbag record
            sim_name = f"sim_{str(s).zfill(3)}"
            topics_to_record = topics_to_record + "/tf"
            commands.append(f"sleep "+str(TIME_START_TRAJ)+f" && cd {output_folder} && rosbag record {topics_to_record} -o {sim_name} __name:={sim_name}")
            
            ##
            ## tmux & sending commands
            ##

            session_name="flamealignment_sim"
            os.system("tmux kill-session -t" + session_name)
            os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

            for i in range(len(commands)):
                os.system('tmux split-window ; tmux select-layout tiled')
            
            for i in range(len(commands)):
                os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ commands[i]+'" '+' C-m')

            print("commands sent")
            time.sleep(3.0)

            ##
            ## wait until the sim is done
            ##

            time.sleep(SIM_DURATION + TIME_START_TRAJ)

            ##
            ## kill the sim
            ##

            os.system("rosnode kill "+sim_name)
            time.sleep(0.5)
            print("Killing the rest")
            os.system(KILL_ALL)
    
if __name__ == '__main__':
    main()