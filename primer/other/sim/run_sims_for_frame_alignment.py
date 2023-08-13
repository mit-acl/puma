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


def main():

    ##
    ## Arguments
    ##

    parser = argparse.ArgumentParser(description="Run simulations for frame alignment.")
    parser.add_argument("output_dir", help="Directory to save bags.")
    parser.add_argument("use_rviz", help="Whether to use rviz.")
    parser.add_argument("num_of_objects", help="Number of objects.")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    USE_RVIZ = args.use_rviz

    ##
    ## Simulation parameters
    ##

    # for dicts
    NUM_OF_AGENTS = [1]
    NUM_OF_OBJECTS = [args.num_of_objects]
    # OBJECTS_TYPE = ["pads", "random"]
    OBJECTS_TYPE = ["random"]

    # others
    NUM_OF_SIMS = 1
    SIM_DURATION = 60  # seconds
    AGNET_NAME = "SQ01s"
    USE_PERFECT_CONTROLLER = "true"
    USE_PERFECT_PREDICTION = "true"
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f primer & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"
    TOPICS_TO_RECORD = "/{}/world /{}/detections /{}/map/poses_only /tf /tf_static".format(*[AGNET_NAME]*3)

    ##
    ## MOT parameters
    ##
    
    KAPPA_MOT = 400

    ##
    ## Trajectory generator parameters
    ##

    TIME_TRAJ_GEN = 3.0 # if you don't wait until snap_sim is ready, it will generate an error "Could not read parameters."
    TIME_TAKEOFF = 5.0
    TIME_MOVE_TO_START = 15.0
    TIME_START_TRAJ = 25.0
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

            ## sim_basestation
            commands.append(f"roslaunch --wait primer sim_base_station_fastsam.launch rviz:={USE_RVIZ} num_of_obs:={d['num_of_objects']} gui_mission:=false objects_type:={d['objects_type']}")
            
            ## fastsam
            commands.append(f"sleep "+str(TIME_FASTSAM_AND_MOT)+" && roslaunch --wait primer fastsam.launch is_sim:=false")

            ## mot
            commands.append(f"sleep "+str(TIME_FASTSAM_AND_MOT)+f" && roslaunch --wait motlee_ros mapper.launch kappa:={KAPPA_MOT}")

            ## trajectory generator onboard
            commands.append(f"sleep "+str(TIME_TRAJ_GEN)+f" &&roslaunch --wait trajectory_generator onboard.launch")

            ## takeoff
            commands.append(f"sleep "+str(TIME_TAKEOFF)+" && rostopic pub -1 /globalflightmode snapstack_msgs/QuadFlightMode '{header: auto, mode: 4}'")

            ## move to the starting position trajectory generator
            commands.append(f"sleep "+str(TIME_MOVE_TO_START)+" && rostopic pub -1 /globalflightmode snapstack_msgs/QuadFlightMode '{header: auto, mode: 4}'")

            ## start trajectory generator
            commands.append(f"sleep "+str(TIME_START_TRAJ)+" && rostopic pub -1 /globalflightmode snapstack_msgs/QuadFlightMode '{header: auto, mode: 4}'")
            
            ## rosbag record
            sim_name = f"sim_{str(s).zfill(3)}"
            commands.append(f"sleep "+str(TIME_START_TRAJ)+f" && cd {output_folder} && rosbag record {TOPICS_TO_RECORD} -o {sim_name} __name:={sim_name}")
            
            ## rosbag record for multiple agents
            # agent_bag_recorders = []
            # for i in range(l):
            #     sim_name = f"sim_{str(s).zfill(3)}"
            #     agent_name = f"SQ{str(i+1).zfill(2)}s"
            #     recorded_topics = TOPICS_TO_RECORD.format(*[agent_name for i in range(9)])
            #     agent_bag_recorder = f"{agent_name}_{RECORD_NODE_NAME}"
            #     agent_bag_recorders.append(agent_bag_recorder)
            #     commands.append("sleep "+str(TIME_TAKEOFF)+" && cd "+folder_bags+" && rosbag record "+recorded_topics+" -o "+sim_name+"_"+agent_name+" __name:="+agent_bag_recorder)
            
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