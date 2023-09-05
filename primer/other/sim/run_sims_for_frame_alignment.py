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
import datetime

def write_readme_file(f, output_folder, d):
    f.write("\n")
    f.write("*******************************************************\n")
    f.write("folder:         {}\n".format(output_folder))
    f.write("data:           {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    f.write("num_of_agents:  {}\n".format(d["num_of_agents"]))
    f.write("num_of_objects: {}\n".format(d["num_of_objects"]))
    f.write("objects_type:   {}\n".format(d["objects_type"]))
    f.write("drift_type:     {}\n".format(d["drift_type"]))
    f.write("traj_type:      {}\n".format(d["traj_type"]))
    f.write("constant_translational_drift_offset: {}\n".format(d["drifts"][:2]))
    f.write("constant_rotational_drift_offset:    {}\n".format(d["drifts"][2]))
    f.write("linear_translational_drift_rate:     {}\n".format(d["drifts"][3:5]))
    f.write("linear_rotational_drift_rate:        {}\n".format(d["drifts"][5]))
    f.write("*******************************************************\n")

def get_drift(drifts, agent_name):
    
    if agent_name == "SQ01s": # hardcoded

        # constant drift params
        constant_drift_x=drifts[0]
        constant_drift_y=drifts[1]
        constant_drift_z=0.0
        constant_drift_roll=0.0
        constant_drift_pitch=0.0
        constant_drift_yaw=drifts[2]
        
        # linear drift params
        linear_drift_rate_x=drifts[3]
        linear_drift_rate_y=drifts[4]
        linear_drift_rate_z=0.0
        linear_drift_rate_roll=0.0
        linear_drift_rate_pitch=0.0
        linear_drift_rate_yaw=drifts[5]
    
        return constant_drift_x, constant_drift_y, constant_drift_z, constant_drift_roll, constant_drift_pitch, constant_drift_yaw, \
            linear_drift_rate_x, linear_drift_rate_y, linear_drift_rate_z, linear_drift_rate_roll, linear_drift_rate_pitch, linear_drift_rate_yaw

    elif agent_name == "SQ02s": # hardcoded

        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def agent_dependent_topics(commands, agent_name, other_agent_names, kappa_mot, time_traj_gen, \
                            time_takeoff, time_move_to_start, time_start_traj, drifts):
    
    """ Add topics that are agent dependent to commands """

    ## sim_onboard
    commands.append(f"roslaunch --wait primer sim_onboard.launch quad:={agent_name} veh:={agent_name[:2]} num:={agent_name[2:4]} x:={0.0} y:={0.0} z:=0.0 rviz:=false use_planner=false")

    ## fastsam (triggered by rosservice call /{agent_name}/pose_corrupter/start_drift)
    commands.append(f"roslaunch --wait primer fastsam.launch quad:={agent_name} is_sim:=true")

    ## mot
    commands.append(f"roslaunch --wait motlee_ros mapper.launch quad:={agent_name} kappa:={kappa_mot}")

    ## frame alignment
    commands.append(f"roslaunch --wait motlee_ros frame_aligner.launch quad1:={agent_name} quad2:={other_agent_names}")

    ## pose corrupter
    constant_drift_x, constant_drift_y, constant_drift_z, constant_drift_roll, constant_drift_pitch, constant_drift_yaw, \
    linear_drift_rate_x, linear_drift_rate_y, linear_drift_rate_z, linear_drift_rate_roll, linear_drift_rate_pitch, linear_drift_rate_yaw = get_drift(drifts, agent_name)
    
    if agent_name == "SQ01s": # hardcoded
        is_constant_drift = drifts[6]
        is_linear_drift = drifts[7]
    elif agent_name == "SQ02s":
        is_constant_drift = False
        is_linear_drift = False

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
    ## and at the same time start introducing drift (rosservice)
    commands.append(f"sleep "+str(time_start_traj)+f" && rosservice call /{agent_name}/pose_corrupter/start_drift")
    commands.append(f"sleep "+str(time_start_traj)+f" && rosservice call /{agent_name}/fast_sam/start_drift")

    return commands

def main():

    ##
    ## Arguments
    ##

    parser = argparse.ArgumentParser(description="Run simulations for frame alignment.")
    parser.add_argument("-o", "--output_dir", help="Directory to save bags.", default="/media/kota/T7/frame/sim/benchmarking")
    parser.add_argument("-v", "--use_rviz", help="Whether to use rviz.", default=False, type=bool)
    parser.add_argument("-n", "--num_of_objects", help="Number of objects.", default=10, type=int)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    USE_RVIZ = args.use_rviz

    ##
    ## Simulation parameters
    ##

    # for dicts
    NUM_OF_AGENTS = [2]
    NUM_OF_OBJECTS = [30] # needs to by synced with plot_anmation.py
    OBJECTS_TYPE = ["pads"]
    TRAJ_TYPE = ["venn_diagram"]
    
    # TODO: there's redandancy in the following two lists, but it's easier to implement this way
    cdx = 1.0 # constant drift x
    cdy = 1.0 # constant drift y
    cdyaw = 10.0 # constant drift yaw
    ldx = 0.05 # linear drift x
    ldy = 0.05 # linear drift y
    ldyaw = 0.05 # linear drift yaw

    # drift parameters [m, m, deg, m/s, m/s, deg/s, "is_constant_drift", "is_linear_drift"]
    # no drift
    # trans constant drift
    # rot constant drift
    # trans and rot constant drift
    # trans linear drift
    # rot linear drift
    # trans and rot linear drift
    # DRIFTS = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False], \
    #             [cdx, cdy, 0.0, 0.0, 0.0, 0.0, True, False], \
    #             [0.0, 0.0, cdyaw, 0.0, 0.0, 0.0, True, False], \
    #             [cdx, cdy, cdyaw, 0.0, 0.0, 0.0, True, False], \
    #             [0.0, 0.0, 0.0, ldx, ldy, 0.0, False, True], \
    #             [0.0, 0.0, 0.0, 0.0, 0.0, ldyaw, False, True], \
    #             [0.0, 0.0, 0.0, ldx, ldy, ldyaw, False, True]]

    DRIFTS = [[0.0, 0.0, 0.0, ldx, ldy, 0.0, False, True], \
                [0.0, 0.0, 0.0, 0.0, 0.0, ldyaw, False, True], \
                [0.0, 0.0, 0.0, ldx, ldy, ldyaw, False, True]]
    
    # others
    NUM_OF_SIMS = 1
    SIM_DURATION = 100  # seconds
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f primer & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"
    CIRCLE_CENTER = [[0.0, 0.0], [0.0, 0.0]] # [m]
    VEN_DIAG_CENTER = [[1.0, 0.0], [-1.0, 0.0]] # [m]

    ##
    ## MOT parameters
    ##
    
    KAPPA_MOT = 600

    ##
    ## Trajectory generator parameters
    ##

    TIME_TRAJ_GEN = 3.0
    TIME_TAKEOFF = 10.0
    TIME_MOVE_TO_START = 25.0
    TIME_START_TRAJ = 40.0

    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## simulation loop
    ##

    # create a dictionary (cause we don't want a nested for loop)
    DICTS = [ {"num_of_agents": num_of_agents, "num_of_objects": num_of_objects, "objects_type": objects_type, "traj_type": traj_type, "drifts": drifts} \
                for num_of_agents in NUM_OF_AGENTS for num_of_objects in NUM_OF_OBJECTS for objects_type in OBJECTS_TYPE for traj_type in TRAJ_TYPE for drifts in DRIFTS]

    ## comment out params we change
    os.system("sed -i '/center_x/s/^/#/g' $(rospack find trajectory_generator)/config/default.yaml")
    os.system("sed -i '/center_y/s/^/#/g' $(rospack find trajectory_generator)/config/default.yaml")

    # loop over the dictionary
    for dic_index, d in enumerate(DICTS):

        if dic_index != 0:
            continue

        print("####### Case {} #######".format(dic_index))

        ## add drift type

        if not d["drifts"][6] and not d["drifts"][7]:
            d["drift_type"] = "no_drift"
        elif d["drifts"][6] and not d["drifts"][7]:
            d["drift_type"] = "constant"
        elif not d["drifts"][6] and d["drifts"][7]:
            d["drift_type"] = "linear"
        elif d["drifts"][6] and d["drifts"][7]:
            raise Exception("Drift type not implemented yet")

        ##
        ## set up folders
        ##

        # output folder
        output_folder = os.path.join(OUTPUT_DIR, "case-{}".format(dic_index))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # write a readme file to explain the parameters for each simulation case
        with open(os.path.join(output_folder, "README.md"), "a") as f:
            write_readme_file(f, output_folder, d)
        with open(os.path.join(OUTPUT_DIR, "README.md"), "a") as f:
            write_readme_file(f, output_folder, d)
        
        ##
        ## loop over the number of simulations
        ##
                
        for s in range(NUM_OF_SIMS):

            ## commands list initialized
            commands = []

            ##
            ## simulation set up
            ##
            
            ## roscore
            commands.append("roscore")
            
            ## publish params for trajectory_generator
            AGENTS_NAMES = [] # create a list of agent names
            for i in range(1, d["num_of_agents"]+1):
                AGENTS_NAMES.append(f"SQ0{i}s")

            ## set params for trajectory_generator
            center = CIRCLE_CENTER if d["traj_type"] == "circle" else VEN_DIAG_CENTER
            for idx, agent_name in enumerate(AGENTS_NAMES):
                commands.append(f"sleep 3.0 && rosparam set /{agent_name}/trajectory_generator/center_x {int(center[idx][0])}")
                commands.append(f"sleep 3.0 && rosparam set /{agent_name}/trajectory_generator/center_y {int(center[idx][1])}")

            ## sim_base_station
            commands.append(f"roslaunch --wait primer sim_base_station_fastsam.launch rviz:={USE_RVIZ} num_of_obs:={d['num_of_objects']} gui_mission:=false objects_type:={d['objects_type']}")
            
            ## for each agent we add topics for onboard fastsam/mot/trajectory_generator
            topics_to_record = ""
            for agent_name in AGENTS_NAMES:
                # return every other agent name
                other_agent_names = [x for x in AGENTS_NAMES if x != agent_name]

                # add topics
                commands = agent_dependent_topics(commands, agent_name, other_agent_names, \
                                                    KAPPA_MOT, TIME_TRAJ_GEN, TIME_TAKEOFF, \
                                                    TIME_MOVE_TO_START, TIME_START_TRAJ, \
                                                    d["drifts"])    
                # add topics to record
                topics_to_record = topics_to_record + "/{}/world /{}/state /{}/detections /{}/map/poses_only /{}/frame_align /{}/corrupted_world /{}/drift ".format(*[agent_name]*7)

            ## rosbag record
            sim_name = f"sim_{str(s).zfill(3)}"
            topics_to_record = topics_to_record + "/tf"
            commands.append(f"sleep "+str(TIME_START_TRAJ-10)+f" && cd {output_folder} && rosbag record {topics_to_record} -o {sim_name} __name:={sim_name}")
            
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

    ## process data
    print("Processing data")


    ##
    ## tmux & sending commands
    ##

    proc_commands = []
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/process_frame_alignment.py -d /media/kota/T7/frame/sim/benchmarking")
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_frame_alignment.py -d /media/kota/T7/frame/sim/benchmarking")
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_animation.py -d /media/kota/T7/frame/sim/benchmarking")

    session_name="processing"
    os.system("tmux kill-session -t" + session_name)
    os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

    for i in range(len(proc_commands)):
        os.system('tmux split-window ; tmux select-layout tiled')
    
    for i in range(len(proc_commands)):
        os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ proc_commands[i]+'" '+' C-m')

    print("proc_commands sent")

    ##
    ## wait until the sim is done
    ##

    time.sleep(500.0)

    ##
    ## kill the sim
    ##

    os.system("rosnode kill "+sim_name)
    time.sleep(0.5)
    print("Killing the rest")
    os.system(KILL_ALL)

    
    ## uncomment params we change
    os.system("sed -i '/center_x/s/^#//g' $(rospack find trajectory_generator)/config/default.yaml")
    os.system("sed -i '/center_y/s/^#//g' $(rospack find trajectory_generator)/config/default.yaml")

if __name__ == '__main__':
    main()