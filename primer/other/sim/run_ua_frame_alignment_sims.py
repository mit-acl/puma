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
from run_many_sims import get_start_end_state, check_goal_reached

def write_readme_file(f, output_folder, d):
    f.write("\n")
    f.write("*******************************************************\n")
    f.write("folder:         {}\n".format(output_folder))
    f.write("data:           {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    f.write("num_of_agents:  {}\n".format(d["num_of_agents"]))
    f.write("num_of_objects: {}\n".format(d["num_of_objects"]))
    f.write("objects_type:   {}\n".format(d["objects_type"]))
    f.write("drift_type:     {}\n".format(d["drift_type"]))
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

def agent_dependent_topics(commands, agent_name, other_agent_names, kappa_mot, time_start_drift, drifts, x_start, y_start, z_start, yaw_start, fastsam_cb_frequency):
    
    """ Add topics that are agent dependent to commands """

    ## sim_onboard
    commands.append(f"roslaunch --wait primer sim_onboard.launch quad:={agent_name} veh:={agent_name[:2]} num:={agent_name[2:4]} x:={x_start} y:={y_start} z:=3 yaw:={yaw_start} perfect_controller:=true rviz:=false use_planner:=true  2> >(grep -v -e TF_REPEATED_DATA -e buffer)")

    ## fastsam (triggered by rosservice call /{agent_name}/fast_sam/start_drift)
    commands.append(f"roslaunch --wait primer fastsam.launch quad:={agent_name} fastsam_cb_frequency:={fastsam_cb_frequency} is_sim:=true")

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

    ## and at the same time start introducing drift (rosservice)
    commands.append(f"sleep "+str(time_start_drift)+f" && rosservice call /{agent_name}/pose_corrupter/start_drift")
    commands.append(f"sleep "+str(time_start_drift)+f" && rosservice call /{agent_name}/fast_sam/start_drift")

    return commands

class TimeKeeper:

    def __init__(self):
        self.current_time = 0.0

    def rostime_cb(self, msg):
        # print("msg.clock.secs: {}".format(msg.clock.secs))
        self.current_time = msg.clock.secs + msg.clock.nsecs*1e-9

def main():

    ##
    ## Arguments
    ##

    parser = argparse.ArgumentParser(description="Run simulations for frame alignment.")
    # parser.add_argument("-o", "--output_dir", help="Directory to save bags.", default="/media/kota/T7/ua-planner/multi-sims/ones_used_in_icra24/videos")
    parser.add_argument("-o", "--output_dir", help="Directory to save bags.", default="/media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos")
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
    OBJECTS_TYPE = ["random"]
    
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
    DRIFTS = [[cdx, cdy, 0.0, 0.0, 0.0, 0.0, True, False], \
                [0.0, 0.0, 0.0, ldx, ldy, ldyaw, False, True]]

    # others
    NUM_OF_SIMS = 1
    SIM_DURATION = 100  # seconds
    KILL_ALL = "killall -9 gazebo & killall -9 gzserver  & killall -9 gzclient & pkill -f primer & pkill -f gazebo_ros & pkill -f spawn_model & pkill -f gzserver & pkill -f gzclient  & pkill -f static_transform_publisher &  killall -9 multi_robot_node & killall -9 roscore & killall -9 rosmaster & pkill rmader_node & pkill -f tracker_predictor & pkill -f swarm_traj_planner & pkill -f dynamic_obstacles & pkill -f rosout & pkill -f behavior_selector_node & pkill -f rviz & pkill -f rqt_gui & pkill -f perfect_tracker & pkill -f rmader_commands & pkill -f dynamic_corridor & tmux kill-server & pkill -f perfect_controller & pkill -f publish_in_gazebo"
    CIRCLE_CENTER = [[0.0, 0.0], [0.0, 0.0]] # [m]
    VEN_DIAG_CENTER = [[1.0, 0.0], [-1.0, 0.0]] # [m]
    FASTSAM_CB_FREQUENCY = 1.0 # [Hz]

    ##
    ## MOT parameters
    ##
    
    KAPPA_MOT = 600

    ##
    ## Trajectory generator parameters
    ##

    TIME_START = 5.0
    TIME_SEND_GOAL = 30.0 
    TIME_START_DRIFT = 30.0

    ##
    ## Start and end states
    ##

    CIRCLE_RADIUS = 5.0 # [m]
    INITIAL_POSITIONS_SHAPE = "square" # "circle", "square", "line"

    ##
    ## make sure ROS (and related stuff) is not running
    ##

    os.system(KILL_ALL)

    ##
    ## we use PUMA
    ##

    os.system("sed -i '/uncertainty_aware:/s/^/#/g' $(rospack find primer)/param/primer.yaml")

    ##
    ## simulation loop
    ##

    # create a dictionary (cause we don't want a nested for loop)
    DICTS = [ {"num_of_agents": num_of_agents, "num_of_objects": num_of_objects, "objects_type": objects_type, "drifts": drifts} \
                for num_of_agents in NUM_OF_AGENTS for num_of_objects in NUM_OF_OBJECTS for objects_type in OBJECTS_TYPE for drifts in DRIFTS]

    # loop over the dictionary
    for dic_index, d in enumerate(DICTS):

        # if dic_index < 10:
        #     continue

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

            ## sim_base_station
            commands.append(f"roslaunch --wait primer sim_base_station_fastsam.launch rviz:={USE_RVIZ} num_of_obs:={d['num_of_objects']} gui_mission:=false objects_type:={d['objects_type']}")
            
            x_start_list, y_start_list, z_start_list, yaw_start_list, x_goal_list, y_goal_list, z_goal_list = get_start_end_state(d['num_of_agents'], CIRCLE_RADIUS, INITIAL_POSITIONS_SHAPE)

            ## for each agent we add topics for onboard fastsam/mot/trajectory_generator
            TOPIC_TO_RECORD = ""
            for agent_name, x_start, y_start, z_start, yaw_start in zip(AGENTS_NAMES, x_start_list, y_start_list, z_start_list, yaw_start_list):
                # return every other agent name
                other_agent_names = [x for x in AGENTS_NAMES if x != agent_name]

                # add topics
                commands = agent_dependent_topics(commands, agent_name, other_agent_names, \
                                                    KAPPA_MOT, TIME_START_DRIFT, d["drifts"], x_start, y_start, yaw_start, z_start, FASTSAM_CB_FREQUENCY)
                # add topics to record
                TOPIC_TO_RECORD = TOPIC_TO_RECORD + """/{}/drone_marker /{}/camera/fisheye1/image_raw /{}/goal /{}/world /{}/detections /{}/map/poses_only /{}/frame_align /{}/corrupted_world /{}/drift /{}/state \
                /{}/primer/fov /{}/primer/pause_sim /{}/primer/best_solution_expert /{}/primer/best_solution_student /{}/term_goal \
                /{}/primer/actual_traj /{}/primer/is_ready /{}/primer/log /{}/primer/obstacle_uncertainty /{}/primer/obstacle_uncertainty_values \
                /{}/primer/obstacle_sigma_values /{}/primer/obstacle_uncertainty_times /{}/primer/moving_direction_uncertainty_values /{}/primer/moving_direction_sigma_values \
                /{}/primer/moving_direction_uncertainty_times /{}/primer/alpha """.format(*[agent_name]*26)

                ## alsways use puma(primer)
                commands.append(f"sleep 2.0 && rosparam set /{agent_name}/primer/uncertainty_aware true")

            ## rosbag record
            sim_name = f"sim_{str(s).zfill(3)}"
            TOPIC_TO_RECORD = TOPIC_TO_RECORD + " /tf /tf_static /obstacles_mesh /clock /trajs /sim_all_agents_goal_reached"
            print("output_folder: {}".format(output_folder))
            commands.append(f"sleep "+str(TIME_START)+f" && cd {output_folder} && rosbag record {TOPIC_TO_RECORD} -o {sim_name} __name:={sim_name}")
            
            ## publish goal
            commands.append(f"sleep "+str(TIME_SEND_GOAL)+f" && roslaunch --wait primer pub_goal.launch x_goal_list:=\"{x_goal_list}\" y_goal_list:=\"{y_goal_list}\" z_goal_list:=\"{z_goal_list}\"")
            
            ## goal checker
            # commands.append(f"roslaunch --wait primer goal_reached_checker.launch num_of_agents:={d['num_of_agents']}")
            
            ## time keeper
            commands.append(f"roslaunch --wait primer time_keeper.launch sim_time:={SIM_DURATION}")

            ## term_goal_sender (keeps agents moving)
            for idx, agent_name in enumerate(AGENTS_NAMES):
                commands.append(f"roslaunch --wait primer term_goal_sender.launch quad:={agent_name} mode:={idx} circle_radius:={CIRCLE_RADIUS}")

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

            ##
            ## wait until the simulation ends (or goal reached)
            ##

            is_control_c_pressed = False
            while True:
                if check_goal_reached(): # it's really not goal reached but time_keeper publishs this topic when time is up
                    break

                # if Contrl-C is pressed, then break
                if rospy.is_shutdown():
                    is_control_c_pressed = True
                    break

                time.sleep(1.0)

            # kill the simulation
            os.system("rosnode kill "+sim_name)
            time.sleep(0.5)
            print("Killing the rest")
            os.system(KILL_ALL)
            time.sleep(10.0)

            # if Contrl-C is pressed, then end the simulation
            if is_control_c_pressed:
                return


    ## process data
    print("Processing data")

    ##
    ## tmux & sending commands
    ##

    proc_commands = []
    # proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/process_frame_alignment.py -d /media/kota/T7/ua-planner/multi-sims -s true")
    # proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/process_ua_planner.py -d /media/kota/T7/ua-planner/multi-sims -s true")
    # proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_frame_alignment.py -d /media/kota/T7/ua-planner")
    # proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_map.py -d /media/kota/T7/ua-planner")
    # proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_animation.py -d /media/kota/T7/ua-planner")
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_map_animation.py -d /media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos")
    proc_commands.append("python ~/Research/primer_ws/src/primer/primer/other/data_extraction/plot_2d_data_animation.py -d /media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos -p tracking")    

    session_name="processing"
    os.system("tmux kill-session -t" + session_name)
    os.system("tmux new -d -s "+str(session_name)+" -x 300 -y 300")

    for i in range(len(proc_commands)):
        os.system('tmux split-window ; tmux select-layout tiled')
    
    for i in range(len(proc_commands)):
        os.system('tmux send-keys -t '+str(session_name)+':0.'+str(i) +' "'+ proc_commands[i]+'" '+' C-m')

    print("proc_commands sent")

    ## uncomment params we change
    os.system("sed -i '/center_x/s/^#//g' $(rospack find trajectory_generator)/config/default.yaml")
    os.system("sed -i '/center_y/s/^#//g' $(rospack find trajectory_generator)/config/default.yaml")
    os.system("sed -i '/uncertainty_aware:/s/^#//g' $(rospack find primer)/param/primer.yaml")

if __name__ == '__main__':
    main()