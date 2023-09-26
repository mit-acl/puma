#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Read the bag files and process the data
#  * The data to extract is:
#  *    (1) travel time - the time it took to reach the goal
#  *    (2) computation time - the time it took to compute the trajectory
#  *    (3) number of collisions
#  *    (4) fov rate - the percentage of time the drone keeps obstacles in its FOV when the drone is actually close to the obstacles
#  *    (5) continuous fov detection - the minimum, average, and maximum of coninuous detection the drone keeps obstacles in its FOV
#  *    (6) translational dynamic constraint violation rate - the percentage of time the drone violates the translational dynamic constraints
#  *    (7) yaw dynamic constraint violation rate - the percentage of time the drone violates the yaw dynamic constraints
#  *    (8) success rate - the percentage of time the drone reaches the goal without any collision or dynamic constraint violation
#  *    (9) accel trajectory smoothness
#  *    (10) jerk trajectory smoothness
#  *    (11) number of stops
#  *
#  * Use the following command to run this script:
#  * $ python process_ua_planner.py 2> >(grep -v -e TF_REPEATED_DATA -e buffer_core.cpp)
#  * grep is for removing the TF_REPEATED_DATA and buffer_core.cpp warnings
#  * -------------------------------------------------------------------------- */

import os
import sys
import rosbag
import rospy
import numpy as np
from statistics import mean
import tf_bag
from tf_bag import BagTfTransformer
from fov_detector import check_obst_is_in_FOV, visualization
import itertools
import math
import datetime
import yaml
import pprint
import rospkg
import pickle
import argparse
from scipy.spatial.transform import Rotation as Rot

def main():

    ## parse arguments
    parser = argparse.ArgumentParser(description="Process UA palanner data")
    parser.add_argument("-d", "--input_dir", help="Input directory.", default="/media/kota/T7/ua-planner/single-sims/bags")
    parser.add_argument("-s", "--use_success_rate", help="Use success rate.", default="true")
    args = parser.parse_args()

    ##
    ## Paramters
    ##

    INPUT_DIR = args.input_dir
    TOPICS_TO_UNPACK = "/tf /tf_static /obstacles_mesh /clock /trajs /sim_all_agents_goal_reached"
    TOPICS_TO_UNPACK_AGENT = "/{}/goal /{}/state /{}/primer/fov /{}/primer/best_solution_expert /{}/primer/best_solution_student /{}/term_goal /{}/primer/actual_traj /{}/primer/is_ready /{}/primer/log /{}/primer/pause_sim"
    PANTHER_YAML_PATH = rospkg.RosPack().get_path("primer") + "/param/primer.yaml"
    with open(PANTHER_YAML_PATH) as f:
        PANTHER_YAML_PARAMS = yaml.safe_load(f)
    AGENT_BBOX = np.array(PANTHER_YAML_PARAMS["drone_bbox"]) 
    DYN_OBSTACLE_BBOX = np.array(PANTHER_YAML_PARAMS["obstacle_bbox"])
    STATIC_OBSTACLE_BBOX = np.array([0.3, 2.5, 2.5]) # static obstacle bbox size is actually defined in dynamic_corridor.py
    BBOX_AGENT_AGENT = AGENT_BBOX / 2  + AGENT_BBOX / 2
    BBOX_AGENT_DYN_OBST = AGENT_BBOX / 2 + DYN_OBSTACLE_BBOX / 2
    BBOX_AGENT_STATIC_OBST = AGENT_BBOX / 2 + STATIC_OBSTACLE_BBOX / 2
    FOV_X_DEG = PANTHER_YAML_PARAMS["fov_x_deg"]
    FOV_Y_DEG = PANTHER_YAML_PARAMS["fov_y_deg"]
    FOV_DEPTH = PANTHER_YAML_PARAMS["fov_depth"]
    MAX_VEL = PANTHER_YAML_PARAMS["v_max"][0]
    MAX_ACC = PANTHER_YAML_PARAMS["a_max"][0]
    MAX_JERK = PANTHER_YAML_PARAMS["j_max"][0]
    MAX_DYAW = PANTHER_YAML_PARAMS["ydot_max"]
    DT = PANTHER_YAML_PARAMS["dc"]
    STOP_VEL_THRESHOLD = 0.01 # m/s
    DISCRETE_TIME_HZ = 10 # Hz

    ##
    ## Get simulation folders
    ##

    sim_folders = []
    for file in os.listdir(INPUT_DIR):
        if os.path.isdir(os.path.join(INPUT_DIR, file)):
            sim_folders.append(os.path.join(INPUT_DIR, file))

    ##
    ## Loop over each simulation folder (e.g. /3_obs/)
    ##

    for sim_folder in sim_folders:

        if sim_folder == "/media/kota/T7/ua-planner/single-sims/bags/pkls" \
            or sim_folder == "/media/kota/T7/ua-planner/single-sims/test0" \
            or sim_folder == "/media/kota/T7/ua-planner/single-sims/ua_data.txt":
            continue

        print("Processing " + sim_folder + "...")

        ##
        ## Data extraction preparation for all simulation folders
        ##

        # (1) travel time
        travel_time_list = []

        # (2) computation time
        computation_time_list = []

        # (3) number of collisions
        num_of_collisions_btwn_agents_list = []
        num_of_collisions_btwn_agents_and_obstacles_list = []

        # (4) fov rate
        fov_rate_list = []

        # (5) continuous fov detection
        continuous_fov_detection_list = []

        # (6) translational dynamic constraint violation rate
        translational_dynamic_constraint_violation_rate_list = []

        # (7) yaw dynamic constraint violation rate
        yaw_dynamic_constraint_violation_rate_list = []

        # (8) success rate
        success_rate_list = []

        # (9) accel trajectory smoothness
        accel_trajectory_smoothness_list = []

        # (10) jerk trajectory smoothness
        jerk_trajectory_smoothness_list = []

        # (11) number of stops
        num_of_stops_list = []

        # (12) moving direction fov rate
        moving_direction_fov_rate_list = []

        # (13) moving direction continuous fov detection
        moving_direction_continuous_fov_detection_list = []

        # parameters
        NUM_OF_AGENTS = 1 #TODO: hardcoded
        NUM_OF_OBSTACLES = 2 # TODO: hardcoded
        AGENTS_LIST = [f"SQ{str(i+1).zfill(2)}s" for i in range(NUM_OF_AGENTS)]
        OBSTACLES_LIST = [f"obs_{4000+i}" for i in range(NUM_OF_OBSTACLES)]
        
        ##
        ## Data extraction preparation for each simulation folder
        ##

        # (1) travel time
        
        # (2) computation time

        # (3) number of collisions
        num_of_collisions_btwn_agents = 0.0
        num_of_collisions_btwn_agents_and_obstacles = 0.0
        
        # (4) fov rate
        fov_rate = { agent: [] for agent in AGENTS_LIST }

        # (5) continuous fov detection (min, avg, max)
        continuous_fov_detection = 0.0

        # (6) translational dynamic constraint violation rate
        translational_dynamic_constraint_violation_rate = 0.0

        # (7) yaw dynamic constraint violation rate
        yaw_dynamic_constraint_violation_rate = 0.0

        # (9) accel trajectory smoothness & (10) jerk trajectory smoothness
        accel_traj_smoothness = 0.0
        jerk_traj_smoothness = 0.0

        # (11) number of stops
        num_of_stops = []

        # (12) moving direction fov rate
        moving_direction_fov_rate = { agent: [] for agent in AGENTS_LIST}

        # (13) moving direction continuous fov detection (min, avg, max)
        moving_direction_continuous_fov_detection = 0.0

        ##
        ## Read the bag files
        ##

        bag_files = [f for f in os.listdir(os.path.join(INPUT_DIR, sim_folder)) if f.endswith(".bag")]
        bag_files.sort()

        for bag_file in bag_files:

            print("Processing " + bag_file + "...")

            num_of_collisions_btwn_agents_and_obstacles = 0
            num_of_collisions_btwn_agents = 0
            
            topics = TOPICS_TO_UNPACK.split(" ")
            agent_names = []
            for i in range(NUM_OF_AGENTS):
                agent_name = f"SQ{str(i+1).zfill(2)}s"
                agent_names.append(agent_name)
                topics.extend(TOPICS_TO_UNPACK_AGENT.format(*[agent_name for i in range(10)]).split(" "))

            with rosbag.Bag(os.path.join(INPUT_DIR, sim_folder, bag_file)) as bag:

                ##
                ## (1) travel time
                ##

                print("(1) travel time")

                sim_start_times = {agent: 0 for agent in AGENTS_LIST}
                sim_start_times_t = {agent: 0 for agent in AGENTS_LIST}
                sim_end_time = 0
                sim_end_time_t = 0
                sim_tmp_pause_times = {agent: 0 for agent in AGENTS_LIST}
                sim_pause_times = {agent: 0 for agent in AGENTS_LIST}
                is_paused = {agent: False for agent in AGENTS_LIST}

                ## not sure why, but when you record rosbag using tmux, it won't stop recording when it's paused so add pause_sim topic to indicate the pause
                ## and according to the topic get total paused time for each agent
                for topic, msg, t in bag.read_messages(topics=topics):
                    if topic in [f"/{agent_name}/primer/pause_sim" for agent_name in agent_names]:
                        if msg.data and not is_paused[agent_name]:
                            sim_tmp_pause_times[agent_name] = t.to_sec()
                            is_paused[agent_name] = True
                        elif not msg.data and is_paused[agent_name]:
                            sim_pause_times[agent_name] += (t.to_sec() - sim_tmp_pause_times[agent_name])
                            is_paused[agent_name] = False

                    if  topic in [f"/{agent_name}/term_goal" for agent_name in agent_names]:
                        sim_start_times[agent_name] = msg.header.stamp.to_sec()
                        sim_start_times_t[agent_name] = t.to_sec()
                    
                    if topic == f"/sim_all_agents_goal_reached":
                        sim_end_time = msg.header.stamp.to_sec()
                        sim_end_time_t = t.to_sec()

                if sim_start_times == []:
                    print("SIMULATION WAS NOT PROPERLY STARTED!!!")
                    continue
                
                ##
                ## (2) computation time
                ##

                print("(2) computation time")

                computation_times = []

                for topic, msg, t in bag.read_messages(topics=topics):
                    if topic in [f"/{agent_name}/primer/log" for agent_name in agent_names]:
                        if msg.success_replanning:
                            computation_times.append(msg.ms_opt)

                ##
                ## (3) number of collisions (using tf)
                ##

                print("(3) number of collisions")

                bag_transformer = BagTfTransformer(bag)
                buffer_time = 5.0 # if you don't have buffer time, lookupTransform might not be able to find the transform (esp if sim_start_time is 0.0)
                sim_start_time = max([*sim_start_times.values(), buffer_time])

                print(f"sim_start_time: {sim_start_time}")

                sim_end_time = sim_end_time if sim_end_time != 0 else bag.get_end_time()
                discrete_times = np.linspace(sim_start_time, sim_end_time, int((sim_end_time - sim_start_time) * DISCRETE_TIME_HZ))

                # get combination of an agent and an agent and an agent and an obstacle
                agent_agent_pairs = list(itertools.combinations(AGENTS_LIST, 2))
                # get a pair of an agent and an obstacle
                agent_obstacle_pairs = list(itertools.product(AGENTS_LIST, OBSTACLES_LIST))

                ##
                ## This is for multagent case but I am not using it right now (see TODO below)
                ##

                # check if the agent-agent pair is in collision
                # if there's only one agent, then skip this part
                # if NUM_OF_AGENTS > 1:
                #     for t in discrete_times:
                #         for agent_agent_pair in agent_agent_pairs:
                #             agent1 = agent_agent_pair[0]
                #             agent2 = agent_agent_pair[1]
                #             agent1_pose = bag_transformer.lookupTransform("world", agent1, rospy.Time.from_sec(t))
                #             agent2_pose = bag_transformer.lookupTransform("world", agent2, rospy.Time.from_sec(t))
                            
                #             x_diff = abs(agent1_pose[0][0] - agent2_pose[0][0])
                #             y_diff = abs(agent1_pose[0][1] - agent2_pose[0][1])
                #             z_diff = abs(agent1_pose[0][2] - agent2_pose[0][2])

                #             if x_diff < BBOX_AGENT_AGENT[0] and y_diff < BBOX_AGENT_AGENT[1] and z_diff < BBOX_AGENT_AGENT[2]:
                #                 num_of_collisions_btwn_agents += 1
                #                 break
            
                #         # check if the agent-obstacle pair is in collision
                #         for agent_obstacle_pair in agent_obstacle_pairs:
                #             agent = agent_obstacle_pair[0]
                #             obstacle = agent_obstacle_pair[1]
                #             agent_pose = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))
                #             obstacle_pose = bag_transformer.lookupTransform("world", obstacle, rospy.Time.from_sec(t))

                #             x_diff = abs(agent_pose[0][0] - obstacle_pose[0][0])
                #             y_diff = abs(agent_pose[0][1] - obstacle_pose[0][1])
                #             z_diff = abs(agent_pose[0][2] - obstacle_pose[0][2])

                #             if x_diff < BBOX_AGENT_OBST[0] and y_diff < BBOX_AGENT_OBST[1] and z_diff < BBOX_AGENT_OBST[2]:
                #                 num_of_collisions_btwn_agents_and_obstacles += 1
                #                 break
                # else:

                ##
                ## TODO: Hardcoded for single agent with one dynamic obstacle and one static obstacle (the scenario where panther fails but puma works)
                ##

                for t in discrete_times:
                    # check if the agent-obstacle pair is in collision
                    for agent_obstacle_pair in agent_obstacle_pairs:
                        agent = agent_obstacle_pair[0]
                        obstacle = agent_obstacle_pair[1]
                        agent_pose = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))
                        obstacle_pose = bag_transformer.lookupTransform("world", obstacle, rospy.Time.from_sec(t))

                        x_diff = abs(agent_pose[0][0] - obstacle_pose[0][0])
                        y_diff = abs(agent_pose[0][1] - obstacle_pose[0][1])
                        z_diff = abs(agent_pose[0][2] - obstacle_pose[0][2])

                        if obstacle == "obs_4000": # dynamic obstacle

                            if x_diff < BBOX_AGENT_DYN_OBST[0] and y_diff < BBOX_AGENT_DYN_OBST[1] and z_diff < BBOX_AGENT_DYN_OBST[2]:
                                num_of_collisions_btwn_agents_and_obstacles += 1
                                
                                break
                        
                        if obstacle == "obs_4001": # static obstacle

                            if x_diff < BBOX_AGENT_STATIC_OBST[0] and y_diff < BBOX_AGENT_STATIC_OBST[1] and z_diff < BBOX_AGENT_STATIC_OBST[2]:
                                num_of_collisions_btwn_agents_and_obstacles += 1
                                break

                ##
                ## (4) fov rate & (5) continuous fov detection
                ##

                print("(4) fov rate & (5) continuous fov detection")

                flight_time = sim_end_time - sim_start_time
                discrete_times = np.linspace(sim_start_time, sim_end_time, int((sim_end_time - sim_start_time) * DISCRETE_TIME_HZ))
                dt = flight_time / len(discrete_times)
                for agent in AGENTS_LIST:
                    is_initialized = False
                    max_streak_in_FOV = 0
                    is_in_FOV_in_prev_timestep = False

                    for idx, t in enumerate(discrete_times):
                        is_in_FOV_in_current_time_step = False
                        agent_pos, agent_quat = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))

                        if not is_initialized:
                            prev_agent_pos = agent_pos
                            prev_agent_quat = agent_quat
                            is_initialized = True
                        is_there_obst_to_look_at = False
                        is_looking_at_least_one = False

                        for obstacle in OBSTACLES_LIST:
                            
                            # check if the obstacle is in the FOV of the agent
                            obst_pos, _ = bag_transformer.lookupTransform("world", obstacle, rospy.Time.from_sec(t))

                            # check if the agent is close to the obstacle
                            dist = np.linalg.norm(np.array(agent_pos) - np.array(obst_pos))
                            # if dist < FOV_DEPTH and prev_agent_pos != agent_pos and prev_agent_quat != agent_quat:
                            if dist < FOV_DEPTH and np.any(prev_agent_pos != agent_pos) and np.any(prev_agent_quat != agent_quat):
                                is_there_obst_to_look_at = True

                                #
                                # Note that check_obst_is_in_FOV takes care of rotation between camera and body but need to make sure
                                # that matches ones in static_transforms.launch
                                #

                                # debug
                                # visualization(agent_pos, agent_quat, obst_pos, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH, idx)
                                if check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH):
                                    is_looking_at_least_one = True

                            # check if the obstacle is in the FOV of the agent continuously
                            if check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH):
                                if not is_in_FOV_in_prev_timestep:
                                    max_streak_in_FOV = 1
                                else:
                                    max_streak_in_FOV += 1
                                    continuous_fov_detection = max(continuous_fov_detection, max_streak_in_FOV)
                                is_in_FOV_in_current_time_step = True
                            
                        prev_agent_pos, prev_agent_quat = agent_pos, agent_quat
                        is_in_FOV_in_prev_timestep = True if is_in_FOV_in_current_time_step else False
                        if not is_there_obst_to_look_at:
                            continue
                        elif is_looking_at_least_one:
                            fov_rate[agent].append(True)
                        else:
                            fov_rate[agent].append(False)
                
                # convert detection frame to seconds
                continuous_fov_detection = continuous_fov_detection * dt
                
                ##
                ## (6) translational dynamic constraint violation rate & (7) yaw dynamic constraint violation rate
                ##

                print("(6) translational dynamic constraint violation rate & (7) yaw dynamic constraint violation rate")

                topic_num = 0
                for topic, msg, t in bag.read_messages(topics=topics):

                    if  topic in [f"/{agent_name}/goal" for agent_name in agent_names]:

                        vel = np.linalg.norm(np.array([msg.v.x, msg.v.y, msg.v.z]))
                        acc = np.linalg.norm(np.array([msg.a.x, msg.a.y, msg.a.z]))
                        jerk = np.linalg.norm(np.array([msg.j.x, msg.j.y, msg.j.z]))
                        dyaw = float(msg.dpsi)

                        if vel > math.sqrt(3)*MAX_VEL + 0.1 or acc > math.sqrt(3)*MAX_ACC + 0.1 or jerk > math.sqrt(3)*MAX_JERK + 0.1:
                            translational_dynamic_constraint_violation_rate += 1
                        if dyaw > MAX_DYAW + 0.1:
                            yaw_dynamic_constraint_violation_rate += 1
                        
                        topic_num += 1
                
                ##
                ## (8) success rate
                ##

                print("(8) success rate")

                reached_goal = False
                for topic, msg, t in bag.read_messages(topics=topics):
                    if topic == f"/sim_all_agents_goal_reached":
                        reached_goal = True
                
                ##
                ## (9) accel trajectory smoothness & (10) jerk trajectory smoothness
                ##

                print("(9) accel trajectory smoothness & (10) jerk trajectory smoothness")

                for topic, msg, t in bag.read_messages(topics=topics):

                    if topic in [f"/{agent_name}/goal" for agent_name in agent_names]:

                        acc = np.linalg.norm(np.array([msg.a.x, msg.a.y, msg.a.z]))
                        jerk = np.linalg.norm(np.array([msg.j.x, msg.j.y, msg.j.z]))

                        accel_traj_smoothness += acc
                        jerk_traj_smoothness += jerk

                ##
                ## (11) number of stops
                ##

                print("(11) number of stops")
                tmp_num_of_stops = 0
                for agent_name in agent_names:
                    is_stopped = True
                    for topic, msg, t in bag.read_messages(topics=topics):
                        if topic == f"/{agent_name}/goal":

                            vel = np.linalg.norm(np.array([msg.v.x, msg.v.y, msg.v.z]))

                            if vel < STOP_VEL_THRESHOLD and not is_stopped:
                                tmp_num_of_stops += 1
                                is_stopped = True
                            elif vel > STOP_VEL_THRESHOLD and is_stopped:
                                is_stopped = False
                            else:
                                pass
                
                if tmp_num_of_stops > 0:
                    tmp_num_of_stops = tmp_num_of_stops - len(agent_names) # the last stop (when goal reached) is not counted
                    num_of_stops.append(tmp_num_of_stops / len(agent_names))
                else:
                    # this means agents didn't complete position exchange
                    pass

                ##
                ## (12) moving direction fov rate & (13) moving direction continuous fov detection
                ##

                print("(12) moving direction fov rate & (13) moving direction continuous fov detection")

                flight_time = sim_end_time - sim_start_time
                discrete_times = np.linspace(sim_start_time, sim_end_time, int((sim_end_time - sim_start_time) * DISCRETE_TIME_HZ))
                dt = flight_time / len(discrete_times)
                for agent in AGENTS_LIST:
                    is_initialized = False
                    max_streak_in_FOV = 0
                    is_in_FOV_in_prev_timestep = False

                    for idx, t in enumerate(discrete_times):
                        is_in_FOV_in_current_time_step = False
                        agent_pos, agent_quat = bag_transformer.lookupTransform("world", agent, rospy.Time.from_sec(t))

                        if not is_initialized:
                            prev_agent_pos = agent_pos
                            prev_agent_quat = agent_quat
                            is_initialized = True
                        looking_at_mv = False

                        # get unit vector of the agent's velocity
                        agent_vel = np.array(agent_pos) - np.array(prev_agent_pos)
                        agent_vel = agent_vel / np.linalg.norm(agent_vel)

                        # get a point in future trajectory
                        traj_point = np.array(agent_pos) + agent_vel * 0.5

                        if np.any(prev_agent_pos != agent_pos) and np.any(prev_agent_quat != agent_quat):

                            #
                            # Note that check_obst_is_in_FOV takes care of rotation between camera and body but need to make sure
                            # that matches ones in static_transforms.launch
                            #

                            if check_obst_is_in_FOV(agent_pos, agent_quat, traj_point, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH):
                                looking_at_mv = True

                        # check if the obstacle is in the FOV of the agent continuously
                        if check_obst_is_in_FOV(agent_pos, agent_quat, traj_point, FOV_X_DEG, FOV_Y_DEG, FOV_DEPTH):
                            if not is_in_FOV_in_prev_timestep:
                                max_streak_in_FOV = 1
                            else:
                                max_streak_in_FOV += 1
                                moving_direction_continuous_fov_detection = max(moving_direction_continuous_fov_detection, max_streak_in_FOV)
                            is_in_FOV_in_current_time_step = True
                            
                        prev_agent_pos, prev_agent_quat = agent_pos, agent_quat
                        is_in_FOV_in_prev_timestep = True if is_in_FOV_in_current_time_step else False
                        
                        if looking_at_mv:
                            moving_direction_fov_rate[agent].append(True)
                        else:
                            moving_direction_fov_rate[agent].append(False)
            
                # convert detection frame to seconds
                moving_direction_continuous_fov_detection = moving_direction_continuous_fov_detection * dt

            ##
            ## Data extraction per bag
            ##

            # (1) travel time
            travel_time_for_each_agent = {agent: sim_end_time_t - sim_start_times_t[agent] - sim_pause_times[agent] for agent in AGENTS_LIST}
            travel_time_list.append(mean(travel_time_for_each_agent.values()))

            # (2) computation time
            computation_time_list.extend(computation_times) # right now I am not including octopus search time

            # (3) number of collisions
            num_of_collisions_btwn_agents_list.append(num_of_collisions_btwn_agents)
            num_of_collisions_btwn_agents_and_obstacles_list.append(num_of_collisions_btwn_agents_and_obstacles)

            # (4) fov rate
            for agent in AGENTS_LIST:
                fov_rate_list.extend(fov_rate[agent])

            # (5) continuous fov detection (min, avg, max)
            continuous_fov_detection_list.append(continuous_fov_detection)

            # (6) translational dynamic constraints violation rate
            translational_dynamic_constraint_violation_rate = translational_dynamic_constraint_violation_rate / topic_num
            translational_dynamic_constraint_violation_rate_list.append(translational_dynamic_constraint_violation_rate)

            # (7) yaw dynamic constraint violation rate
            yaw_dynamic_constraint_violation_rate = yaw_dynamic_constraint_violation_rate / topic_num
            yaw_dynamic_constraint_violation_rate_list.append(yaw_dynamic_constraint_violation_rate)

            print(num_of_collisions_btwn_agents_and_obstacles)

            # (8) success rate
            success = True if reached_goal \
                and num_of_collisions_btwn_agents < 0.5 \
                and num_of_collisions_btwn_agents_and_obstacles <= 0.5 else False
            success_rate_list.append(success)

            # (9) accel trajectory smoothness & (10) jerk trajectory smoothness
            accel_traj_smoothness = accel_traj_smoothness * DT
            accel_trajectory_smoothness_list.append(accel_traj_smoothness)
            jerk_traj_smoothness = jerk_traj_smoothness * DT
            jerk_trajectory_smoothness_list.append(jerk_traj_smoothness)

            # (11) number of stops
            num_of_stops_list.extend(num_of_stops)

            # (12) moving direction fov rate
            for agent in AGENTS_LIST:
                moving_direction_fov_rate_list.extend(moving_direction_fov_rate[agent])
            
            # (13) moving direction continuous fov detection (min, avg, max)
            moving_direction_continuous_fov_detection_list.append(moving_direction_continuous_fov_detection)

        ##
        ## Data pickling
        ##

        # make directory
        if not os.path.exists(os.path.join(INPUT_DIR, sim_folder, "pkls")):
            os.makedirs(os.path.join(INPUT_DIR, sim_folder, "pkls"))

        pickling_folder = os.path.join(INPUT_DIR, sim_folder, "pkls")

        # (1) travel time
        with open(os.path.join(pickling_folder, "travel_time_list.pkl"), "wb") as f:
            pickle.dump(travel_time_list, f)
        
        # (2) computation time
        with open(os.path.join(pickling_folder, "computation_time_list.pkl"), "wb") as f:
            pickle.dump(computation_time_list, f)
        
        # (3) number of collisions
        with open(os.path.join(pickling_folder, "num_of_collisions_btwn_agents_list.pkl"), "wb") as f:
            pickle.dump(num_of_collisions_btwn_agents_list, f)
        
        with open(os.path.join(pickling_folder, "num_of_collisions_btwn_agents_and_obstacles_list.pkl"), "wb") as f:
            pickle.dump(num_of_collisions_btwn_agents_and_obstacles_list, f)
        
        # (4) fov rate
        with open(os.path.join(pickling_folder, "fov_rate_list.pkl"), "wb") as f:
            pickle.dump(fov_rate_list, f)
        
        # (5) continuous fov detection (min, avg, max)
        with open(os.path.join(pickling_folder, "continuous_fov_detection_list.pkl"), "wb") as f:
            pickle.dump(continuous_fov_detection_list, f)
        
        # (6) translational dynamic constraints violation rate
        with open(os.path.join(pickling_folder, "translational_dynamic_constraint_violation_rate_list.pkl"), "wb") as f:
            pickle.dump(translational_dynamic_constraint_violation_rate_list, f)
        
        # (7) yaw dynamic constraint violation rate
        with open(os.path.join(pickling_folder, "yaw_dynamic_constraint_violation_rate_list.pkl"), "wb") as f:
            pickle.dump(yaw_dynamic_constraint_violation_rate_list, f)
        
        # (8) success rate
        with open(os.path.join(pickling_folder, "success_rate_list.pkl"), "wb") as f:
            pickle.dump(success_rate_list, f)
        
        # (9) accel trajectory smoothness & (10) jerk trajectory smoothness
        with open(os.path.join(pickling_folder, "accel_trajectory_smoothness_list.pkl"), "wb") as f:
            pickle.dump(accel_trajectory_smoothness_list, f)
        
        with open(os.path.join(pickling_folder, "jerk_trajectory_smoothness_list.pkl"), "wb") as f:
            pickle.dump(jerk_trajectory_smoothness_list, f)
        
        # (11) number of stops
        with open(os.path.join(pickling_folder, "num_of_stops_list.pkl"), "wb") as f:
            pickle.dump(num_of_stops_list, f)

        # (12) moving direction fov rate
        with open(os.path.join(pickling_folder, "moving_direction_fov_rate_list.pkl"), "wb") as f:
            pickle.dump(moving_direction_fov_rate_list, f)
        
        # (13) moving direction continuous fov detection (min, avg, max)
        with open(os.path.join(pickling_folder, "moving_direction_continuous_fov_detection_list.pkl"), "wb") as f:
            pickle.dump(moving_direction_continuous_fov_detection_list, f)

        ##
        ## Data print per simulation environment (eg. 1_obs_1_agent)
        ##

        d_string     = f"date                                             :{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        sf_string    = f"simulation folder                                :{os.path.join(INPUT_DIR, sim_folder)}"
        tt_string    = f"travel time                                      :{round(mean(travel_time_list),3)} [s]"
        ct_string    = f"computational time                               :{round(mean(computation_time_list),3)} [ms]"
        ncba_string  = f"number of collisions btwn agents                 :{round(mean(num_of_collisions_btwn_agents_list),3)}"
        ncbao_string = f"number of collisions btwn agents and obstacles   :{round(mean(num_of_collisions_btwn_agents_and_obstacles_list),3)}"
        fr_string    = f"fov rate                                         :{round(mean(fov_rate_list),3)*100} [%]"
        cfd_string   = f"continuous fov detection                         :{round(mean(continuous_fov_detection_list),3)} [s]"
        mvfr_string  = f"moving direction fov rate                        :{round(mean(moving_direction_fov_rate_list),3)*100} [%]"
        mvcfd_string  = f"moving direction continuous fov detection        :{round(mean(moving_direction_continuous_fov_detection_list),3)}"
        tdcvr_string = f"translational dynamic constraint violation rate  :{round(mean(translational_dynamic_constraint_violation_rate_list),3)*100} [%]"
        ydcvr_string = f"yaw dynamic constraint violation rate            :{round(mean(yaw_dynamic_constraint_violation_rate_list),3)*100} [%]"
        sr_string    = f"success rate                                     :{round(mean(success_rate_list),3)*100} [%]"
        ats_string   = f"accel trajectory smoothness                      :{round(mean(accel_trajectory_smoothness_list),3)}"
        jts_string   = f"jerk trajectory smoothness                       :{round(mean(jerk_trajectory_smoothness_list),3)}"
        ns_string    = f"number of stops                                  :{round(mean(num_of_stops_list),3)}"
        
        print("\n")
        print("=============================================")
        print(sf_string)
        print(tt_string)
        print(ct_string)
        print(ncba_string)
        print(ncbao_string)
        print(fr_string)
        print(cfd_string)
        print(mvfr_string)
        print(mvcfd_string)
        print(tdcvr_string)
        print(ydcvr_string)
        print(sr_string)
        print(ats_string)
        print(jts_string)
        print(ns_string)
        print("=============================================")

        ##
        ## Save the data
        ##

        with open(os.path.join(INPUT_DIR, "ua_data.txt"), "a") as f:
            f.write("\n")
            f.write("=============================================\n")
            f.write(d_string + "\n")
            f.write(sf_string + "\n")
            f.write(tt_string + "\n")
            f.write(ct_string + "\n")
            f.write(ncba_string + "\n")
            f.write(ncbao_string + "\n")
            f.write(fr_string + "\n")
            f.write(cfd_string + "\n")
            f.write(mvfr_string + "\n")
            f.write(mvcfd_string + "\n")
            f.write(tdcvr_string + "\n")
            f.write(ydcvr_string + "\n")
            f.write(sr_string + "\n")
            f.write(ats_string + "\n")
            f.write(jts_string + "\n")
            f.write(ns_string + "\n")
            f.write("=============================================\n")
        
        with open(os.path.join(sim_folder, "ua_data.txt"), "a") as f:
            f.write("\n")
            f.write("=============================================\n")
            f.write(d_string + "\n")
            f.write(sf_string + "\n")
            f.write(tt_string + "\n")
            f.write(ct_string + "\n")
            f.write(ncba_string + "\n")
            f.write(ncbao_string + "\n")
            f.write(fr_string + "\n")
            f.write(cfd_string + "\n")
            f.write(mvfr_string + "\n")
            f.write(mvcfd_string + "\n")
            f.write(tdcvr_string + "\n")
            f.write(ydcvr_string + "\n")
            f.write(sr_string + "\n")
            f.write(ats_string + "\n")
            f.write(jts_string + "\n")
            f.write(ns_string + "\n")
            f.write("=============================================\n")

if __name__ == '__main__':
    main()