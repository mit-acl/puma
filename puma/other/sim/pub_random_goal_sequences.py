#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  *
#  * Publish goal to all the agents when they are all ready
#  * -------------------------------------------------------------------------- */

import math
import os
import sys
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State
from panther_msgs.msg import GoalReached
from panther_msgs.msg import IsReady
import numpy as np
from random import *
import tf2_ros
from numpy import linalg as LA

class ReadyCheck:

    def __init__(self, x_goal_list, y_goal_list, z_goal_list, n_agents):

        # number of agents
        assert len(x_goal_list) == len(y_goal_list) == len(z_goal_list)
        self.num_of_agents = n_agents

        # goal lists
        self.x_goal_list = x_goal_list
        self.y_goal_list = y_goal_list
        self.z_goal_list = z_goal_list

        # goal counter
        self.goal_cnt = 0

        # subscribers
        self.sub_goal_reached = rospy.Subscriber("/sim_all_agents_goal_reached", GoalReached, self.goal_reachedCB)
        self.sub_is_ready = rospy.Subscriber("/SQ01s/puma/is_ready", IsReady, self.is_ready_checker)

        # publishers
        self.pub_goal = rospy.Publisher("/SQ01s/term_goal", PoseStamped, queue_size=1)

    # ready checker
    def is_ready_checker(self, timer):
            rospy.sleep(1)
            self.publish_goal()

    def goal_reachedCB(self, data):
        if data.is_goal_reached:
            if self.goal_cnt < len(self.x_goal_list):
                rospy.sleep(1)
                self.publish_goal()
            else:
                print("All goals are reached")
                
                # kill rosbag
                os.system("rosnode kill bag_recorder")
                rospy.sleep(1)

                # kill tmux
                os.system("tmux kill-server")
                
                rospy.signal_shutdown("All goals are reached")


    # publish goal
    def publish_goal(self):
        ## publish goal
        x, y, z = self.x_goal_list[self.goal_cnt], self.y_goal_list[self.goal_cnt], self.z_goal_list[self.goal_cnt]
        ## TODO: may need to change the goal orientation
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)
        self.goal_cnt = self.goal_cnt + 1

def startNode(x_goal_list, y_goal_list, z_goal_list, n_agents):

    ReadyCheck(x_goal_list, y_goal_list, z_goal_list, n_agents)
    rospy.spin()

if __name__ == '__main__':

    ##
    ## get params
    ##

    x_goal_list = rospy.get_param("/pub_goal/x_goal_list")
    y_goal_list = rospy.get_param("/pub_goal/y_goal_list")
    z_goal_list = rospy.get_param("/pub_goal/z_goal_list")
    n_agents = rospy.get_param("/pub_goal/n_agents")

    print("x_goal_list: ", x_goal_list)
    print("y_goal_list: ", y_goal_list)
    print("z_goal_list: ", z_goal_list)

    if x_goal_list == [0] and y_goal_list == [0] and z_goal_list == [0]:

        x_goal_list, y_goal_list, z_goal_list = [], [], []

        # set seed
        seed(2)

        print("No goal is provided, create a random goal")
        n_goals = 10
        for i in range(n_goals):
            x_goal_list.append(uniform(-5, 5))
            y_goal_list.append(uniform(-5, 5))
            z_goal_list.append(1.0)

    rospy.init_node('sendGoal')
    startNode(x_goal_list, y_goal_list, z_goal_list, n_agents)