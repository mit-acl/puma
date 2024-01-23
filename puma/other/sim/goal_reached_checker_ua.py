#!/usr/bin/env python
# coding=utf-8

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
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
import numpy as np
from random import *
import tf2_ros
from numpy import linalg as LA

class GoalReachedCheck:

    def __init__(self, num_of_agents):

        rospy.sleep(3)

        # goal radius
        self.goal_radius = 0.5 #needs to be the same as the one in puma.yaml

        # number of agents
        self.num_of_agents = num_of_agents

        # is initialized?
        self.initialized = False

        # state and term_goal
        self.state_pos = np.empty([self.num_of_agents,3])
        self.term_goal_pos = np.empty([self.num_of_agents,3])

        # publisher init
        self.goal_reached = GoalReached()
        self.pubIsGoalReached = rospy.Publisher('/sim_all_agents_goal_reached', GoalReached, queue_size=1, latch=True)

        # is goal reached
        self.is_goal_reached = False

        # timer for goal reached checker
        self.timer_goal_reached_checker = rospy.Timer(rospy.Duration(0.1), self.goal_reached_checker)

        # subscribers
        self.sub_state = rospy.Subscriber("/SQ01s/state", State, self.stateCB)
        self.sub_term_goal = rospy.Subscriber("/SQ01s/term_goal", PoseStamped, self.term_goalCB)


    # goal reached checker
    def goal_reached_checker(self, timer):
        if not self.is_goal_reached and self.initialized:
            if (LA.norm(self.state_pos[0,:] - self.term_goal_pos[0,:]) > self.goal_radius):
                return
            else:
                self.is_goal_reached = True
                now = rospy.get_rostime()
                self.goal_reached.header.stamp = now
                self.goal_reached.is_goal_reached = True
                self.pubIsGoalReached.publish(self.goal_reached)
        else:
            if (LA.norm(self.state_pos[0,:] - self.term_goal_pos[0,:]) > self.goal_radius):
                self.is_goal_reached = False
                return

    def stateCB(self, data):
        self.state_pos[0,0:3] = np.array([data.pos.x, data.pos.y, data.pos.z])

    def term_goalCB(self, data):
        self.term_goal_pos[0,0:3] = np.array([data.pose.position.x, data.pose.position.y, 1.0])
        print("term goal received")
        self.initialized = True

def startNode(num_agents):

    GoalReachedCheck(num_agents)

    ## Subscribe to the state of each agent and the terminal goal
    # for i in range(num_agents):
    #     call_back_function = getattr(c, "SQ%02dstateCB" % (i+1))
    #     rospy.Subscriber("SQ%02ds/state" % (i+1), State, call_back_function)
    #     call_back_function = getattr(c, "SQ%02dterm_goalCB" % (i+1))
    #     rospy.Subscriber("SQ%02ds/term_goal" % (i+1), PoseStamped, call_back_function)
    
    rospy.spin()

if __name__ == '__main__':

    ##
    ## get params
    ##

    # num_of_agents = rospy.get_param("goal_reached_checker/num_of_agents")

    rospy.init_node('goalReachedCheck')
    startNode(1)